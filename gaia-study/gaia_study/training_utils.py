import json
import logging
import os
import re
import subprocess
from typing import Tuple, Optional

from app.config import Config
from app.utils.tokenizer import count_tokens
from app.utils.mcp_client import ai_execute as mcp_ai_execute, ai_read as mcp_ai_read, ai_write as mcp_ai_write

logger = logging.getLogger("GAIA.TrainingUtils")

def get_base_model_name(config: Config) -> Optional[str]:
    """Parses docker-compose.yml to get the filename of the base model."""
    try:
        # Prefer MCP read of compose for auditing, fallback to local file read
        content = None
        try:
            r = mcp_ai_read(config.DOCKER_COMPOSE_PATH)
            if r.get("ok"):
                content = r.get("content")
        except Exception:
            content = None
        if content is None:
            with open(config.DOCKER_COMPOSE_PATH, 'r') as f:
                content = f.read()
        
        # Use regex to find the model path mounted to /models/gaia-prime.gguf
        match = re.search(r'- .*[/\\_]([^/\\_]+.gguf):/models/gaia-prime.gguf', content)
        if match:
            logger.info(f"Found base model: {match.group(1)}")
            return match.group(1)
    except Exception as e:
        logger.error(f"Error reading docker-compose.yml: {e}")
    return None

def check_for_training_delta(config: Config) -> Tuple[bool, int, str]:
    """
    Checks for a difference between the main dataset and the training log.

    Returns:
        A tuple containing: (needs_training, token_count, delta_file_path)
    """
    base_model_name = get_base_model_name(config)
    if not base_model_name:
        return False, 0, ""

    dataset_path = os.path.join(config.KNOWLEDGE_DIR, "gaia_dataset.jsonl")
    log_path = os.path.join(config.KNOWLEDGE_DIR, "training_log.json")
    delta_path = os.path.join(config.KNOWLEDGE_DIR, "gaia_delta.jsonl")

    if not os.path.exists(dataset_path):
        return False, 0, ""

    with open(dataset_path, 'r') as f:
        full_dataset = [json.loads(line) for line in f]

    trained_data = {}
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            trained_data = json.load(f)

    # Check if the model has changed since the last training run
    last_trained_model = trained_data.get("last_trained_model")
    if last_trained_model != base_model_name:
        logger.info(f"Base model has changed from {last_trained_model} to {base_model_name}. Full retraining is needed.")
        with open(delta_path, 'w') as f:
            for item in full_dataset:
                f.write(json.dumps(item) + '\n')
        token_count = sum(count_tokens(item['text']) for item in full_dataset)
        return True, token_count, delta_path

    # Identify new data since the last run
    trained_entries = set(trained_data.get("trained_entries", []))
    new_entries = [entry for entry in full_dataset if entry['text'] not in trained_entries]

    if not new_entries:
        return False, 0, ""

    with open(delta_path, 'w') as f:
        for item in new_entries:
            f.write(json.dumps(item) + '\n')
    
    token_count = sum(count_tokens(item['text']) for item in new_entries)
    return True, token_count, delta_path

def get_next_model_version(config: Config, full_retrain: bool) -> str:
    """
    Determines the next version number for the fine-tuned model.
    """
    base_model_name = get_base_model_name(config).replace(".gguf", "")
    lora_dir = config.LORA_ADAPTERS_DIR
    
    versions = [d for d in os.listdir(lora_dir) if d.startswith(base_model_name)]
    if not versions:
        return f"{base_model_name}_1.0"

    latest_version = sorted(versions)[-1]
    major, minor = map(int, latest_version.split('_')[-1].split('.'))

    if full_retrain:
        return f"{base_model_name}_{major + 1}.0"
    else:
        return f"{base_model_name}_{major}.{minor + 1}"

def convert_to_gguf(model_path: str, output_path: str):
    """Converts a Hugging Face model to GGUF format."""
    try:
        # Prefer MCP execution so conversions can be audited / sandboxed.
        command = f"python -m llama_cpp.convert --model_dir {pipes.quote(model_path)} --outfile {pipes.quote(output_path)} --outtype q4_k_m"
        try:
            r = mcp_ai_execute(command, timeout=600, shell=True, dry_run=False)
            if not r.get("ok"):
                raise RuntimeError(r.get("error") or "MCP conversion failed")
            logger.info(f"Successfully converted model to GGUF at {output_path}")
        except Exception:
            # Fall back to local execution
            subprocess.run([
                "python", "-m", "llama_cpp.convert",
                "--model_dir", model_path,
                "--outfile", output_path,
                "--outtype", "q4_k_m"
            ], check=True)
            logger.info(f"Successfully converted model to GGUF at {output_path}")
    except Exception as e:
        logger.error(f"Failed to convert model to GGUF: {e}")

def update_training_log(config: Config, new_entries: list, new_model_name: str):
    """
    Updates the training log with the newly trained data.
    """
    log_path = os.path.join(config.KNOWLEDGE_DIR, "training_log.json")
    
    trained_data = {}
    try:
        r = mcp_ai_read(log_path)
        if r.get("ok") and r.get("content"):
            trained_data = json.loads(r.get("content"))
        elif os.path.exists(log_path):
            with open(log_path, 'r') as f:
                trained_data = json.load(f)
    except Exception:
        trained_data = {}

    trained_entries = set(trained_data.get("trained_entries", []))
    for entry in new_entries:
        trained_entries.add(entry['text'])
    
    trained_data["last_trained_model"] = get_base_model_name(config)
    trained_data["trained_entries"] = list(trained_entries)
    trained_data["last_fine_tuned_model"] = new_model_name

    try:
        mcp_ai_write(log_path, json.dumps(trained_data, indent=4))
    except Exception:
        with open(log_path, 'w') as f:
            json.dump(trained_data, f, indent=4)


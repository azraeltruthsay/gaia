"""
gaia-study/gaia_study/adapter_to_gguf.py — Post-training GGUF LoRA conversion

Converts safetensors LoRA adapters to GGUF format for loading via gaia_cpp.
Called automatically after QLoRA training completes.

Requires:
  - llama.cpp convert_lora_to_gguf.py (cloned to /tmp/llama_cpp_conv/)
  - gguf Python package (pip installed)
  - torch (already in gaia-study)

The converter reads adapter_config.json to find the base model path,
then produces adapter.gguf in the same directory.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("GAIA.AdapterToGGUF")

# llama.cpp converter script location
CONVERTER_DIR = "/tmp/llama_cpp_conv"
CONVERTER_SCRIPT = f"{CONVERTER_DIR}/convert_lora_to_gguf.py"
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_CPP_TAG = os.environ.get("LLAMA_CPP_TAG", "b8250")


def _ensure_converter() -> bool:
    """Ensure the llama.cpp converter is available."""
    if Path(CONVERTER_SCRIPT).exists():
        return True

    logger.info("Cloning llama.cpp %s for LoRA converter...", LLAMA_CPP_TAG)
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", LLAMA_CPP_TAG,
             LLAMA_CPP_REPO, CONVERTER_DIR],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.error("Failed to clone llama.cpp: %s", result.stderr)
            return False
        return Path(CONVERTER_SCRIPT).exists()
    except Exception as e:
        logger.error("Clone failed: %s", e)
        return False


def convert_adapter_to_gguf(adapter_dir: str,
                            base_model_path: str = None,
                            output_filename: str = "adapter.gguf") -> str:
    """
    Convert a safetensors LoRA adapter to GGUF format.

    Args:
        adapter_dir: Directory containing adapter_config.json + adapter_model.safetensors
        base_model_path: Path to the base model (auto-detected from adapter_config if None)
        output_filename: Name for the output GGUF file

    Returns:
        Path to the output GGUF file, or empty string on failure.
    """
    adapter_path = Path(adapter_dir)

    # Validate input
    config_file = adapter_path / "adapter_config.json"
    if not config_file.exists():
        logger.error("No adapter_config.json in %s", adapter_dir)
        return ""

    safetensors_file = adapter_path / "adapter_model.safetensors"
    if not safetensors_file.exists():
        logger.error("No adapter_model.safetensors in %s", adapter_dir)
        return ""

    # Auto-detect base model from adapter config
    if base_model_path is None:
        with open(config_file) as f:
            config = json.load(f)
        base_model_path = config.get("base_model_name_or_path", "")
        if not base_model_path or not Path(base_model_path).exists():
            logger.error("Base model not found: %s", base_model_path)
            return ""

    # Ensure converter is available
    if not _ensure_converter():
        logger.error("llama.cpp converter not available")
        return ""

    output_path = adapter_path / output_filename

    logger.info("Converting %s → %s (base: %s)",
                adapter_dir, output_path, base_model_path)

    try:
        result = subprocess.run(
            [sys.executable, CONVERTER_SCRIPT,
             "--outfile", str(output_path),
             str(adapter_dir),
             "--base", str(base_model_path)],
            capture_output=True, text=True, timeout=300,
            cwd=CONVERTER_DIR,
        )

        if result.returncode != 0:
            logger.error("Conversion failed:\nstdout: %s\nstderr: %s",
                         result.stdout[-500:], result.stderr[-500:])
            return ""

        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info("GGUF adapter created: %s (%.1f MB)", output_path, size_mb)
            return str(output_path)
        else:
            logger.error("Converter ran but output file not found: %s", output_path)
            return ""

    except subprocess.TimeoutExpired:
        logger.error("Conversion timed out (300s)")
        return ""
    except Exception as e:
        logger.error("Conversion failed: %s", e)
        return ""

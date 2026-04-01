#!/usr/bin/env python3
"""
Train CodeMind code_replace adapter — teaches the model to generate
accurate OLD_TEXT/NEW_TEXT patches for code editing.

Uses QLoRA subprocess isolation for deterministic VRAM release.
Designed to run inside gaia-study container with GPU access.

Usage (inside container):
    python /gaia/GAIA_Project/scripts/train_code_replace.py

Or from host:
    docker exec gaia-study python /gaia/GAIA_Project/scripts/train_code_replace.py
"""

import json
import logging
import multiprocessing
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Paths (inside container: /models, /gaia/GAIA_Project, etc.)
BASE_MODEL = os.getenv(
    "BASE_MODEL_PATH",
    "/models/Huihui-Qwen3-8B-GAIA-Prime-adaptive",
)
TRAINING_DATA = "/gaia/GAIA_Project/knowledge/curricula/training_data/code_replace_v1.json"
ADAPTER_DIR = "/models/lora_adapters/tier3_session/code_replace_v1"
ADAPTER_NAME = "code_replace_v1"

# Training hyperparams — tuned for 36 replace-format samples on RTX 5080 16GB
TRAINING_CONFIG = {
    "base_model_path": BASE_MODEL,
    "adapter_dir": ADAPTER_DIR,
    "adapter_name": ADAPTER_NAME,
    # QLoRA NF4 quantization (load to CPU, quantize, migrate to GPU)
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    # LoRA config — match code_skill_v1 for consistency
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    # Training params
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "max_steps": 200,          # More steps for format learning
    "warmup_steps": 10,
    "target_loss": 0.10,       # Slightly higher target — format precision matters more
    "convergence_patience": 5,  # Be patient for stable convergence
    "num_train_epochs": None,   # Use max_steps, not epochs
    "max_training_time": 900,   # 15 min max
    # No resume — fresh adapter
    "resume_from": None,
    # Samples populated below
    "samples": [],
}


def load_training_data() -> list:
    """Load and transform training samples into instruction/output format."""
    with open(TRAINING_DATA) as f:
        raw_samples = json.load(f)

    logger.info("Loaded %d raw samples from %s", len(raw_samples), TRAINING_DATA)
    return raw_samples


def main():
    logger.info("=" * 60)
    logger.info("CodeMind code_replace_v1 Adapter Training")
    logger.info("=" * 60)
    logger.info("Base model: %s", BASE_MODEL)
    logger.info("Adapter output: %s", ADAPTER_DIR)

    # Verify base model exists
    if not os.path.isdir(BASE_MODEL):
        logger.error("Base model not found: %s", BASE_MODEL)
        sys.exit(1)

    # Load training data
    samples = load_training_data()
    if not samples:
        logger.error("No training samples found")
        sys.exit(1)

    TRAINING_CONFIG["samples"] = samples
    logger.info("Training with %d samples", len(samples))

    # Create adapter output directory
    os.makedirs(ADAPTER_DIR, exist_ok=True)

    # Import subprocess training (all CUDA imports happen in child process)
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/gaia-common")
    from gaia_study.training_subprocess import run_training, PROGRESS_FILE

    # Use multiprocessing spawn for clean CUDA isolation
    ctx = multiprocessing.get_context("spawn")

    logger.info("Spawning training subprocess...")
    logger.info("Progress file: %s", PROGRESS_FILE)

    start_time = time.time()
    proc = ctx.Process(target=run_training, args=(TRAINING_CONFIG,))
    proc.start()

    # Monitor progress
    last_step = -1
    while proc.is_alive():
        time.sleep(5)
        try:
            with open(PROGRESS_FILE) as f:
                progress = json.load(f)
            state = progress.get("state", "unknown")
            step = progress.get("step", 0)
            total = progress.get("total_steps", 0)
            loss = progress.get("loss", 0)

            if step != last_step or state != "training":
                elapsed = time.time() - start_time
                logger.info(
                    "[%s] step %d/%d, loss=%.4f, elapsed=%.0fs",
                    state, step, total, loss, elapsed,
                )
                last_step = step
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    proc.join()
    elapsed = time.time() - start_time

    # Read final progress
    try:
        with open(PROGRESS_FILE) as f:
            final = json.load(f)
        state = final.get("state", "unknown")
        loss = final.get("loss", 0)
        steps = final.get("step", 0)
    except Exception:
        state = "unknown"
        loss = 0
        steps = 0

    if proc.exitcode == 0 and state == "completed":
        logger.info("=" * 60)
        logger.info("Training COMPLETE")
        logger.info("  Steps: %d", steps)
        logger.info("  Final loss: %.4f", loss)
        logger.info("  Duration: %.0fs", elapsed)
        logger.info("  Adapter: %s", ADAPTER_DIR)
        logger.info("=" * 60)

        # Write metadata
        metadata = {
            "name": ADAPTER_NAME,
            "version": "1.0.0",
            "display_name": "Code Replace V1",
            "description": f"Code replace format adapter — {len(samples)} OLD_TEXT/NEW_TEXT examples",
            "tier": 3,
            "pillar": "cognition",
            "rank": 16,
            "alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "training": {
                "method": "qlora",
                "samples": len(samples),
                "steps": steps,
                "learning_rate": 2e-4,
                "batch_size": 1,
                "final_loss": loss,
                "duration_seconds": elapsed,
                "source_documents": [
                    {"path": TRAINING_DATA, "samples": len(samples)},
                ],
                "incremental": False,
                "resumed_from": None,
            },
            "governance": {
                "requires_approval": True,
                "safety_checked": False,
                "restrictions": [],
            },
            "compatibility": {
                "conflicts_with": [],
                "requires": [],
            },
            "usage": {
                "load_count": 0,
                "total_tokens_generated": 0,
            },
            "tags": ["code", "replace", "patch", "codemind"],
            "activation_triggers": [],
        }
        with open(os.path.join(ADAPTER_DIR, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Metadata written to %s/metadata.json", ADAPTER_DIR)
    else:
        logger.error("Training FAILED (exit code %d, state=%s)", proc.exitcode, state)
        if state == "failed":
            logger.error("Error: %s", final.get("error", "unknown"))
        sys.exit(1)


if __name__ == "__main__":
    main()

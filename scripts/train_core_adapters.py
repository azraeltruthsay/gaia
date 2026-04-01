#!/usr/bin/env python3
"""
Train Core-tier adapters — conversational_v1 and code_replace_v1 for Core (4B).

The same curricula used for Prime, retrained on Core's base model so both tiers
can use the adapters for generation and observation.

Usage:
    python training_lifecycle.py -- docker exec gaia-study python /gaia/GAIA_Project/scripts/train_core_adapters.py
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

# ── Paths ──────────────────────────────────────────────────────────────────

BASE_MODEL = "/models/Qwen3.5-4B-GAIA-Core-Multimodal-v4"

ADAPTERS = [
    {
        "name": "core_conversational_v1",
        "display_name": "Core Conversational V1",
        "training_data": "/gaia/GAIA_Project/knowledge/curricula/conversational/train.json",
        "output_dir": "/models/lora_adapters/tier1_global/core_conversational_v1",
        "description": "Conversational adapter for Core (4B) — same curriculum as Prime conversational_v1",
        "pillar": "personality",
        "tags": ["conversational", "personality", "phrases", "core"],
        # Conversational: gentle learning rate, higher target loss
        "learning_rate": 1.5e-4,
        "target_loss": 0.15,
        "max_steps": 300,
    },
    {
        "name": "core_code_replace_v1",
        "display_name": "Core Code Replace V1",
        "training_data": "/gaia/GAIA_Project/knowledge/curricula/training_data/code_replace_v1.json",
        "output_dir": "/models/lora_adapters/tier3_session/core_code_replace_v1",
        "description": "Code replace format adapter for Core (4B) — OLD_TEXT/NEW_TEXT patches",
        "pillar": "cognition",
        "tags": ["code", "replace", "patch", "codemind", "core"],
        # Code format: standard learning rate, tight target loss
        "learning_rate": 2e-4,
        "target_loss": 0.10,
        "max_steps": 200,
    },
]

# Shared training config
SHARED_CONFIG = {
    "base_model_path": BASE_MODEL,
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 10,
    "convergence_patience": 5,
    "num_train_epochs": None,
    "max_training_time": 900,
    "resume_from": None,
}


def train_adapter(adapter_config: dict) -> bool:
    """Train a single adapter in a subprocess."""
    name = adapter_config["name"]
    training_data_path = adapter_config["training_data"]
    output_dir = adapter_config["output_dir"]

    logger.info("─" * 40)
    logger.info("Training: %s", name)
    logger.info("─" * 40)

    # Load training data
    with open(training_data_path) as f:
        samples = json.load(f)
    logger.info("  Loaded %d training samples", len(samples))

    # Build config
    config = {**SHARED_CONFIG}
    config["adapter_dir"] = output_dir
    config["adapter_name"] = name
    config["samples"] = samples
    config["learning_rate"] = adapter_config["learning_rate"]
    config["target_loss"] = adapter_config["target_loss"]
    config["max_steps"] = adapter_config["max_steps"]

    os.makedirs(output_dir, exist_ok=True)

    # Import training subprocess
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/gaia-common")
    from gaia_study.training_subprocess import run_training, PROGRESS_FILE

    ctx = multiprocessing.get_context("spawn")
    start_time = time.time()

    proc = ctx.Process(target=run_training, args=(config,))
    proc.start()

    # Monitor
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
                logger.info("  [%s] step %d/%d, loss=%.4f, elapsed=%.0fs",
                            state, step, total, loss, time.time() - start_time)
                last_step = step
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    proc.join()
    elapsed = time.time() - start_time

    # Read final state
    try:
        with open(PROGRESS_FILE) as f:
            final = json.load(f)
        final_state = final.get("state", "unknown")
        final_loss = final.get("loss", 0)
        final_steps = final.get("step", 0)
    except Exception:
        final_state = "unknown"
        final_loss = 0
        final_steps = 0

    if proc.exitcode == 0 and final_state == "completed":
        logger.info("  COMPLETE: %d steps, loss=%.4f, %.0fs", final_steps, final_loss, elapsed)

        # Write metadata
        metadata = {
            "name": name,
            "version": "1.0.0",
            "display_name": adapter_config["display_name"],
            "description": adapter_config["description"],
            "tier": 1 if "tier1" in output_dir else 3,
            "pillar": adapter_config["pillar"],
            "rank": 16,
            "alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "training": {
                "method": "qlora",
                "samples": len(samples),
                "steps": final_steps,
                "learning_rate": adapter_config["learning_rate"],
                "batch_size": 1,
                "final_loss": final_loss,
                "duration_seconds": elapsed,
                "base_model": BASE_MODEL,
                "source_documents": [
                    {"path": training_data_path, "samples": len(samples)},
                ],
            },
            "tags": adapter_config["tags"],
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        return True
    else:
        logger.error("  FAILED: exit=%d state=%s", proc.exitcode, final_state)
        return False


def main():
    logger.info("=" * 60)
    logger.info("Core Adapter Training — 2 adapters on %s", BASE_MODEL)
    logger.info("=" * 60)

    if not os.path.isdir(BASE_MODEL):
        logger.error("Base model not found: %s", BASE_MODEL)
        sys.exit(1)

    results = {}
    for adapter in ADAPTERS:
        success = train_adapter(adapter)
        results[adapter["name"]] = success

    logger.info("=" * 60)
    logger.info("RESULTS:")
    for name, ok in results.items():
        logger.info("  %s: %s", name, "PASS" if ok else "FAIL")
    logger.info("=" * 60)

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

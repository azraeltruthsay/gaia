#!/usr/bin/env python3
"""
Train GAIA Primary School — unified identity + voice + tool calling.

Trains both Prime (8B) and Core (4B) versions using the same curriculum.
Uses the training lifecycle manager for proper state management.

Usage:
    python training_lifecycle.py -- docker exec gaia-study python /gaia/GAIA_Project/scripts/train_primary_school.py

Or directly (if lifecycle already handled):
    docker exec gaia-study python /gaia/GAIA_Project/scripts/train_primary_school.py
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

TRAINING_DATA = "/gaia/GAIA_Project/knowledge/curricula/primary_school/train.json"

MODELS = [
    {
        "name": "primary_school_prime_v2",
        "display_name": "Primary School V2 (Prime 8B, clean base)",
        "base_model": "/models/Qwen/Qwen3-8B",
        "output_dir": "/models/lora_adapters/tier1_global/primary_school_prime_v2",
        "description": "Unified identity + voice + tool calling on clean Qwen3-8B (non-abliterated)",
        "tier": 1,
        "pillar": "primary",
        "tags": ["primary_school", "identity", "voice", "tool_calling", "prime", "clean_base"],
        "learning_rate": 1.5e-4,
        "target_loss": 0.35,    # 8B converges higher than 4B on diverse curriculum
        "max_steps": 600,       # ~1.5 epochs — sufficient for 8B
        "gradient_accumulation_steps": 8,  # Override: reduce peak VRAM for 8B
    },
    {
        "name": "primary_school_core",
        "display_name": "Primary School (Core 4B)",
        "base_model": "/models/Qwen/Qwen3.5-4B",
        "output_dir": "/models/lora_adapters/tier1_global/primary_school_core",
        "description": "Unified identity + voice + tool calling + epistemic honesty on clean Qwen3.5-4B (non-abliterated)",
        "tier": 1,
        "pillar": "primary",
        "tags": ["primary_school", "identity", "voice", "tool_calling", "epistemic_honesty", "core", "clean_base"],
        "learning_rate": 1.5e-4,
        "target_loss": 0.25,    # Core needs a bit more slack
        "max_steps": 1000,      # ~2.5 epochs on expanded curriculum
    },
]

SHARED_CONFIG = {
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
    "warmup_steps": 15,
    "convergence_patience": 5,
    "num_train_epochs": None,
    "max_training_time": 3600,  # 60 min per model (doubled for larger curriculum)
    "resume_from": None,
}


def train_model(model_config: dict, samples: list) -> bool:
    """Train a single model in a subprocess."""
    name = model_config["name"]
    base = model_config["base_model"]
    output = model_config["output_dir"]

    logger.info("═" * 50)
    logger.info("Training: %s", name)
    logger.info("  Base: %s", base)
    logger.info("  Samples: %d", len(samples))
    logger.info("═" * 50)

    if not os.path.isdir(base):
        logger.error("Base model not found: %s", base)
        return False

    config = {**SHARED_CONFIG}
    config["base_model_path"] = base
    config["adapter_dir"] = output
    config["adapter_name"] = name
    config["samples"] = samples
    config["learning_rate"] = model_config["learning_rate"]
    config["target_loss"] = model_config["target_loss"]
    config["max_steps"] = model_config["max_steps"]
    # Per-model overrides (e.g. gradient_accumulation_steps for 8B VRAM)
    for key in ("gradient_accumulation_steps", "batch_size", "lora_r", "lora_alpha"):
        if key in model_config:
            config[key] = model_config[key]

    os.makedirs(output, exist_ok=True)

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/gaia-common")
    from gaia_study.training_subprocess import run_training, PROGRESS_FILE

    ctx = multiprocessing.get_context("spawn")
    start_time = time.time()

    proc = ctx.Process(target=run_training, args=(config,))
    proc.start()

    last_step = -1
    while proc.is_alive():
        time.sleep(5)
        try:
            with open(PROGRESS_FILE) as f:
                progress = json.load(f)
            state = progress.get("state", "?")
            step = progress.get("step", 0)
            total = progress.get("total_steps", 0)
            loss = progress.get("loss", 0)
            if step != last_step or state != "training":
                logger.info("  [%s] step %d/%d loss=%.4f (%.0fs)",
                            state, step, total, loss, time.time() - start_time)
                last_step = step
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    proc.join()
    elapsed = time.time() - start_time

    try:
        with open(PROGRESS_FILE) as f:
            final = json.load(f)
        final_state = final.get("state", "?")
        final_loss = final.get("loss", 0)
        final_steps = final.get("step", 0)
    except Exception:
        final_state, final_loss, final_steps = "?", 0, 0

    if proc.exitcode == 0 and final_state == "completed":
        logger.info("  DONE: %d steps, loss=%.4f, %.0fs", final_steps, final_loss, elapsed)

        metadata = {
            "name": name,
            "version": "1.0.0",
            "display_name": model_config["display_name"],
            "description": model_config["description"],
            "tier": model_config["tier"],
            "pillar": model_config["pillar"],
            "rank": 16,
            "alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "training": {
                "method": "qlora",
                "curriculum": "primary_school",
                "samples": len(samples),
                "steps": final_steps,
                "final_loss": final_loss,
                "duration_seconds": elapsed,
                "base_model": base,
                "skills": ["identity", "voice", "tool_calling", "restraint"],
            },
            "tags": model_config["tags"],
        }
        with open(os.path.join(output, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        return True
    else:
        logger.error("  FAILED: exit=%d state=%s", proc.exitcode, final_state)
        return False


def main():
    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  GAIA PRIMARY SCHOOL — Training Session  ║")
    logger.info("╚══════════════════════════════════════════╝")

    with open(TRAINING_DATA) as f:
        samples = json.load(f)
    logger.info("Loaded %d training samples", len(samples))

    results = {}
    for model in MODELS:
        ok = train_model(model, samples)
        results[model["name"]] = ok

    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║            RESULTS                       ║")
    logger.info("╚══════════════════════════════════════════╝")
    for name, ok in results.items():
        status = "PASS ✓" if ok else "FAIL ✗"
        logger.info("  %s: %s", name, status)

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

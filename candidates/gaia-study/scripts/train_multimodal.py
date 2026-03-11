#!/usr/bin/env python3
"""
Train identity LoRA on multimodal Qwen3.5 bases.

Produces adapters with correct model.language_model.layers.* key paths
so they merge cleanly onto ForConditionalGeneration (vision-preserving).

Usage (inside gaia-study container):
  python /app/scripts/train_multimodal.py --model core
  python /app/scripts/train_multimodal.py --model nano
  python /app/scripts/train_multimodal.py --model all
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("GAIA.TrainMultimodal")

# Model configs
MODELS = {
    "core": {
        "base_path": "/models/Qwen3.5-4B-Abliterated",
        "output_dir": "/models/lora_adapters/tier1_global/self-model-core-mm",
        "max_steps": 300,
        "max_seq_length": 512,
    },
    "nano": {
        "base_path": "/models/Qwen3.5-0.8B-Abliterated",
        "output_dir": "/models/lora_adapters/tier1_global/self-model-nano-mm",
        "max_steps": 300,
        "max_seq_length": 512,
    },
}

CURRICULUM_PATH = "/knowledge/curricula/self-model/train.jsonl"


def load_curriculum(path: str):
    """Load JSONL training data."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    logger.info("Loaded %d training samples from %s", len(samples), path)
    return samples


def train_model(model_key: str, samples: list):
    """Train a single model tier."""
    from gaia_study.qlora_trainer import QLoRAConfig, QLoRATrainer

    cfg = MODELS[model_key]
    logger.info("=" * 70)
    logger.info("TRAINING: %s (%s)", model_key.upper(), cfg["base_path"])
    logger.info("=" * 70)

    # Training config from curriculum.json settings
    qlora_config = QLoRAConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=cfg["max_steps"],
        warmup_steps=30,
        max_seq_length=cfg["max_seq_length"],
        logging_steps=10,
        save_steps=50,
        target_loss=0.05,
        convergence_patience=3,
    )

    def on_progress(progress):
        logger.info(
            "  Step %d/%d — loss: %.4f (avg: %.4f) — %.0fs elapsed",
            progress.current_step, progress.total_steps,
            progress.current_loss, progress.avg_loss,
            progress.elapsed_seconds,
        )

    trainer = QLoRATrainer(
        base_model_path=cfg["base_path"],
        config=qlora_config,
        output_dir=cfg["output_dir"],
        progress_callback=on_progress,
    )

    t0 = time.time()

    # Setup
    logger.info("Setting up trainer...")
    if not trainer.setup():
        logger.error("Trainer setup failed for %s", model_key)
        trainer.cleanup()
        return False

    # Prepare dataset
    logger.info("Preparing dataset...")
    train_dataset = trainer.prepare_dataset(samples, format_type="instruction")

    # Train
    logger.info("Starting training...")
    success, metrics = trainer.train(train_dataset, f"self-model-{model_key}-mm", timeout_seconds=1200)

    if success:
        # Save
        trainer.save_adapter(f"self-model-{model_key}-mm", metadata={
            "model_key": model_key,
            "base_model": cfg["base_path"],
            "multimodal": True,
            "stop_reason": metrics.get("stop_reason", "unknown"),
            "final_loss": metrics.get("final_loss", 0),
            "total_steps": metrics.get("total_steps", 0),
            "duration_seconds": time.time() - t0,
        })
        logger.info(
            "%s complete: %d steps, loss=%.4f, stop=%s, %.0fs",
            model_key.upper(),
            metrics.get("total_steps", 0),
            metrics.get("final_loss", 0),
            metrics.get("stop_reason", "?"),
            time.time() - t0,
        )
    else:
        logger.error("%s training FAILED: %s", model_key.upper(), metrics.get("error", "unknown"))

    trainer.cleanup()
    return success


def main():
    parser = argparse.ArgumentParser(description="Train identity LoRA on multimodal bases")
    parser.add_argument("--model", choices=["core", "nano", "all"], required=True,
                        help="Which model to train")
    args = parser.parse_args()

    samples = load_curriculum(CURRICULUM_PATH)
    if not samples:
        logger.error("No training samples found")
        sys.exit(1)

    models_to_train = ["core", "nano"] if args.model == "all" else [args.model]
    results = {}

    for model_key in models_to_train:
        results[model_key] = train_model(model_key, samples)

    logger.info("=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    for k, v in results.items():
        logger.info("  %s: %s", k.upper(), "OK" if v else "FAILED")

    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()

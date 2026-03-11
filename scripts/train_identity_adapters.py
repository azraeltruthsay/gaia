#!/usr/bin/env python3
"""Train QLoRA identity adapters for Core (4B) and Nano (0.8B) models.

Reads pre-formatted JSONL training data from the self-model curriculum
and trains separate LoRA adapters for each model tier.

Run inside gaia-study container:
    docker compose exec -T gaia-study python /knowledge/scripts/train_identity_adapters.py --tier core
    docker compose exec -T gaia-study python /knowledge/scripts/train_identity_adapters.py --tier nano
"""

import argparse
import json
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("identity_training")

TIERS = {
    "core": {
        "base_model_path": "/models/Qwen3.5-4B-Abliterated",
        "adapter_name": "self-model-core",
        "output_dir": "/models/lora_adapters/tier1_global/self-model-core",
        "max_steps": 300,
        "lora_r": 16,
        "lora_alpha": 32,
    },
    "nano": {
        "base_model_path": "/models/Qwen3.5-0.8B-Abliterated",
        "adapter_name": "self-model-nano",
        "output_dir": "/models/lora_adapters/tier1_global/self-model-nano",
        "max_steps": 300,
        "lora_r": 16,
        "lora_alpha": 32,
    },
}


def load_samples(path: str):
    """Load JSONL training samples."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def main():
    parser = argparse.ArgumentParser(description="Train identity QLoRA adapter")
    parser.add_argument("--tier", choices=["core", "nano"], required=True)
    parser.add_argument("--train-data", default="/knowledge/curricula/self-model/train.jsonl")
    parser.add_argument("--val-data", default="/knowledge/curricula/self-model/validation.jsonl")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps")
    parser.add_argument("--dry-run", action="store_true", help="Just load data and validate")
    args = parser.parse_args()

    tier_config = TIERS[args.tier]
    if args.max_steps:
        tier_config["max_steps"] = args.max_steps

    logger.info("=== Identity Adapter Training: %s ===", args.tier.upper())
    logger.info("Base model: %s", tier_config["base_model_path"])
    logger.info("Output: %s", tier_config["output_dir"])

    # Load training data
    train_samples = load_samples(args.train_data)
    logger.info("Loaded %d training samples", len(train_samples))

    if args.dry_run:
        logger.info("Dry run — skipping training")
        return

    from gaia_study.qlora_trainer import QLoRATrainer, QLoRAConfig

    config = QLoRAConfig(
        lora_r=tier_config["lora_r"],
        lora_alpha=tier_config["lora_alpha"],
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=tier_config["max_steps"],
        warmup_steps=10,
        max_seq_length=512,
        logging_steps=10,
        save_steps=50,
        target_loss=0.05,
        convergence_patience=3,
    )

    def on_progress(progress):
        logger.info(
            "Step %d/%d | loss=%.4f | avg_loss=%.4f | elapsed=%.0fs",
            progress.current_step, progress.total_steps,
            progress.current_loss, progress.avg_loss,
            progress.elapsed_seconds,
        )

    trainer = QLoRATrainer(
        base_model_path=tier_config["base_model_path"],
        config=config,
        output_dir=tier_config["output_dir"],
        progress_callback=on_progress,
    )

    # Setup model
    logger.info("Setting up model...")
    t0 = time.time()
    if not trainer.setup():
        logger.error("Model setup failed!")
        sys.exit(1)
    logger.info("Setup complete in %.1fs", time.time() - t0)

    # Prepare dataset
    logger.info("Preparing dataset...")
    dataset = trainer.prepare_dataset(train_samples, format_type="instruction")

    # Train
    logger.info("Starting training (%d steps)...", config.max_steps)
    success, metrics = trainer.train(dataset, tier_config["adapter_name"], timeout_seconds=1800)

    if success:
        # Save adapter
        adapter_path = trainer.save_adapter(
            tier_config["adapter_name"],
            metadata={
                "tier": args.tier,
                "base_model": tier_config["base_model_path"],
                "pillar": "identity",
                "curriculum": "self-model",
                "training_samples": len(train_samples),
                "metrics": {k: v for k, v in metrics.items() if k != "loss_history"},
            },
        )
        logger.info("Adapter saved to %s", adapter_path)
        logger.info("Final loss: %.4f | Steps: %d | Stop reason: %s",
                     metrics.get("final_loss", -1),
                     metrics.get("total_steps", -1),
                     metrics.get("stop_reason", "unknown"))
    else:
        logger.error("Training failed: %s", metrics.get("error", "unknown"))
        sys.exit(1)

    trainer.cleanup()
    logger.info("=== Training complete for %s ===", args.tier.upper())


if __name__ == "__main__":
    main()

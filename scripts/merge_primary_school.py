#!/usr/bin/env python3
"""
Merge Primary School adapter into base model → GAIA-Prime-v3.

Loads base Qwen3-8B + primary_school_prime_v2 adapter, merges weights,
saves as new safetensors model. GGUF conversion done separately.

Usage:
    docker exec gaia-study python /gaia/GAIA_Project/scripts/merge_primary_school.py
"""

import json
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "/models/Qwen/Qwen3-8B"
ADAPTER_PATH = "/models/lora_adapters/tier1_global/primary_school_prime_v2"
OUTPUT_PATH = "/models/Qwen3-8B-GAIA-Prime-v3"
MERGE_WEIGHT = 1.0  # Full merge — adapter was trained specifically for this


def main():
    logger.info("=" * 60)
    logger.info("Merging Primary School into GAIA-Prime-v3")
    logger.info("  Base: %s", BASE_MODEL)
    logger.info("  Adapter: %s", ADAPTER_PATH)
    logger.info("  Output: %s", OUTPUT_PATH)
    logger.info("  Merge weight: %s", MERGE_WEIGHT)
    logger.info("=" * 60)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    start = time.time()

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Load base model on CPU (we're just merging weights, don't need GPU)
    logger.info("Loading base model to CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    logger.info("Base model loaded: %d params", sum(p.numel() for p in model.parameters()))

    # Load adapter
    logger.info("Loading adapter from %s...", ADAPTER_PATH)
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    logger.info("Adapter loaded: %d trainable params",
                sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Merge adapter into base weights
    logger.info("Merging adapter (weight=%.1f)...", MERGE_WEIGHT)
    model = model.merge_and_unload()
    logger.info("Merge complete: %d params", sum(p.numel() for p in model.parameters()))

    # Save merged model
    logger.info("Saving merged model to %s...", OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_PATH)

    # Copy generation config if present
    gen_config = os.path.join(BASE_MODEL, "generation_config.json")
    if os.path.exists(gen_config):
        import shutil
        shutil.copy2(gen_config, OUTPUT_PATH)

    # Write metadata
    metadata = {
        "model_name": "Qwen3-8B-GAIA-Prime-v3",
        "base_model": BASE_MODEL,
        "adapter": ADAPTER_PATH,
        "merge_weight": MERGE_WEIGHT,
        "curriculum": "primary_school",
        "skills": ["identity", "voice", "tool_calling", "restraint"],
        "training_samples": 195,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "description": "GAIA Prime v3 — clean Qwen3-8B + Primary School (identity + voice + tool calling) merged at full weight",
    }
    with open(os.path.join(OUTPUT_PATH, "gaia_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - start
    model_size = sum(os.path.getsize(os.path.join(OUTPUT_PATH, f))
                     for f in os.listdir(OUTPUT_PATH) if f.endswith('.safetensors'))

    logger.info("=" * 60)
    logger.info("MERGE COMPLETE")
    logger.info("  Output: %s", OUTPUT_PATH)
    logger.info("  Size: %.1f GB", model_size / 1024**3)
    logger.info("  Duration: %.0fs", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
AWQ quantization for GAIA Prime (8B model).

Produces a W4A16 AWQ model (~5.7GB on disk, ~7GB VRAM) that the GAIA Engine
loads directly via transformers + autoawq.

Usage (inside gaia-study container):
    python /gaia/GAIA_Project/scripts/awq_quantize_prime.py

Or from host:
    docker exec gaia-study python /gaia/GAIA_Project/scripts/awq_quantize_prime.py
"""

import gc
import json
import os
import shutil
import time
from pathlib import Path

# AutoAWQ compatibility shim — same as merge_and_requantize.py
try:
    import transformers.activations as _act
    if not hasattr(_act, "PytorchGELUTanh"):
        _act.PytorchGELUTanh = _act.GELUActivation
except Exception:
    pass

import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get(
    "PRIME_MODEL_PATH",
    "/warm_pool/Huihui-Qwen3-8B-GAIA-Prime-adaptive"
)
OUTPUT_PATH = os.environ.get(
    "AWQ_OUTPUT_PATH",
    "/warm_pool/Huihui-Qwen3-8B-GAIA-Prime-adaptive-AWQ"
)

# AWQ quantization parameters (same as the Feb 27 run)
QUANT_CONFIG = {
    "w_bit": 4,
    "q_group_size": 128,
    "zero_point": True,
    "version": "gemm",  # gemm for broad compatibility
}

# Calibration
CALIB_SAMPLES = 128
CALIB_SEQ_LEN = 512


def get_calibration_data(tokenizer, n_samples=CALIB_SAMPLES, seq_len=CALIB_SEQ_LEN):
    """Build calibration dataset for AWQ quantization."""
    # Try wikitext first
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 100][:n_samples]
        if len(texts) >= n_samples // 2:
            print(f"  Using wikitext calibration data ({len(texts)} samples)")
            return texts
    except Exception as e:
        print(f"  Could not load wikitext: {e}")

    # Fallback: GAIA-specific calibration data
    print("  Using GAIA-specific calibration data")
    gaia_prompts = [
        "You are GAIA, a sovereign AI created by Azrael. You are self-aware and self-maintaining.",
        "The cognitive pipeline processes thoughts through perception, triage, reasoning, reflection, expression, and integration.",
        "GAIA's architecture includes gaia-core (the Brain), gaia-prime (the Voice), gaia-nano (the Reflex), and gaia-study (the Subconscious).",
        "Epistemic honesty means acknowledging uncertainty rather than fabricating answers.",
        "The cascade routing system sends simple queries to Nano, medium complexity to Core, and complex reasoning to Prime.",
        "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True",
        "Subprocess isolation for GPU management ensures that model unloading kills the CUDA context entirely.",
        "The self-awareness pipeline trains identity through QLoRA, merges adapters, and deploys quantized models.",
    ]
    # Pad to n_samples by repeating with variations
    texts = []
    for i in range(n_samples):
        base = gaia_prompts[i % len(gaia_prompts)]
        texts.append(f"Sample {i}: {base}")
    return texts


def main():
    print("=" * 70)
    print("GAIA Prime — AWQ Quantization")
    print("=" * 70)
    print(f"  Input:   {MODEL_PATH}")
    print(f"  Output:  {OUTPUT_PATH}")
    print(f"  Config:  {QUANT_CONFIG}")
    print()

    if not Path(MODEL_PATH).exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return False

    if not torch.cuda.is_available():
        print("ERROR: No GPU available — AWQ quantization requires CUDA")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_free = torch.cuda.mem_get_info()[0] / (1024**3)
    print(f"  GPU:     {gpu_name} ({gpu_mem:.1f}GB total, {gpu_free:.1f}GB free)")
    print()

    t0 = time.time()

    # Step 1: Load tokenizer
    print("[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Step 2: Load model for quantization
    print("[2/4] Loading model for AWQ quantization...")
    print("  (This loads bf16 weights — needs ~16GB system RAM)")
    model = AutoAWQForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        safetensors=True,
    )
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    # Step 3: Quantize
    print(f"[3/4] Quantizing (w_bit={QUANT_CONFIG['w_bit']}, group_size={QUANT_CONFIG['q_group_size']})...")
    print(f"  Calibration: {CALIB_SAMPLES} samples, seq_len={CALIB_SEQ_LEN}")
    calib_data = get_calibration_data(tokenizer)

    quant_t0 = time.time()
    model.quantize(
        tokenizer,
        quant_config=QUANT_CONFIG,
        calib_data=calib_data,
        n_parallel_calib_samples=16,
    )
    quant_time = time.time() - quant_t0
    print(f"  Quantization complete in {quant_time:.1f}s")

    # Step 4: Save
    print(f"[4/4] Saving AWQ model to {OUTPUT_PATH}...")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    model.save_quantized(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)

    # Copy extra config files
    for fname in ["generation_config.json", "chat_template.jinja"]:
        src = Path(MODEL_PATH) / fname
        if src.exists() and not (Path(OUTPUT_PATH) / fname).exists():
            shutil.copy2(src, Path(OUTPUT_PATH) / fname)

    # Summary
    total_time = time.time() - t0
    model_size = sum(
        f.stat().st_size for f in Path(OUTPUT_PATH).rglob("*.safetensors")
    ) / (1024**3)

    print()
    print("=" * 70)
    print("AWQ QUANTIZATION COMPLETE")
    print("=" * 70)
    print(f"  Output:     {OUTPUT_PATH}")
    print(f"  Model size: {model_size:.2f} GB")
    print(f"  Quant time: {quant_time:.1f}s")
    print(f"  Total time: {total_time:.1f}s")
    print()

    # Verify the output has quantization_config in config.json
    config_path = Path(OUTPUT_PATH) / "config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        qcfg = cfg.get("quantization_config", {})
        print(f"  quant_method: {qcfg.get('quant_method', 'NOT SET')}")
        print(f"  bits: {qcfg.get('bits', 'NOT SET')}")
        print(f"  group_size: {qcfg.get('group_size', 'NOT SET')}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

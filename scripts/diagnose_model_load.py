#!/usr/bin/env python3
"""Diagnose where each parameter lands when loading a model the same way
train_core_multimodal.py does. Used to investigate why Core 2.1 trained
on Unified-v5-Multimodal got loss bouncing 13-28 (vs Core 2.0 on raw
google/gemma-4-E4B which converged to 0.7).

Hypothesis: accelerate device_map="auto" silently offloads some layers
to CPU when loading bf16-saved weights with quantization_config.

Usage:
    python diagnose_model_load.py /models/Gemma4-E4B-GAIA-Unified-v5-Multimodal
    python diagnose_model_load.py /models/google/gemma-4-E4B
"""
import sys
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def diagnose(base_model: str) -> None:
    print(f"=== {base_model} ===")
    print(f"Free VRAM before load: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")

    skip_modules = ["lm_head", "vision_tower", "audio_tower",
                    "embed_vision", "embed_audio"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=skip_modules,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    print(f"torch.cuda.memory_allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Free VRAM after load:        {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")

    # Per-parameter device counter
    device_counts = Counter()
    dtype_counts = Counter()
    for name, p in model.named_parameters():
        device_counts[str(p.device)] += 1
        dtype_counts[str(p.dtype)] += 1
    print(f"Parameter device distribution: {dict(device_counts)}")
    print(f"Parameter dtype distribution:  {dict(dtype_counts)}")

    # Spot-check the xln target layers (30-41) and the towers
    print("\n-- Target LoRA layers (30, 35, 41) --")
    for layer_idx in (30, 35, 41):
        target = f"language_model.layers.{layer_idx}.self_attn.q_proj"
        for name, p in model.named_parameters():
            if name.startswith(target):
                print(f"  {name}: device={p.device} dtype={p.dtype} shape={tuple(p.shape)}")
                break

    print("\n-- Audio tower spot check --")
    for name, p in model.named_parameters():
        if "audio_tower" in name and "weight" in name:
            print(f"  {name}: device={p.device} dtype={p.dtype} shape={tuple(p.shape)}")
            break

    print("\n-- Vision tower spot check --")
    for name, p in model.named_parameters():
        if "vision_tower" in name and "weight" in name:
            print(f"  {name}: device={p.device} dtype={p.dtype} shape={tuple(p.shape)}")
            break

    # Module class spot-check
    print("\n-- Module classes for xln targets --")
    for name, m in model.named_modules():
        if name == "language_model.layers.30.self_attn.q_proj":
            print(f"  {name}: type={type(m).__name__}")
            for attr in ("weight", "bias"):
                if hasattr(m, attr):
                    a = getattr(m, attr)
                    if torch.is_tensor(a):
                        print(f"    .{attr}: device={a.device} dtype={a.dtype}")

    # hf_device_map (the canonical accelerate placement)
    print("\n-- hf_device_map --")
    if hasattr(model, "hf_device_map"):
        dm = model.hf_device_map
        cpu_keys = [k for k, v in dm.items() if v in ("cpu", "disk")]
        print(f"  total entries: {len(dm)}")
        print(f"  CPU/disk entries: {len(cpu_keys)}")
        if cpu_keys:
            print(f"  first 5 CPU: {cpu_keys[:5]}")
        gpu_keys = [k for k, v in dm.items() if v not in ("cpu", "disk")]
        print(f"  GPU entries: {len(gpu_keys)}")
        # Check our target xln range
        target_layers = [k for k in dm if any(f"layers.{i}" in k for i in range(30, 42))]
        target_devices = Counter(dm[k] for k in target_layers)
        print(f"  layers 30-41 device counts: {dict(target_devices)}")
    else:
        print("  no hf_device_map attribute")

    print()


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "/models/Gemma4-E4B-GAIA-Unified-v5-Multimodal"
    diagnose(base)

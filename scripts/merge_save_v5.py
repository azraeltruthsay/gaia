#!/usr/bin/env python3
"""Merge the saved V5 LoRA adapter and save the merged + tower-grafted model.

Used after train_core_multimodal.py runs out of disk while writing the
final safetensors. The adapter at
/models/lora_adapters/gemma4_e4b_core_multimodal_v5/ is fully saved; we
just re-do the merge/dequantize/graft/save tail.
"""
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

# Reuse the helpers from train_core_multimodal.py
import sys
sys.path.insert(0, "/gaia/GAIA_Project/scripts")
from train_core_multimodal import dequantize_linear4bit_modules, save_with_tower_graft

BASE = "/models/Gemma4-E4B-GAIA-Unified-v5-Multimodal"
ADAPTER = "/models/lora_adapters/gemma4_e4b_core_multimodal_v5"
OUT = Path("/models/Gemma4-E4B-GAIA-Core-Multimodal-V5")

print(f"Base:    {BASE}")
print(f"Adapter: {ADAPTER}")
print(f"Out:     {OUT}")

print("\nLoading processor...")
processor = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)

print("Loading base with NF4 (skip vision/audio towers + lm_head)...")
qcfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["vision_tower", "audio_tower", "embed_vision",
                           "embed_audio", "lm_head"],
)
model = AutoModelForImageTextToText.from_pretrained(
    BASE, trust_remote_code=True, quantization_config=qcfg, device_map="cuda",
)

print("Loading adapter on top...")
model = PeftModel.from_pretrained(model, ADAPTER)

print("Merging adapter into model...")
merged = model.merge_and_unload()
del model
torch.cuda.empty_cache()
print(f"  VRAM after merge: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

print("Dequantizing 4bit linear → bf16 on CPU...")
dequantize_linear4bit_modules(merged)

print("Saving (merge + tower graft)...")
OUT.mkdir(parents=True, exist_ok=True)
save_with_tower_graft(merged, BASE, OUT, processor)

print("Done.")

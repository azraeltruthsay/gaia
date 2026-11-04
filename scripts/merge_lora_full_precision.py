#!/usr/bin/env python3
"""Merge a LoRA adapter into full-precision weights, on the CPU, without PEFT.

train_core_multimodal.py merges inside its 4-bit model. PEFT warns this may round,
and it does: Prime v2's identity adapter said "I'm GAIA" unmerged and "I'm Qwen"
merged (2026-11-04). Worse, the saved weights then sit on the NF4 grid, so the next
adapter's small changes round straight back. Here each targeted weight gets
base + (lora_B @ lora_A) * alpha / r, and every other tensor is copied untouched.
"""
import argparse
import glob
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, default=1.0, help="how much of the adapter to apply")
    a = ap.parse_args()
    weights = {}
    for path in sorted(glob.glob(f"{a.base}/*.safetensors")):
        with safe_open(path, "pt") as h:
            for key in h.keys():
                weights[key] = h.get_tensor(key)
    config = json.load(open(f"{a.adapter}/adapter_config.json"))
    scale = config["lora_alpha"] / config["r"] * a.strength
    merged = 0
    with safe_open(f"{a.adapter}/adapter_model.safetensors", "pt") as h:
        for key in h.keys():
            if ".lora_A." not in key:
                continue
            name = key.replace("base_model.model.", "", 1).replace(".lora_A.weight", ".weight")
            target = name if name in weights else name.replace("model.language_model.", "model.", 1)
            if target not in weights:
                raise SystemExit(f"no base weight for {key} (tried {name}, {target})")
            delta = h.get_tensor(key.replace("lora_A", "lora_B")).float() @ h.get_tensor(key).float()
            weights[target] = (weights[target].float() + delta * scale).to(weights[target].dtype)
            merged += 1
    if merged != sum(".lora_A." in k for k in safe_open(f"{a.adapter}/adapter_model.safetensors", "pt").keys()):
        raise SystemExit("merged a different number of modules than the adapter holds")
    tmp = a.out + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    save_file(weights, f"{tmp}/model.safetensors", metadata={"format": "pt"})
    for name in os.listdir(a.base):
        src = os.path.join(a.base, name)
        if not name.endswith((".safetensors", ".index.json")) and os.path.isfile(src):
            shutil.copy(src, tmp)
    shutil.rmtree(a.out, ignore_errors=True)
    os.rename(tmp, a.out)
    print(f"merged {merged} modules into {a.out}", {str(v.dtype) for v in list(weights.values())[:50]})


if __name__ == "__main__":
    main()

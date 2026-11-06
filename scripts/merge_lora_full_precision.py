#!/usr/bin/env python3
"""Merge a LoRA adapter into full-precision weights, on the CPU, without PEFT.

train_core_multimodal.py merges inside its 4-bit model. PEFT warns this may round,
and it does: Prime v2's identity adapter said "I'm GAIA" unmerged and "I'm Qwen"
merged (2026-11-04). Worse, the saved weights then sit on the NF4 grid, so the next
adapter's small changes round straight back. Here each targeted weight gets
base + (lora_B @ lora_A) * alpha / r, and every other tensor is copied untouched.
The base is streamed tensor by tensor into shards (default 4 GB), so a 12B
merges in about one shard of memory.
"""
import argparse
import glob
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


MAX_SHARD_BYTES = 4 * 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, default=1.0, help="how much of the adapter to apply")
    ap.add_argument("--max-shard-gb", type=float, default=MAX_SHARD_BYTES / 2**30)
    a = ap.parse_args()
    config = json.load(open(f"{a.adapter}/adapter_config.json"))
    scale = config["lora_alpha"] / config["r"] * a.strength
    # The adapter is small (a 12B's r16 adapter is ~260 MB in fp32); keep it.
    # The base is streamed: one tensor at a time, written out in shards of at
    # most --max-shard-gb, so peak memory is one shard, not the whole model.
    # Loading all of a 12B's base (24 GB) didn't fit next to GAIA (2026-11-06).
    lora = {}
    with safe_open(f"{a.adapter}/adapter_model.safetensors", "pt") as h:
        for key in h.keys():
            if ".lora_A." in key:
                name = key.replace("base_model.model.", "", 1).replace(".lora_A.weight", ".weight")
                lora[name] = (h.get_tensor(key).float(), h.get_tensor(key.replace("lora_A", "lora_B")).float())
    base_files = sorted(glob.glob(f"{a.base}/*.safetensors"))
    base_names = set()
    for path in base_files:
        with safe_open(path, "pt") as h:
            base_names.update(h.keys())
    targets = {}
    for name in lora:
        target = name if name in base_names else name.replace("model.language_model.", "model.", 1)
        if target not in base_names:
            raise SystemExit(f"no base weight for {name} (tried {target})")
        targets[target] = name

    tmp = a.out + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    limit = int(a.max_shard_gb * 2**30)
    shard, size, written, weight_map, merged, dtypes = {}, 0, [], {}, 0, set()

    def flush():
        nonlocal shard, size
        if shard:
            fname = f"part-{len(written) + 1:05d}.safetensors"
            save_file(shard, f"{tmp}/{fname}", metadata={"format": "pt"})
            written.append((fname, list(shard)))
            shard, size = {}, 0

    for path in base_files:
        with safe_open(path, "pt") as h:
            for key in h.keys():
                t = h.get_tensor(key)
                if key in targets:
                    A, B = lora[targets[key]]
                    t = (t.float() + (B @ A) * scale).to(t.dtype)
                    merged += 1
                nbytes = t.numel() * t.element_size()
                if shard and size + nbytes > limit:
                    flush()
                shard[key] = t
                size += nbytes
                if len(dtypes) < 8:
                    dtypes.add(str(t.dtype))
    flush()
    if merged != len(lora):
        raise SystemExit(f"merged {merged} modules but the adapter holds {len(lora)}")
    n = len(written)
    total = 0
    for i, (fname, keys) in enumerate(written, 1):
        final = "model.safetensors" if n == 1 else f"model-{i:05d}-of-{n:05d}.safetensors"
        os.rename(f"{tmp}/{fname}", f"{tmp}/{final}")
        total += os.path.getsize(f"{tmp}/{final}")
        for k in keys:
            weight_map[k] = final
    if n > 1:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
                  open(f"{tmp}/model.safetensors.index.json", "w"), indent=2)
    for name in os.listdir(a.base):
        src = os.path.join(a.base, name)
        if not name.endswith((".safetensors", ".index.json")) and os.path.isfile(src):
            shutil.copy(src, tmp)
    shutil.rmtree(a.out, ignore_errors=True)
    os.rename(tmp, a.out)
    print(f"merged {merged} modules into {a.out} ({n} shard{'s' if n > 1 else ''})", dtypes)


if __name__ == "__main__":
    main()

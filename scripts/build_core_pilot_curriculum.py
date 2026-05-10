#!/usr/bin/env python3
"""Build the Core 2.x pilot curriculum.

The pilot tests whether raw google/gemma-4-E4B can learn chat alignment
from a tractable curriculum. ~10K samples mixing:
  - 7K Alpaca instruction-following (subsample of 52K)
  - 650 GAIA-specific text (existing core2/text.jsonl)
  - 433 vision pairs (existing core2/vision_pairs.jsonl + images/)
  - 320 audio pairs (existing core2/audio_pairs.jsonl + audio/)

Output:
  knowledge/curricula/core_pilot/
    text.jsonl
    vision_pairs.jsonl   (symlink to core2)
    audio_pairs.jsonl    (symlink to core2)
    images/              (symlink to core2/images)
    audio/               (symlink to core2/audio)
"""
import json
import os
import random
import shutil
from pathlib import Path

from datasets import load_dataset


CORE_PILOT = Path("/gaia/GAIA_Project/knowledge/curricula/core_pilot")
CORE2 = Path("/gaia/GAIA_Project/knowledge/curricula/core2")

ALPACA_TARGET = 7000
GAIA_TEXT_SOURCE = CORE2 / "text.jsonl"
SEED = 42


def fmt_alpaca(sample: dict) -> dict:
    """Alpaca → {instruction, output, category} (collapse 'input' into instruction)."""
    instr = sample.get("instruction", "").strip()
    inp = (sample.get("input") or "").strip()
    out = sample.get("output", "").strip()
    if inp:
        instr = f"{instr}\n\n{inp}"
    return {"instruction": instr, "output": out, "category": "alpaca"}


def main() -> None:
    random.seed(SEED)
    CORE_PILOT.mkdir(parents=True, exist_ok=True)

    # 1. Pull Alpaca subset
    print(f"Loading Alpaca dataset...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    print(f"  total Alpaca: {len(ds)}")
    indices = random.sample(range(len(ds)), ALPACA_TARGET)
    alpaca_samples = []
    for i in indices:
        s = fmt_alpaca(ds[i])
        if not s["instruction"] or not s["output"]:
            continue
        if len(s["output"]) < 10:  # filter trivial
            continue
        alpaca_samples.append(s)
    print(f"  filtered Alpaca: {len(alpaca_samples)}")

    # 2. Load GAIA-specific text from core2
    gaia_samples = []
    with open(GAIA_TEXT_SOURCE) as f:
        for line in f:
            d = json.loads(line)
            instr = d.get("instruction") or d.get("prompt") or ""
            out = d.get("output") or d.get("response") or ""
            if instr and out:
                gaia_samples.append({"instruction": instr, "output": out, "category": "gaia"})
    print(f"  GAIA text: {len(gaia_samples)}")

    # 3. Combine and shuffle text
    all_text = alpaca_samples + gaia_samples
    random.shuffle(all_text)
    out_text = CORE_PILOT / "text.jsonl"
    with open(out_text, "w") as f:
        for s in all_text:
            f.write(json.dumps(s) + "\n")
    print(f"Wrote {len(all_text)} text samples → {out_text}")

    # 4. Symlink multimodal assets (don't duplicate disk)
    for name in ("vision_pairs.jsonl", "audio_pairs.jsonl",
                 "images", "audio"):
        src = CORE2 / name
        dst = CORE_PILOT / name
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        if src.exists():
            os.symlink(src, dst)
            print(f"  symlinked {name}")
        else:
            print(f"  WARN: {src} missing")

    # 5. Counts
    vis = sum(1 for _ in open(CORE_PILOT / "vision_pairs.jsonl"))
    aud = sum(1 for _ in open(CORE_PILOT / "audio_pairs.jsonl"))
    print(f"\n=== Pilot curriculum summary ===")
    print(f"  Text samples (Alpaca + GAIA): {len(all_text)}")
    print(f"  Vision pairs:                 {vis}")
    print(f"  Audio pairs:                  {aud}")
    print(f"  Total:                        {len(all_text) + vis + aud}")


if __name__ == "__main__":
    main()

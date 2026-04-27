#!/usr/bin/env python3
"""
Build a real-image curriculum from COCO captions for Core multimodal.

The programmatic curriculum in `build_core_multimodal_curriculum.py` (12
colors × 8 shades + shapes + text) taught the model that visual tokens
carry signal, but the mapping from [visual tokens] → ["red"] didn't
overpower Gemma 4's strong pretraining prior for hex codes and tag
completions. Real-image captions provide thousands of natural
image→language pairs that sit closer to the distribution Gemma 4
already knows how to handle.

This builder:
  1. Streams `Multimodal-Fatima/COCO_captions_train` (images bundled
     inline as PIL, 5 captions per image)
  2. Saves N images to knowledge/curricula/core-multimodal-coco/images/
  3. Writes a vision_pairs.jsonl with instruction-varied prompts:
       - "Describe this image."
       - "What do you see in this picture?"
       - "Write a short caption for this image."
  4. Picks ONE caption per image at random so we don't overfit to a
     single phrasing.

Run inside gaia-study (has datasets, PIL).

Usage:
    python scripts/build_core_coco_curriculum.py --n 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path

log = logging.getLogger("coco_curriculum")

ROOT = Path("/gaia/GAIA_Project/knowledge/curricula/core-multimodal-coco")
IMAGES_DIR = ROOT / "images"
OUTPUT_JSONL = ROOT / "vision_pairs.jsonl"

PROMPTS = [
    "Describe this image.",
    "What do you see in this picture?",
    "Write a short caption for this image.",
    "What is happening in this image?",
    "Briefly describe what's shown here.",
]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000,
                        help="Number of image-caption pairs to generate (default 2000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true",
                        help="Remove existing images before generating")
    parser.add_argument("--max-image-dim", type=int, default=512,
                        help="Resize longer edge to this many pixels (default 512)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.clean and IMAGES_DIR.exists():
        log.info("Cleaning %s...", IMAGES_DIR)
        shutil.rmtree(IMAGES_DIR)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    log.info("Streaming Multimodal-Fatima/COCO_captions_train (N=%d)...", args.n)
    ds = load_dataset("Multimodal-Fatima/COCO_captions_train",
                      split="train", streaming=True)

    pairs = []
    saved = 0
    skipped = 0

    for example in ds:
        if saved >= args.n:
            break

        try:
            img = example.get("image")
            filename = example.get("filename") or f"coco_{saved:06d}.jpg"
            # This dataset exposes captions as a list of 5 strings under
            # `sentences_raw`. Fallbacks for other COCO-style datasets.
            captions = (
                example.get("sentences_raw")
                or example.get("sentences")
                or example.get("captions")
                or []
            )
            # Normalize to a flat list of strings
            flat = []
            for s in captions:
                if isinstance(s, str):
                    flat.append(s.strip())
                elif isinstance(s, dict):
                    raw = s.get("raw") or s.get("caption") or ""
                    if raw:
                        flat.append(raw.strip())
            captions = [c for c in flat if c]
            if not captions:
                skipped += 1
                continue

            if img is None:
                skipped += 1
                continue

            # Resize to cap longer edge — reduces tokenization load + disk
            if hasattr(img, "mode") and img.mode != "RGB":
                img = img.convert("RGB")
            if hasattr(img, "thumbnail"):
                img.thumbnail((args.max_image_dim, args.max_image_dim))

            # Write to disk
            out_name = Path(filename).with_suffix(".jpg").name
            out_path = IMAGES_DIR / out_name
            img.save(out_path, "JPEG", quality=88)

            caption = rng.choice(captions)
            prompt = rng.choice(PROMPTS)
            pairs.append({
                "image": f"images/{out_name}",
                "instruction": prompt,
                "output": caption,
            })
            saved += 1

            if saved % 250 == 0:
                log.info("  saved %d / %d (skipped %d)", saved, args.n, skipped)

        except Exception as e:
            log.warning("Sample %d failed (%s) — skipping", saved + skipped, e)
            skipped += 1

    with open(OUTPUT_JSONL, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    image_count = len(list(IMAGES_DIR.glob("*.jpg")))
    size_mb = sum(p.stat().st_size for p in IMAGES_DIR.glob("*.jpg")) / 1024**2

    log.info("Done.")
    log.info("  Vision pairs: %d written to %s", len(pairs), OUTPUT_JSONL)
    log.info("  Images: %d (%.1f MB on disk)", image_count, size_mb)
    log.info("  Skipped: %d", skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())

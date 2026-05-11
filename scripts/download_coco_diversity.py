#!/usr/bin/env python3
"""Download diverse COCO val2014 images for vision training (GAIA_Project-6s8).

Pilot 2 + Core 2.x v1 confirmed: image diversity > caption count for
multimodal training. 421 unique images with 5 captions each still
plateaued vision loss at 13+. This script pulls 5000 NEW diverse images
(skipping ones we already have) and emits a fresh vision_pairs.jsonl
with one caption per image.

Source: Multimodal-Fatima/COCO_captions_train (113K available, val2014
images bundled as PIL inside each entry).

Output:
  knowledge/curricula/core_v2x_vision/images/        ~400 MB of NEW images
  knowledge/curricula/core_v2x_vision/vision_pairs.jsonl  5K pairs

Usage:
  python download_coco_diversity.py --target 5000
"""
import argparse
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset


OUT_DIR = Path("/gaia/GAIA_Project/knowledge/curricula/core_v2x_vision")
EXISTING_IMAGES_DIR = Path("/gaia/GAIA_Project/knowledge/curricula/core2/images")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=5000,
                        help="Number of new images to download")
    parser.add_argument("--caption-instruction", default="Describe this image.",
                        help="Instruction prompt paired with each caption")
    args = parser.parse_args()

    images_dir = OUT_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = OUT_DIR / "vision_pairs.jsonl"

    # Identify images we already have on disk (don't re-download)
    existing_cocoids = set()
    if EXISTING_IMAGES_DIR.exists():
        for fname in os.listdir(EXISTING_IMAGES_DIR):
            if "_000000" in fname:
                try:
                    img_id = int(fname.split("_000000")[1].split(".")[0])
                    existing_cocoids.add(img_id)
                except (ValueError, IndexError):
                    pass
    print(f"Existing on-disk image IDs: {len(existing_cocoids)}")

    print("Streaming Multimodal-Fatima/COCO_captions_train...")
    ds = load_dataset("Multimodal-Fatima/COCO_captions_train",
                      split="train", streaming=True)

    new_saved = 0
    pairs = []
    seen_new = set()
    skipped_existing = 0

    with open(pairs_path, "w") as pf:
        for entry in ds:
            if new_saved >= args.target:
                break

            cocoid = entry.get("cocoid")
            if cocoid is None or cocoid in existing_cocoids:
                skipped_existing += 1
                continue
            if cocoid in seen_new:
                continue
            seen_new.add(cocoid)

            # Extract one caption (first available) — favoring diversity
            captions = entry.get("sentences_raw") or []
            cap_strs = []
            for c in captions:
                if isinstance(c, dict):
                    cap_strs.append((c.get("raw") or c.get("sentence") or "").strip())
                elif isinstance(c, str):
                    cap_strs.append(c.strip())
            cap_strs = [c for c in cap_strs if c and len(c) > 5]
            if not cap_strs:
                continue
            caption = cap_strs[0]  # first caption; image diversity is the goal

            # Save image to disk
            img = entry.get("image")
            if img is None:
                continue
            fname_src = entry.get("filename") or f"COCO_val2014_{cocoid:012d}.jpg"
            out_fname = f"coco_{fname_src}"
            out_path = images_dir / out_fname
            try:
                # PIL JpegImageFile can be saved directly; convert to RGB for safety
                img.convert("RGB").save(out_path, format="JPEG", quality=85)
            except Exception as e:
                print(f"  save failed for {fname_src}: {e}")
                continue

            # Emit pair (matching the schema train_core_multimodal expects:
            # image path RELATIVE to images_root; category for per-cat logging)
            pair = {
                "image": f"images/{out_fname}",
                "instruction": args.caption_instruction,
                "output": caption,
                "category": "vision_diverse",
            }
            pf.write(json.dumps(pair) + "\n")
            pairs.append(pair)
            new_saved += 1

            if new_saved % 500 == 0:
                print(f"  saved {new_saved}/{args.target}...")

    print(f"\nDone.")
    print(f"  new images saved:      {new_saved}")
    print(f"  skipped existing:      {skipped_existing}")
    print(f"  pairs jsonl:           {pairs_path}")
    print(f"  images dir:            {images_dir}")
    print(f"  unique cocoids:        {len(seen_new)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Build the v3 multimodal Core curriculum.

Combines:
  - 2000 COCO image-caption pairs (in-distribution natural scenes,
    same as v2)
  - 406 synthetic primitive pairs (5-color smoke + shape recognition,
    re-added to fix the v2 OOD regression seen in e2l)

Output:
  /gaia/GAIA_Project/knowledge/curricula/core-multimodal-v3/
    vision_pairs.jsonl    — merged pairs with paths relative to images/
    images/               — symlinks to source images

The training script (train_core_multimodal.py) references curriculum
paths as constants, so v3 is created as a new directory rather than
mutating v2. Update VISION_CURRICULUM and VISION_IMAGES_ROOT in the
training script (or pass overrides) to use v3.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PROJ = Path("/gaia/GAIA_Project")
SRC_COCO = PROJ / "knowledge/curricula/core-multimodal-coco"
SRC_PRIMS = PROJ / "knowledge/curricula/core-multimodal"
OUT_DIR = PROJ / "knowledge/curricula/core-multimodal-v3"


def symlink_images(src_images_dir: Path, prefix: str) -> int:
    """Symlink every image from src_images_dir into OUT_DIR/images/<prefix>_<name>.

    Returns the count of links created.
    """
    out_images = OUT_DIR / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in src_images_dir.glob("*"):
        if not src.is_file():
            continue
        if src.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        # Source file basenames are unique within each source dir, but we
        # apply a prefix anyway to make the merged dir self-documenting.
        dst = out_images / f"{prefix}_{src.name}"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        count += 1
    return count


def merge_pairs() -> tuple[list[dict], dict]:
    """Read both source jsonl files, rewrite image paths to v3 layout."""
    merged: list[dict] = []
    stats = {"coco": 0, "prim": 0, "skipped": 0}

    coco_path = SRC_COCO / "vision_pairs.jsonl"
    if coco_path.exists():
        for line in coco_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            old_img = d.get("image", "")
            # Existing COCO entries store paths like "images/COCO_val2014_*.jpg"
            base = Path(old_img).name
            d["image"] = f"images/coco_{base}"
            d["category"] = "coco"
            merged.append(d)
            stats["coco"] += 1
    else:
        stats["skipped"] += 1

    prims_path = SRC_PRIMS / "vision_pairs.jsonl"
    if prims_path.exists():
        for line in prims_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            old_img = d.get("image", "")
            base = Path(old_img).name
            d["image"] = f"images/prim_{base}"
            d["category"] = "primitive"
            merged.append(d)
            stats["prim"] += 1
    else:
        stats["skipped"] += 1

    return merged, stats


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Symlinking COCO images from {SRC_COCO}/images/...")
    n_coco = symlink_images(SRC_COCO / "images", "coco")
    print(f"  {n_coco} COCO image links")

    print(f"Symlinking primitive images from {SRC_PRIMS}/images/...")
    n_prim = symlink_images(SRC_PRIMS / "images", "prim")
    print(f"  {n_prim} primitive image links")

    print("Merging vision_pairs.jsonl files...")
    pairs, stats = merge_pairs()
    print(f"  COCO pairs:      {stats['coco']}")
    print(f"  Primitive pairs: {stats['prim']}")
    print(f"  Total:           {len(pairs)}")

    out_jsonl = OUT_DIR / "vision_pairs.jsonl"
    with open(out_jsonl, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"\nWrote {out_jsonl}")
    print(f"Curriculum root: {OUT_DIR}")

    # Sanity check: every referenced image exists
    images_dir = OUT_DIR / "images"
    missing = 0
    for p in pairs:
        if not (images_dir / Path(p["image"]).name).exists():
            missing += 1
    print(f"\nSanity: {missing} missing image references")
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

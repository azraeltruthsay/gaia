#!/usr/bin/env python3
"""
Build the v4 multimodal Core curricula.

Two variants:
  v4a — Single-phase, primitive-heavy mix:
        2000 COCO + 1218 oversampled primitives (3x oversample of unique
        170 primitive images) → 3218 pairs, ~38% primitive weight.
        v3 was 17% primitive weight; v4a tests whether more weight
        overcomes Gemma 4's hex-code colour prior.

  v4b — Two-phase curriculum learning:
        Phase 1 — primitives only (406 pairs).
        Phase 2 — 2000 COCO + 406 primitives (matches v3 mix).
        Tests whether teaching primitives first then refining with COCO
        gives stronger colour grounding than the interleaved mix.

Output:
  /gaia/GAIA_Project/knowledge/curricula/core-multimodal-v4a/
    vision_pairs.jsonl, images/ (symlinks)
  /gaia/GAIA_Project/knowledge/curricula/core-multimodal-v4b-phase1/
    vision_pairs.jsonl, images/ (primitives only)
  v4b phase 2 reuses core-multimodal-v3 directly.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

PROJ = Path("/gaia/GAIA_Project")
SRC_COCO = PROJ / "knowledge/curricula/core-multimodal-coco"
SRC_PRIMS = PROJ / "knowledge/curricula/core-multimodal"

V4A_DIR = PROJ / "knowledge/curricula/core-multimodal-v4a"
V4B_PHASE1_DIR = PROJ / "knowledge/curricula/core-multimodal-v4b-phase1"

OVERSAMPLE_FACTOR = 3  # 406 → 1218 primitive pairs


def symlink_images(src_dir: Path, out_dir: Path, prefix: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in src_dir.glob("*"):
        if not src.is_file() or src.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        dst = out_dir / f"{prefix}_{src.name}"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        count += 1
    return count


def load_pairs(path: Path) -> list[dict]:
    pairs = []
    if not path.exists():
        return pairs
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            pairs.append(json.loads(line))
    return pairs


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def rewrite_image_path(pair: dict, prefix: str) -> dict:
    """Rewrite image path to the v4 layout (images/<prefix>_<basename>)."""
    out = dict(pair)
    base = Path(pair.get("image", "")).name
    out["image"] = f"images/{prefix}_{base}"
    return out


def build_v4a():
    """v4a: 2000 COCO + ~1200 oversampled primitives, single-phase."""
    print("=== Building v4a (single-phase, primitive-heavy) ===")
    images_dir = V4A_DIR / "images"
    n_coco = symlink_images(SRC_COCO / "images", images_dir, "coco")
    n_prim = symlink_images(SRC_PRIMS / "images", images_dir, "prim")
    print(f"  COCO image links:      {n_coco}")
    print(f"  Primitive image links: {n_prim}")

    coco_pairs = [
        {**rewrite_image_path(p, "coco"), "category": "coco"}
        for p in load_pairs(SRC_COCO / "vision_pairs.jsonl")
    ]
    prim_pairs = [
        {**rewrite_image_path(p, "prim"), "category": "primitive"}
        for p in load_pairs(SRC_PRIMS / "vision_pairs.jsonl")
    ]
    # Oversample primitives — each row repeated OVERSAMPLE_FACTOR times.
    oversampled = []
    rng = random.Random(42)
    for _ in range(OVERSAMPLE_FACTOR):
        chunk = list(prim_pairs)
        rng.shuffle(chunk)
        oversampled.extend(chunk)

    merged = coco_pairs + oversampled
    rng.shuffle(merged)  # global shuffle
    write_jsonl(V4A_DIR / "vision_pairs.jsonl", merged)

    print(f"  COCO pairs:        {len(coco_pairs)}")
    print(f"  Primitive pairs:   {len(prim_pairs)} (raw) → {len(oversampled)} ({OVERSAMPLE_FACTOR}× oversample)")
    print(f"  Total:             {len(merged)}")
    print(f"  Primitive weight:  {len(oversampled) / len(merged):.0%}")
    print(f"  Out: {V4A_DIR}")


def build_v4b_phase1():
    """v4b phase 1: primitives only, 5 epochs of focused colour/shape teaching."""
    print("\n=== Building v4b phase 1 (primitives only) ===")
    images_dir = V4B_PHASE1_DIR / "images"
    n_prim = symlink_images(SRC_PRIMS / "images", images_dir, "prim")
    print(f"  Primitive image links: {n_prim}")

    prim_pairs = [
        {**rewrite_image_path(p, "prim"), "category": "primitive"}
        for p in load_pairs(SRC_PRIMS / "vision_pairs.jsonl")
    ]
    write_jsonl(V4B_PHASE1_DIR / "vision_pairs.jsonl", prim_pairs)

    print(f"  Pairs: {len(prim_pairs)}")
    print(f"  Out: {V4B_PHASE1_DIR}")
    print("  Phase 2 reuses /knowledge/curricula/core-multimodal-v3/ unchanged.")


def main() -> int:
    build_v4a()
    build_v4b_phase1()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build Core 2.x v2 curriculum — image-diversity-corrected.

v1 attempt (killed at step 3060/8000) confirmed vision loss stuck at
13.5 due to image-diversity bottleneck (421 unique images × 5 captions
each ≠ training signal of 2105 distinct image groundings). Per
GAIA_Project-6s8 plan: download 5K NEW diverse images (now done) and
rebuild curriculum with image-diverse vision.

Output:
  knowledge/curricula/core_v2x_v2/
    text.jsonl           ~12K (unchanged from v1)
    vision_pairs.jsonl   ~7K (5K NEW + 2K from existing 421 × 5 captions)
    audio_pairs.jsonl    320 (unchanged)
    images/              symlink combining old + new
    audio/               symlink

Key changes from v1:
  - Vision pool: 433 → ~7K pairs across 5421 unique images
  - Vision category renamed to 'vision_diverse' for new + 'vision' for old
  - Text section unchanged (it converged beautifully in v1)
"""
import json
import os
import shutil
import sys
from pathlib import Path


CORE_V2X = Path("/gaia/GAIA_Project/knowledge/curricula/core_v2x")
CORE_V2X_V2 = Path("/gaia/GAIA_Project/knowledge/curricula/core_v2x_v2")
CORE_V2X_VISION = Path("/gaia/GAIA_Project/knowledge/curricula/core_v2x_vision")


def main() -> int:
    CORE_V2X_V2.mkdir(parents=True, exist_ok=True)

    # 1. Text: reuse v1 unchanged
    src_text = CORE_V2X / "text.jsonl"
    dst_text = CORE_V2X_V2 / "text.jsonl"
    if dst_text.exists() or dst_text.is_symlink():
        dst_text.unlink()
    os.symlink(src_text, dst_text)
    n_text = sum(1 for _ in open(src_text))
    print(f"Text: {n_text} (symlinked from v1)")

    # 2. Vision: merge old (multi-caption, 2K) + new (single-caption, 5K)
    vision_old = CORE_V2X / "vision_pairs.jsonl"
    vision_new = CORE_V2X_VISION / "vision_pairs.jsonl"
    merged_vision = CORE_V2X_V2 / "vision_pairs.jsonl"
    with open(merged_vision, "w") as out:
        old_count = 0
        with open(vision_old) as f:
            for line in f:
                d = json.loads(line.strip())
                # Mark these as 'vision' (old, image-redundant) vs 'vision_diverse' (new)
                d["category"] = "vision"
                out.write(json.dumps(d) + "\n")
                old_count += 1
        new_count = 0
        with open(vision_new) as f:
            for line in f:
                # vision_diverse already in category field
                out.write(line)
                new_count += 1
    print(f"Vision: {old_count} old (vision) + {new_count} new (vision_diverse) = {old_count + new_count}")

    # 3. Audio: reuse v1 unchanged
    src_audio = CORE_V2X / "audio_pairs.jsonl"
    dst_audio = CORE_V2X_V2 / "audio_pairs.jsonl"
    if dst_audio.exists() or dst_audio.is_symlink():
        dst_audio.unlink()
    os.symlink(src_audio, dst_audio)
    n_audio = sum(1 for _ in open(src_audio))
    print(f"Audio: {n_audio} (symlinked from v1)")

    # 4. Images: need a merged dir. Use a directory with symlinks into both
    # old and new image dirs (avoids copying ~365MB).
    images_dst = CORE_V2X_V2 / "images"
    if images_dst.exists():
        shutil.rmtree(images_dst) if images_dst.is_dir() and not images_dst.is_symlink() else images_dst.unlink()
    images_dst.mkdir(parents=True)
    # Symlink each image file individually (training reads images/<filename>)
    old_images_dir = Path("/gaia/GAIA_Project/knowledge/curricula/core2/images")
    new_images_dir = CORE_V2X_VISION / "images"
    n_linked = 0
    for src_dir in (old_images_dir, new_images_dir):
        if not src_dir.exists():
            continue
        for fname in os.listdir(src_dir):
            link_path = images_dst / fname
            if link_path.exists():
                continue
            os.symlink(src_dir / fname, link_path)
            n_linked += 1
    print(f"Images: {n_linked} symlinks aggregated")

    # 5. Audio dir symlink
    audio_dst = CORE_V2X_V2 / "audio"
    if audio_dst.exists() or audio_dst.is_symlink():
        audio_dst.unlink() if audio_dst.is_symlink() else shutil.rmtree(audio_dst)
    audio_src = CORE_V2X / "audio"
    if audio_src.exists():
        os.symlink(audio_src, audio_dst)
        print(f"Audio dir: symlinked")

    print(f"\n=== Core 2.x v2 curriculum ===")
    print(f"  Text:   {n_text}")
    print(f"  Vision: {old_count + new_count}")
    print(f"  Audio:  {n_audio}")
    print(f"  Total:  {n_text + old_count + new_count + n_audio}")
    print(f"  Output: {CORE_V2X_V2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

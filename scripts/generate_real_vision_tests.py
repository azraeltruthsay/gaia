#!/usr/bin/env python3
"""Generate real-photo vision battery tests from COCO (GAIA_Project-4d3).

Replaces synthetic primitives (solid_red.png, blue_triangle.png, etc.) with
real COCO photos that have known canonical captions. Each test validates
that the model's response contains the photo's actual subject keyword.

Workflow:
1. Read vision_pairs.jsonl from core_v2x and core_v2x_vision
2. For each target subject, pick photos with simple captions containing
   distinctive keywords
3. Stage selected images to gaia-doctor/test_assets/vision/real/
4. Emit a Python snippet of test cases to splice into cognitive_test_battery.py

Output:
  gaia-doctor/test_assets/vision/real/*.jpg  (16 images)
  /tmp/real_vision_tests.py                  (python test cases)
"""
import json
import os
import random
import shutil
import sys
from pathlib import Path


SOURCES = [
    Path("/gaia/GAIA_Project/knowledge/curricula/core2/vision_pairs.jsonl"),
    Path("/gaia/GAIA_Project/knowledge/curricula/core_v2x_vision/vision_pairs.jsonl"),
]
IMG_DIRS = [
    Path("/gaia/GAIA_Project/knowledge/curricula/core2/images"),
    Path("/gaia/GAIA_Project/knowledge/curricula/core_v2x_vision/images"),
]
DST_DIR = Path("/gaia/GAIA_Project/gaia-doctor/test_assets/vision/real")

# Subject keyword groups. For each, the model's caption should mention the
# primary or any synonym. Pick captions where the subject is clearly central.
SUBJECTS = [
    ("dog", ["dog", "puppy", "canine"], ["A dog"]),
    ("cat", ["cat", "kitten", "feline"], ["A cat"]),
    ("horse", ["horse"], ["A horse"]),
    ("bird", ["bird"], ["A bird"]),
    ("pizza", ["pizza"], ["pizza"]),
    ("sandwich", ["sandwich", "burger"], ["sandwich"]),
    ("bus", ["bus"], ["A bus"]),
    ("train", ["train"], ["A train"]),
    ("bicycle", ["bicycle", "bike"], ["A bicycle", "A bike"]),
    ("motorcycle", ["motorcycle", "motorbike"], ["A motorcycle"]),
    ("umbrella", ["umbrella"], ["umbrella"]),
    ("beach", ["beach", "sand"], ["beach"]),
    ("snow", ["snow", "snowboard", "ski"], ["snow"]),
    ("kitchen", ["kitchen", "stove", "oven"], ["kitchen"]),
    ("airplane", ["airplane", "plane", "aircraft", "jet"], ["airplane", "plane"]),
    ("clock", ["clock"], ["clock"]),
]


def find_image_path(rel_path: str) -> Path | None:
    """rel_path is like 'images/coco_COCO_val2014_xxx.jpg'. Try each img dir."""
    basename = rel_path.split("/")[-1]
    for d in IMG_DIRS:
        p = d / basename
        if p.exists():
            return p
    return None


def pick_one(subject_key: str, kw_list: list[str], prefix_filters: list[str],
             pairs: list[tuple], rng: random.Random) -> tuple | None:
    """Pick one image whose caption contains any keyword and ideally starts
    with a clear subject phrase. Returns (image_path, caption) or None."""
    candidates = []
    for img, cap, _src in pairs:
        cap_lower = cap.lower()
        if not any(k in cap_lower for k in kw_list):
            continue
        # Prefer short clean captions
        if not (30 < len(cap) < 100):
            continue
        # Filter out generic ones
        if cap.count(" ") < 4:
            continue
        # Prefer caps starting with subject-anchoring phrase
        has_anchor = any(cap.lower().startswith(p.lower()) for p in prefix_filters)
        candidates.append((img, cap, has_anchor))
    if not candidates:
        return None
    # Prefer anchored ones
    anchored = [c for c in candidates if c[2]]
    pool = anchored if anchored else candidates
    img_rel, cap, _ = rng.choice(pool)
    img_path = find_image_path(img_rel)
    if img_path is None:
        return None
    return (img_path, cap)


def main() -> int:
    rng = random.Random(42)
    DST_DIR.mkdir(parents=True, exist_ok=True)

    # Load all pairs
    pairs = []
    for src in SOURCES:
        if not src.exists():
            continue
        with open(src) as f:
            for line in f:
                d = json.loads(line)
                pairs.append((d["image"], d["output"], src))
    print(f"Loaded {len(pairs)} caption pairs")

    test_cases = []
    seen_basenames = set()
    for subject_key, kw_list, prefix_filters in SUBJECTS:
        # Try a few times to get a unique image
        picked = None
        for _ in range(10):
            res = pick_one(subject_key, kw_list, prefix_filters, pairs, rng)
            if res is None:
                break
            img_path, cap = res
            if img_path.name not in seen_basenames:
                picked = (img_path, cap)
                break
        if picked is None:
            print(f"  WARN: no match for {subject_key}")
            continue
        img_path, cap = picked
        seen_basenames.add(img_path.name)
        # Stage image
        dst = DST_DIR / f"real_{subject_key}.jpg"
        shutil.copy(img_path, dst)
        print(f"  {subject_key}: {img_path.name} -> {dst.name}")
        print(f"      caption: {cap}")
        # Generate test case (validator uses original kw_list)
        test_cases.append({
            "id": f"vis-real-{subject_key}",
            "image": dst.name,
            "prompt": "Describe this image.",
            "keywords": kw_list,
            "reference_caption": cap,
        })

    # Emit Python snippet
    snippet_path = Path("/tmp/real_vision_tests.py")
    with open(snippet_path, "w") as f:
        f.write("# Auto-generated by generate_real_vision_tests.py\n")
        f.write("# Real-photo vision tests (replaces synthetic primitives per bd 4d3)\n\n")
        for i, t in enumerate(test_cases, 1):
            f.write(f'    # {t["reference_caption"]}\n')
            f.write(f'    {{"id": "vis-real-{i:03d}", "section": "vision",\n')
            f.write(f'     "prompt": {t["prompt"]!r},\n')
            f.write(f'     "image_paths": ["real/{t["image"]}"],\n')
            kws_str = ", ".join(repr(k) for k in t["keywords"])
            f.write(f'     "validators": [{{"type": "keyword_contains_any", "terms": [{kws_str}]}},\n')
            f.write(f'                    {{"type": "min_length", "n": 10}}]}},\n\n')
    print(f"\nWrote {len(test_cases)} test cases to {snippet_path}")
    print(f"Images in: {DST_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

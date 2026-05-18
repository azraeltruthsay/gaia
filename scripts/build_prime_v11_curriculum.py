#!/usr/bin/env python3
"""Prepare Prime tier (Qwen3-VL) training curriculum from V11 spiral.

Supersedes the legacy build_prime_curriculum.py (which hand-curated
identity samples — the V8-V11 arc proved that's the wrong approach;
architecture identity belongs in the system prompt, not weights).

V11's curriculum JSONL is already template-agnostic. Each sample is
`{instruction, output, category}` — the Gemma 4 chat-template wrapping
is applied at training time by format_text_pair(). For Prime training
with Qwen3-VL ChatML, we use the SAME content; only the training-time
wrapper changes.

This script:
  1. Mirrors V11's text.jsonl into a Prime-flavored curriculum dir
     (knowledge/curricula/core_v2x_prime/).
  2. Symlinks vision/audio source files alongside (same data).
  3. Writes a side-by-side preview showing the first N samples rendered
     in Qwen3 ChatML wrapping so the format is sanity-checked BEFORE
     committing to a training run.
  4. Reports per-category histograms for cross-check against V11.

Output:
  /gaia/GAIA_Project/knowledge/curricula/core_v2x_prime/text.jsonl
  /gaia/GAIA_Project/knowledge/curricula/core_v2x_prime/chatml_preview.txt
"""
import json
import shutil
from collections import Counter
from pathlib import Path


SRC = Path("/gaia/GAIA_Project/knowledge/curricula/core_v2x_spiral/text.jsonl")
DST_DIR = Path("/gaia/GAIA_Project/knowledge/curricula/core_v2x_prime")
PREVIEW_N = 12


def qwen3_chatml_wrap(instruction: str, output: str,
                      system: str | None = None) -> str:
    """Render a sample in Qwen3 ChatML format — same wrapping that
    training time will produce on the Prime path."""
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    parts.append(f"<|im_start|>user\n{instruction}<|im_end|>")
    parts.append(f"<|im_start|>assistant\n{output}<|im_end|>")
    return "\n".join(parts)


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: {SRC} not found — run build_spiral_curriculum.py first")
        return 1

    DST_DIR.mkdir(parents=True, exist_ok=True)
    dst_jsonl = DST_DIR / "text.jsonl"

    # Copy text.jsonl (content is template-agnostic; no conversion needed)
    shutil.copy2(SRC, dst_jsonl)
    print(f"Copied {SRC} → {dst_jsonl}")

    # Symlink vision + audio so the training script can find them under
    # the Prime curriculum name without duplicating ~7 GB of images.
    for child in ("vision_pairs.jsonl", "audio_pairs.jsonl", "images", "audio"):
        src_child = SRC.parent / child
        dst_child = DST_DIR / child
        if src_child.exists() and not dst_child.exists():
            try:
                dst_child.symlink_to(src_child)
                print(f"  symlinked {child}")
            except OSError as e:
                print(f"  WARN: failed to symlink {child}: {e}")

    # Per-category stats + collect preview samples (spread across categories)
    cats: Counter = Counter()
    n_total = 0
    samples_by_cat: dict[str, list[dict]] = {}
    with open(dst_jsonl) as f:
        for line in f:
            d = json.loads(line)
            c = d.get("category", "?")
            cats[c] += 1
            n_total += 1
            samples_by_cat.setdefault(c, []).append(d)

    print(f"\nPrime curriculum: {n_total} samples")
    print("Per-category:")
    for c, n in sorted(cats.items()):
        print(f"  {c:25s} {n:6d}")

    # Build preview spread across categories — one per category up to
    # PREVIEW_N total, so the user sees the format applied to different
    # content types (alpaca, tool_routing, multiturn, gaia_identity, etc.)
    preview_samples: list[dict] = []
    for c in sorted(samples_by_cat):
        if samples_by_cat[c]:
            preview_samples.append(samples_by_cat[c][0])
    # Fill remaining slots with additional samples from the largest cat
    if len(preview_samples) < PREVIEW_N:
        largest = max(samples_by_cat, key=lambda c: len(samples_by_cat[c]))
        for s in samples_by_cat[largest][1:]:
            if len(preview_samples) >= PREVIEW_N:
                break
            preview_samples.append(s)

    preview_path = DST_DIR / "chatml_preview.txt"
    with open(preview_path, "w") as f:
        f.write("=" * 76 + "\n")
        f.write(
            f"Qwen3 ChatML preview — {len(preview_samples)} samples from Prime curriculum\n"
        )
        f.write("These are the exact strings that training-time format_text_pair()\n")
        f.write("will emit when targeting Prime (Qwen3 ChatML) instead of Core\n")
        f.write("(Gemma 4 turn tags).\n")
        f.write("=" * 76 + "\n\n")
        for i, s in enumerate(preview_samples):
            wrapped = qwen3_chatml_wrap(
                s.get("instruction", ""),
                s.get("output", ""),
            )
            f.write(f"--- sample {i} [category={s.get('category','?')}] ---\n")
            f.write(wrapped)
            f.write("\n\n")
    print(f"\nWrote preview: {preview_path}")
    print(f"  ({len(preview_samples)} samples, spread across categories)")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

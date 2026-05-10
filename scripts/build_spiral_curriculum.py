#!/usr/bin/env python3
"""Build spiral-ordered text curriculum (parent: GAIA_Project-y4e, fixes: pqb).

Reads knowledge/curricula/core_v2x/text.jsonl (with 'category' on each
sample) and emits a phase-ordered jsonl following the user's spiral
curriculum design from 2026-05-09 planning:

  Phase 1: Foundation       (chat-heavy, build base format)
  Phase 2: Identity build   (rehearse chat, introduce identity + tools)
  Phase 3: Multimodal deep  (placeholder — text-side rehearsal here;
                             vision/audio anchored separately at end of
                             training due to homogeneous-batch sort order)
  Phase 4: Tools + triage   (rehearse multimodal, deepen tools)
  Phase 5: Consolidation    (everything mixed)

Each phase samples WITHOUT replacement from each category bucket,
shuffles the result within phase, then concatenates phases. Category
data persists on every sample.

Output: knowledge/curricula/core_v2x_spiral/text.jsonl
        (vision_pairs.jsonl + audio_pairs.jsonl symlinked from core_v2x)

The training script needs SequentialSampler to honor this order — see
the --no-shuffle flag in train_core_multimodal.py.
"""
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path


CORE_V2X = Path("/gaia/GAIA_Project/knowledge/curricula/core_v2x")
SPIRAL = Path("/gaia/GAIA_Project/knowledge/curricula/core_v2x_spiral")
SEED = 42

# Phase recipes: each phase specifies category proportions.
# Numbers don't have to sum to 1 — they're relative weights inside the phase.
# 'rehearsal' suffix means re-sample from that bucket (with replacement)
# after the bucket has been exhausted in earlier phases.
PHASES = [
    # Phase 1: Foundation — chat format dominant
    {
        "name": "P1_foundation",
        "weights": {
            "alpaca": 0.70,
            "gaia_identity": 0.20,
            "multiturn": 0.10,
        },
    },
    # Phase 2: Identity build (rehearse chat, deepen identity)
    {
        "name": "P2_identity",
        "weights": {
            "gaia_identity": 0.50,
            "alpaca_rehearsal": 0.30,
            "tool_routing": 0.15,
            "deliberation": 0.05,
        },
    },
    # Phase 3: Skill mix (text-side rehearsal — multimodal handled by
    # the dataset's text→vision→audio sort order at end of each epoch)
    {
        "name": "P3_skill_mix",
        "weights": {
            "tool_routing": 0.35,
            "multiturn_rehearsal": 0.25,
            "alpaca_rehearsal": 0.20,
            "gaia_identity_rehearsal": 0.15,
            "deliberation": 0.05,
        },
    },
    # Phase 4: Tools + triage (rehearse all prior, lean into tools)
    {
        "name": "P4_tools",
        "weights": {
            "tool_routing": 0.40,
            "multiturn_rehearsal": 0.25,
            "alpaca_rehearsal": 0.20,
            "gaia_identity_rehearsal": 0.15,
        },
    },
    # Phase 5: Consolidation (everything balanced)
    {
        "name": "P5_consolidation",
        "weights": {
            "alpaca_rehearsal": 0.30,
            "gaia_identity_rehearsal": 0.25,
            "tool_routing_rehearsal": 0.20,
            "multiturn_rehearsal": 0.20,
            "deliberation": 0.05,
        },
    },
]


def main() -> int:
    rng = random.Random(SEED)
    SPIRAL.mkdir(parents=True, exist_ok=True)

    # Load text samples and group by category
    by_cat = defaultdict(list)
    with open(CORE_V2X / "text.jsonl") as f:
        for line in f:
            d = json.loads(line.strip())
            cat = d.get("category", "unknown")
            by_cat[cat].append(d)
    print("Input categories:")
    total_text = 0
    for c in sorted(by_cat):
        n = len(by_cat[c])
        print(f"  {c}: {n}")
        total_text += n
    print(f"  TOTAL TEXT: {total_text}")

    # Build phases by sampling from buckets per recipe.
    # 'fresh' samples come without replacement from the bucket;
    # 'rehearsal' samples are re-drawn (with replacement) — these don't
    # exhaust the bucket, allowing earlier-phase content to recur.
    consumed = defaultdict(set)  # cat -> indices already used as 'fresh'
    phased_output = []
    phase_boundaries = []  # tuples (phase_name, start_idx, end_idx)

    # Allocate phase sizes. We have ~12K samples total. Distribute:
    # P1: 30%, P2: 20%, P3: 20%, P4: 15%, P5: 15% — summing to 100%
    phase_size_pct = [0.30, 0.20, 0.20, 0.15, 0.15]
    target_phase_sizes = [int(total_text * p) for p in phase_size_pct]
    # Pad last phase to absorb rounding remainder
    target_phase_sizes[-1] += total_text - sum(target_phase_sizes)
    print("\nPhase sizes:", target_phase_sizes)

    for phase_idx, phase in enumerate(PHASES):
        phase_target = target_phase_sizes[phase_idx]
        weights = phase["weights"]
        wsum = sum(weights.values())
        # Per-category allocation within this phase
        per_cat_count = {k: int(phase_target * v / wsum)
                         for k, v in weights.items()}
        # Pad last to absorb rounding
        last_key = list(weights.keys())[-1]
        per_cat_count[last_key] += phase_target - sum(per_cat_count.values())

        phase_samples = []
        for key, n_wanted in per_cat_count.items():
            if n_wanted <= 0:
                continue
            base_cat = key.replace("_rehearsal", "")
            pool = by_cat.get(base_cat, [])
            if not pool:
                # Bucket missing entirely — silently skip
                continue
            if "_rehearsal" in key:
                # With-replacement sample (allow recurrence of earlier samples)
                for _ in range(n_wanted):
                    phase_samples.append(rng.choice(pool))
            else:
                # Without-replacement: take indices not yet consumed
                avail = [i for i in range(len(pool)) if i not in consumed[base_cat]]
                if not avail:
                    # Bucket exhausted — fall through to with-replacement
                    for _ in range(n_wanted):
                        phase_samples.append(rng.choice(pool))
                    continue
                rng.shuffle(avail)
                take = avail[:n_wanted]
                if len(take) < n_wanted:
                    # Pad shortfall with replacement draws from same pool
                    short = n_wanted - len(take)
                    take_extra = [rng.choice(pool) for _ in range(short)]
                    for i in take:
                        consumed[base_cat].add(i)
                        phase_samples.append(pool[i])
                    phase_samples.extend(take_extra)
                else:
                    for i in take:
                        consumed[base_cat].add(i)
                        phase_samples.append(pool[i])

        # Shuffle within phase
        rng.shuffle(phase_samples)

        start_idx = len(phased_output)
        phased_output.extend(phase_samples)
        end_idx = len(phased_output)
        phase_boundaries.append((phase["name"], start_idx, end_idx))
        print(f"\n{phase['name']}: {end_idx - start_idx} samples (range {start_idx}:{end_idx})")
        # Sanity histogram
        hist = defaultdict(int)
        for s in phase_samples:
            hist[s.get("category", "?")] += 1
        for c in sorted(hist):
            print(f"  {c}: {hist[c]}")

    # Write output
    out_path = SPIRAL / "text.jsonl"
    with open(out_path, "w") as f:
        for r in phased_output:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(phased_output)} samples → {out_path}")

    # Phase boundary metadata for later inspection
    with open(SPIRAL / "phase_boundaries.json", "w") as f:
        json.dump([{"phase": p[0], "start": p[1], "end": p[2]}
                   for p in phase_boundaries], f, indent=2)

    # Symlink vision/audio + images/audio dirs unchanged from core_v2x
    for name in ("vision_pairs.jsonl", "audio_pairs.jsonl",
                 "images", "audio"):
        src = CORE_V2X / name
        dst = SPIRAL / name
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        if src.exists():
            os.symlink(src, dst)
            print(f"  symlinked {name}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

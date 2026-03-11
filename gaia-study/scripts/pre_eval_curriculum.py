#!/usr/bin/env python3
"""Pre-evaluate curriculum samples against the live Core CPU llama-server.

Scores each sample via token-level F1, classifying as LEARNED or GAP.
Writes a filtered JSONL containing only GAP samples for focused retraining.

Usage:
    docker compose exec gaia-core python /gaia/GAIA_Project/gaia-study/scripts/pre_eval_curriculum.py
    docker compose exec gaia-core python /gaia/GAIA_Project/gaia-study/scripts/pre_eval_curriculum.py \
        --endpoint http://localhost:8092 --threshold 0.5 --output /knowledge/curricula/self-model/train_filtered.jsonl
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from collections import defaultdict


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens for F1 comparison."""
    return re.findall(r"\w+", text.lower())


def token_f1(predicted: str, expected: str) -> float:
    """Compute token-level F1 between predicted and expected text."""
    pred_tokens = tokenize(predicted)
    exp_tokens = tokenize(expected)
    if not pred_tokens or not exp_tokens:
        return 0.0

    pred_set = defaultdict(int)
    exp_set = defaultdict(int)
    for t in pred_tokens:
        pred_set[t] += 1
    for t in exp_tokens:
        exp_set[t] += 1

    # Overlap using min counts (handles duplicates correctly)
    overlap = 0
    for t in pred_set:
        if t in exp_set:
            overlap += min(pred_set[t], exp_set[t])

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(exp_tokens)
    return 2 * precision * recall / (precision + recall)


def query_model(endpoint: str, instruction: str, max_tokens: int = 256, timeout: int = 30) -> str:
    """Send a chat completion request to the llama-server."""
    url = f"{endpoint}/v1/chat/completions"
    payload = json.dumps({
        "model": "core",
        "messages": [{"role": "user", "content": instruction}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def load_curriculum(path: str) -> list[dict]:
    """Load JSONL curriculum file."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARN: Skipping malformed line {line_num}: {e}", file=sys.stderr)
    return samples


def main():
    parser = argparse.ArgumentParser(description="Pre-evaluate curriculum against Core CPU llama-server")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("CORE_CPU_ENDPOINT", "http://localhost:8092"),
        help="Core CPU llama-server endpoint (default: $CORE_CPU_ENDPOINT or http://localhost:8092)",
    )
    parser.add_argument(
        "--input",
        default="/knowledge/curricula/self-model/train.jsonl",
        help="Path to curriculum JSONL (default: /knowledge/curricula/self-model/train.jsonl)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for filtered JSONL output (default: <input_dir>/train_filtered.jsonl)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="F1 threshold: samples above this are LEARNED (default: 0.5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens per inference call (default: 256)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout per request in seconds (default: 30)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample details",
    )
    args = parser.parse_args()

    # Resolve output path
    if args.output is None:
        input_dir = os.path.dirname(args.input)
        args.output = os.path.join(input_dir, "train_filtered.jsonl")

    # Load curriculum
    samples = load_curriculum(args.input)
    if not samples:
        print(f"ERROR: No samples loaded from {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Pre-evaluating {len(samples)} samples against {args.endpoint} ...")
    print(f"  Threshold: {args.threshold}  |  Max tokens: {args.max_tokens}")
    print()

    # Evaluate each sample
    learned = []
    gaps = []
    errors = []
    category_stats = defaultdict(lambda: {"learned": 0, "gap": 0, "error": 0})

    t0 = time.time()
    for i, sample in enumerate(samples):
        instruction = sample.get("instruction", "")
        expected = sample.get("output", "")
        category = sample.get("category", "unknown")

        try:
            predicted = query_model(args.endpoint, instruction, args.max_tokens, args.timeout)
            f1 = token_f1(predicted, expected)
            is_learned = f1 > args.threshold

            if is_learned:
                learned.append((sample, f1))
                category_stats[category]["learned"] += 1
            else:
                gaps.append((sample, f1))
                category_stats[category]["gap"] += 1

            status = "LEARNED" if is_learned else "GAP"
            if args.verbose:
                print(f"  [{i+1:3d}/{len(samples)}] {status}  F1={f1:.3f}  cat={category}  inst={instruction[:60]}...")
            else:
                # Progress indicator every 20 samples
                if (i + 1) % 20 == 0 or i == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (len(samples) - i - 1) / rate if rate > 0 else 0
                    print(f"  [{i+1:3d}/{len(samples)}]  elapsed={elapsed:.0f}s  rate={rate:.1f} samples/s  ETA={eta:.0f}s")

        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            errors.append((sample, str(e)))
            category_stats[category]["error"] += 1
            print(f"  [{i+1:3d}/{len(samples)}] ERROR  cat={category}  {e}", file=sys.stderr)

    elapsed_total = time.time() - t0

    # Summary
    print()
    print("=" * 60)
    print(f"Results ({elapsed_total:.1f}s total):")
    print(f"  LEARNED (F1 > {args.threshold:.2f}):  {len(learned)} samples")
    print(f"  GAP     (F1 <= {args.threshold:.2f}): {len(gaps)} samples")
    if errors:
        print(f"  ERRORS:              {len(errors)} samples")
    print()

    # Per-category breakdown
    print("Per-category:")
    for cat in sorted(category_stats.keys()):
        s = category_stats[cat]
        total = s["learned"] + s["gap"] + s["error"]
        print(f"  {cat:25s}  LEARNED={s['learned']:3d}  GAP={s['gap']:3d}  total={total:3d}")
    print()

    # F1 distribution summary
    all_scores = [(s, f) for s, f in learned] + [(s, f) for s, f in gaps]
    if all_scores:
        scores = [f for _, f in all_scores]
        avg_f1 = sum(scores) / len(scores)
        min_f1 = min(scores)
        max_f1 = max(scores)
        print(f"F1 distribution:  avg={avg_f1:.3f}  min={min_f1:.3f}  max={max_f1:.3f}")
        # Histogram buckets
        buckets = [0] * 10
        for s in scores:
            bucket = min(int(s * 10), 9)
            buckets[bucket] += 1
        print("  Histogram: ", end="")
        for j, count in enumerate(buckets):
            lo = j * 0.1
            hi = lo + 0.1
            print(f"[{lo:.1f}-{hi:.1f}]={count}", end="  ")
        print()
        print()

    # Write filtered output (GAP samples only)
    with open(args.output, "w", encoding="utf-8") as f:
        for sample, f1_score in gaps:
            # Annotate with the eval score for reference
            sample_out = dict(sample)
            sample_out["_pre_eval_f1"] = round(f1_score, 4)
            f.write(json.dumps(sample_out, ensure_ascii=False) + "\n")

    print(f"Filtered dataset: {args.output} ({len(gaps)} samples)")

    # Also write a summary JSON for programmatic consumption
    summary_path = args.output.replace(".jsonl", "_summary.json")
    summary = {
        "total_samples": len(samples),
        "learned": len(learned),
        "gaps": len(gaps),
        "errors": len(errors),
        "threshold": args.threshold,
        "endpoint": args.endpoint,
        "elapsed_seconds": round(elapsed_total, 1),
        "avg_f1": round(avg_f1, 4) if all_scores else None,
        "per_category": {
            cat: dict(s) for cat, s in sorted(category_stats.items())
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON:    {summary_path}")


if __name__ == "__main__":
    main()

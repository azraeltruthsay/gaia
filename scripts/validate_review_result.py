#!/usr/bin/env python3
"""
validate_review_result.py — Parse and validate a CC ReviewResult JSON.

Usage:
    python - < scripts/validate_review_result.py --input review.json
    echo '{"service_id": ...}' | python - < scripts/validate_review_result.py

Routes the result based on promotion_recommendation:
  - approve:            exits 0, prints summary
  - approve_with_notes: exits 0, prints discrepancies as warnings
  - reject:             exits 1, prints discrepancies as errors

Also copies the validated result to the training corpus directory.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from gaia_common.utils.review_prompt_builder import ReviewResult


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and route a CC ReviewResult")
    parser.add_argument("--input", help="Path to ReviewResult JSON file (default: stdin)")
    parser.add_argument("--corpus-dir", default="/knowledge/curricula/code-architect/reviews",
                        help="Training corpus directory for archiving reviews")
    parser.add_argument("--quiet", action="store_true", help="Suppress summary output")
    args = parser.parse_args()

    # Load JSON
    if args.input:
        raw = Path(args.input).read_text(encoding="utf-8")
    else:
        raw = sys.stdin.read()

    # Parse and validate
    try:
        data = json.loads(raw)
        result = ReviewResult.model_validate(data)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"ERROR: ReviewResult validation failed: {e}", file=sys.stderr)
        sys.exit(2)

    # Archive to corpus
    corpus_dir = Path(args.corpus_dir)
    if corpus_dir.exists() or corpus_dir.parent.exists():
        corpus_dir.mkdir(parents=True, exist_ok=True)
        ts = result.review_timestamp.strftime("%Y%m%dT%H%M%S")
        corpus_file = corpus_dir / f"{result.service_id}_{ts}.json"
        corpus_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"Archived to: {corpus_file}", file=sys.stderr)

    # Summary
    if not args.quiet:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Review: {result.service_id} ({result.review_direction})", file=sys.stderr)
        print(f"Reviewer: {result.reviewer}", file=sys.stderr)
        print(f"Fidelity: {result.overall_fidelity_score:.0%}", file=sys.stderr)
        print(f"Recommendation: {result.promotion_recommendation}", file=sys.stderr)
        print(f"Discrepancies: {len(result.discrepancies)}", file=sys.stderr)
        print(f"Open Question Updates: {len(result.open_question_updates)}", file=sys.stderr)
        print(f"Summary: {result.summary_note}", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)

    # Print discrepancies
    for d in result.discrepancies:
        level = "ERROR" if d.severity in ("critical", "major") else "WARNING"
        if not args.quiet:
            print(f"  [{level}] [{d.dimension}] {d.blueprint_claim}", file=sys.stderr)
            print(f"    Evidence: {d.code_evidence}", file=sys.stderr)
            print(f"    Fix: {d.recommendation}", file=sys.stderr)
            if d.affected_file:
                print(f"    File: {d.affected_file}", file=sys.stderr)
            print(file=sys.stderr)

    # Output validated JSON to stdout (for piping)
    print(result.model_dump_json(indent=2))

    # Exit code based on recommendation
    if result.promotion_recommendation == "reject":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

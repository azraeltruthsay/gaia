#!/usr/bin/env python3
"""
run_blueprint_precheck.py — Run mechanical pre-check of blueprint claims against source code.

Usage (inside Docker container):
    python - --blueprint /knowledge/blueprints/gaia-core.yaml --source-dir /app/gaia_core \
        < scripts/run_blueprint_precheck.py

Outputs JSON to stdout: PreCheckResult as dict.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gaia_common.models.blueprint import BlueprintModel
from gaia_common.utils.blueprint_precheck import run_blueprint_precheck


def _load_blueprint(path: str) -> BlueprintModel:
    """Load a blueprint from YAML file."""
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML not available", file=sys.stderr)
        sys.exit(1)

    bp_path = Path(path)
    if not bp_path.exists():
        print(f"Error: Blueprint not found: {bp_path}", file=sys.stderr)
        sys.exit(1)

    raw = yaml.safe_load(bp_path.read_text(encoding="utf-8"))
    return BlueprintModel.model_validate(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run blueprint pre-check against source code")
    parser.add_argument("--blueprint", required=True, help="Path to blueprint YAML file")
    parser.add_argument("--source-dir", required=True, help="Directory containing source files to check")
    parser.add_argument("--output", help="Output file path (default: stdout)")
    parser.add_argument("--categories", nargs="*", help="Limit to specific categories (endpoint, dependency, etc.)")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        print(f"Error: {source_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    blueprint = _load_blueprint(args.blueprint)
    result = run_blueprint_precheck(
        blueprint,
        str(source_dir),
        categories=args.categories,
    )

    # Serialize to JSON
    output = {
        "service_id": result.service_id,
        "timestamp": result.timestamp.isoformat(),
        "items": [
            {
                "category": item.category,
                "blueprint_claim": item.blueprint_claim,
                "status": item.status,
                "source_file": item.source_file,
                "detail": item.detail,
            }
            for item in result.items
        ],
        "summary": {
            "total": result.summary.total,
            "found": result.summary.found,
            "missing": result.summary.missing,
            "diverged": result.summary.diverged,
        },
        "prompt_text": result.to_prompt_text(),
    }

    result_json = json.dumps(output, indent=2)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(result_json, encoding="utf-8")
        print(f"Wrote pre-check results to {args.output}", file=sys.stderr)
    else:
        print(result_json)

    s = result.summary
    pct = (s.found / s.total * 100) if s.total else 0
    print(
        f"Pre-check: {s.total} checks | {s.found} found | {s.missing} missing | {pct:.0f}% complete",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

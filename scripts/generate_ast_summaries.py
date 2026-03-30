#!/usr/bin/env python3
"""
generate_ast_summaries.py — Generate AST summaries for all Python files in a source directory.

Usage (inside Docker container):
    python - --source-dir /app/gaia_core < scripts/generate_ast_summaries.py

Or directly:
    python scripts/generate_ast_summaries.py --source-dir /app/gaia_core

Outputs JSON to stdout: {"filename": {summary_dict}, ...}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gaia_common.utils.ast_summarizer import summarize_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AST summaries for Python source files")
    parser.add_argument("--source-dir", required=True, help="Directory containing Python source files")
    parser.add_argument("--output", help="Output file path (default: stdout)")
    parser.add_argument("--exclude", nargs="*", default=["__pycache__", ".pyc"], help="Patterns to exclude")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        print(f"Error: {source_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    summaries: dict[str, dict] = {}
    py_files = sorted(source_dir.rglob("*.py"))

    for py_file in py_files:
        # Skip __pycache__ and test files
        rel_path = str(py_file.relative_to(source_dir))
        if any(excl in rel_path for excl in args.exclude):
            continue

        try:
            source = py_file.read_text(encoding="utf-8")
            if not source.strip():
                continue
            summary = summarize_file(source, filename=rel_path)
            summaries[rel_path] = summary.to_dict()
        except SyntaxError as e:
            print(f"Warning: Syntax error in {rel_path}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to summarize {rel_path}: {e}", file=sys.stderr)

    result = json.dumps(summaries, indent=2, default=str)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(result, encoding="utf-8")
        print(f"Wrote {len(summaries)} summaries to {args.output}", file=sys.stderr)
    else:
        print(result)

    print(f"Summarized {len(summaries)} files from {source_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()

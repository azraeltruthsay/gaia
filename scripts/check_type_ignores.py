#!/usr/bin/env python3
"""
check_type_ignores.py — Compare # type: ignore counts between candidate and live.

Walks both directories, counts `# type: ignore` per .py file (by relative path),
and fails if any file in the candidate has more ignores than the corresponding
live file.

New files (no live counterpart) are allowed any count.
Missing live counterparts are treated as 0 ignores.

Usage:
    python scripts/check_type_ignores.py <candidate_dir> <live_dir>

Exit codes:
    0 — No file increased its type: ignore count
    1 — At least one file increased (details printed to stderr)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_TYPE_IGNORE_RE = re.compile(r"#\s*type:\s*ignore")


def count_type_ignores(filepath: Path) -> int:
    """Count # type: ignore occurrences in a file."""
    try:
        text = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return 0
    return len(_TYPE_IGNORE_RE.findall(text))


def collect_counts(directory: Path) -> dict[str, int]:
    """Walk a directory and return {relative_path: ignore_count}."""
    counts: dict[str, int] = {}
    if not directory.exists():
        return counts
    for py_file in directory.rglob("*.py"):
        rel = str(py_file.relative_to(directory))
        counts[rel] = count_type_ignores(py_file)
    return counts


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <candidate_dir> <live_dir>", file=sys.stderr)
        return 2

    candidate_dir = Path(sys.argv[1])
    live_dir = Path(sys.argv[2])

    if not candidate_dir.exists():
        print(f"Error: candidate directory not found: {candidate_dir}", file=sys.stderr)
        return 2

    candidate_counts = collect_counts(candidate_dir)
    live_counts = collect_counts(live_dir)

    violations: list[str] = []
    for rel_path, candidate_count in sorted(candidate_counts.items()):
        live_count = live_counts.get(rel_path, 0)
        if candidate_count > live_count:
            violations.append(
                f"  {rel_path}: {live_count} -> {candidate_count} "
                f"(+{candidate_count - live_count})"
            )

    if violations:
        print("type: ignore count increased in the following files:", file=sys.stderr)
        for v in violations:
            print(v, file=sys.stderr)
        return 1

    total_candidate = sum(candidate_counts.values())
    total_live = sum(live_counts.values())
    print(
        f"type: ignore check passed. "
        f"Candidate: {total_candidate}, Live: {total_live}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

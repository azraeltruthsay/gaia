"""CodeMind Changelog — append-only JSONL audit log.

Each entry records a CodeMind cycle: trigger, scope, files touched,
validation results, and outcome. Written to /shared/codemind/changelog.jsonl.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GAIA.CodeMind.Changelog")

DEFAULT_CHANGELOG_PATH = os.environ.get(
    "CODEMIND_CHANGELOG_PATH",
    "/shared/codemind/changelog.jsonl",
)


def append_entry(
    entry: Dict[str, Any],
    path: str = DEFAULT_CHANGELOG_PATH,
) -> None:
    """Append a single entry to the changelog JSONL file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    logger.debug("Changelog entry appended: %s", entry.get("cycle_id", "?"))


def read_entries(
    path: str = DEFAULT_CHANGELOG_PATH,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Read the most recent changelog entries (newest first)."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            lines = f.readlines()
        entries = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries[offset:offset + limit]
    except OSError as e:
        logger.warning("Failed to read changelog: %s", e)
        return []


def summary(path: str = DEFAULT_CHANGELOG_PATH) -> Dict[str, Any]:
    """Return a summary of the changelog."""
    entries = read_entries(path, limit=1000)
    if not entries:
        return {"total": 0, "recent": []}

    outcomes = {}
    for e in entries:
        oc = e.get("outcome", "unknown")
        outcomes[oc] = outcomes.get(oc, 0) + 1

    return {
        "total": len(entries),
        "outcomes": outcomes,
        "latest": entries[0] if entries else None,
        "recent": entries[:5],
    }

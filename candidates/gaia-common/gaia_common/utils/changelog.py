"""
Changelog utility — append-only JSONL change log for GAIA.

Provides append_entry() and read_entries() for machine-queryable
change tracking. Source of truth: /logs/changelog.jsonl
"""

import fcntl
import json
import os
import random
import string
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

CHANGELOG_PATH = Path(os.getenv("CHANGELOG_JSONL", "/logs/changelog.jsonl"))

VALID_TYPES = {"feat", "fix", "refactor", "docs", "promote", "config", "manual"}


def _generate_id() -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"chg_{now}_{suffix}"


def append_entry(
    type: str,
    service: str,
    summary: str,
    author: str = "claude",
    source: str = "manual",
    commit_hash: Optional[str] = None,
    files_changed: Optional[int] = None,
    detail: Optional[str] = None,
) -> dict:
    """Append a changelog entry to the JSONL file. Returns the entry dict."""
    entry = {
        "id": _generate_id(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": type if type in VALID_TYPES else "manual",
        "service": service,
        "summary": summary,
        "author": author,
        "source": source,
    }
    if commit_hash:
        entry["commit_hash"] = commit_hash
    if files_changed is not None:
        entry["files_changed"] = files_changed
    if detail:
        entry["detail"] = detail

    CHANGELOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHANGELOG_PATH, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

    return entry


def read_entries(
    limit: int = 200,
    type_filter: Optional[str] = None,
    service_filter: Optional[str] = None,
) -> list[dict]:
    """Read changelog entries, newest first, with optional filters."""
    if not CHANGELOG_PATH.exists():
        return []

    entries = []
    with open(CHANGELOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if type_filter and entry.get("type") != type_filter:
                continue
            if service_filter and entry.get("service") != service_filter:
                continue
            entries.append(entry)

    entries.reverse()
    return entries[:limit]

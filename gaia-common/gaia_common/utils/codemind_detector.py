"""CodeMind Detector — aggregates signals from immune MRI, observer-scorer, and code_review.

Reads the detect queue (/shared/codemind/detect_queue.jsonl), deduplicates,
and prioritizes issues for the CodeMind engine to process.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GAIA.CodeMind.Detector")

DETECT_QUEUE_PATH = os.environ.get(
    "CODEMIND_DETECT_QUEUE",
    "/shared/codemind/detect_queue.jsonl",
)

# Priority mapping for trigger sources
SOURCE_PRIORITY = {
    "user_request": 1,
    "immune_irritation": 2,
    "drift_detection": 3,
    "code_review": 3,
    "sleep_cycle": 4,
}


def emit_detection(
    source: str,
    issue_type: str,
    file_path: str,
    description: str,
    severity: str = "warn",
    metadata: Optional[Dict[str, Any]] = None,
    queue_path: str = DETECT_QUEUE_PATH,
) -> None:
    """Emit a detection event to the CodeMind detect queue."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "issue_type": issue_type,
        "file_path": file_path,
        "description": description,
        "severity": severity,
        "priority": SOURCE_PRIORITY.get(source, 4),
        "metadata": metadata or {},
    }
    p = Path(queue_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    logger.debug("Detection emitted: %s [%s] %s", source, issue_type, file_path)


def read_detections(
    queue_path: str = DETECT_QUEUE_PATH,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Read pending detections from the queue, sorted by priority."""
    p = Path(queue_path)
    if not p.exists():
        return []
    entries = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as e:
        logger.warning("Failed to read detect queue: %s", e)
        return []

    # Sort by priority (lower = higher priority)
    entries.sort(key=lambda e: e.get("priority", 4))
    return entries[:limit]


def deduplicate(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate detections based on (file_path, issue_type) key."""
    seen = set()
    result = []
    for entry in entries:
        key = (entry.get("file_path", ""), entry.get("issue_type", ""))
        if key not in seen:
            seen.add(key)
            result.append(entry)
    return result


def consume_detections(
    queue_path: str = DETECT_QUEUE_PATH,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Read, deduplicate, and clear consumed entries from the queue.

    Returns up to `limit` unique detections sorted by priority.
    Remaining entries are written back.
    """
    all_entries = read_detections(queue_path, limit=1000)
    if not all_entries:
        return []

    unique = deduplicate(all_entries)
    consumed = unique[:limit]
    remaining = unique[limit:]

    # Rewrite queue with remaining entries
    p = Path(queue_path)
    try:
        with open(p, "w", encoding="utf-8") as f:
            for entry in remaining:
                f.write(json.dumps(entry, default=str) + "\n")
    except OSError as e:
        logger.warning("Failed to rewrite detect queue: %s", e)

    return consumed


def queue_size(queue_path: str = DETECT_QUEUE_PATH) -> int:
    """Return the number of pending detections."""
    p = Path(queue_path)
    if not p.exists():
        return 0
    try:
        with open(p, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0

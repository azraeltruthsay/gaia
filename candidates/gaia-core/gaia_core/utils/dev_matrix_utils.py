"""
dev_matrix_utils.py
- Utilities for loading, diffing, and updating dev_matrix.json
- Enforces absolute path, atomic writes, and audit logging
- Designed for use in self-review and approval flows
"""
import json
import os
import difflib
from pathlib import Path
from typing import Any, Tuple
from datetime import datetime, timezone

DEV_MATRIX_PATH = Path(os.environ.get("DEV_MATRIX_PATH", "app/shared/dev_matrix.json")).resolve()


def load_dev_matrix(path: Path = DEV_MATRIX_PATH) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dev_matrix(data: Any, path: Path = DEV_MATRIX_PATH) -> None:
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def diff_dev_matrix(old: Any, new: Any) -> str:
    old_str = json.dumps(old, indent=2, ensure_ascii=False).splitlines()
    new_str = json.dumps(new, indent=2, ensure_ascii=False).splitlines()
    diff = difflib.unified_diff(old_str, new_str, fromfile="dev_matrix.json (old)", tofile="dev_matrix.json (new)")
    return "\n".join(diff)


def mark_task_complete(task_key: str, prompt: str = None, path: Path = DEV_MATRIX_PATH) -> Tuple[Any, Any, str]:
    """
    Loads dev_matrix, marks the given task complete (if present). Supports both dict and list formats.
    Returns (old, new, diff) where diff is a unified_diff string of the JSON representation.
    """
    old = load_dev_matrix(path)
    # Deep copy via json round-trip to avoid mutating in-place
    new = json.loads(json.dumps(old))

    ts = datetime.now(timezone.utc).isoformat()

    # Case 1: dict keyed by task_key
    if isinstance(new, dict):
        if task_key in new:
            entry = new[task_key]
            entry["status"] = "resolved"
            entry.setdefault("audit", [])
            entry["audit"].append({"by": "gaia_self_review", "note": prompt or "marked complete", "ts": ts})
            entry["resolved"] = ts
        else:
            # Create a new entry
            new[task_key] = {"task": task_key, "status": "resolved", "resolved": ts, "completion_note": prompt or "", "audit": [{"by": "gaia_self_review", "note": prompt or "marked complete", "ts": ts}]}

    # Case 2: list of task objects containing a 'task' field
    elif isinstance(new, list):
        matched = False
        for item in new:
            if isinstance(item, dict) and item.get("task") == task_key:
                item["status"] = "resolved"
                item.setdefault("audit", [])
                item["audit"].append({"by": "gaia_self_review", "note": prompt or "marked complete", "ts": ts})
                item["resolved"] = ts
                matched = True
                break
        if not matched:
            new.append({"task": task_key, "status": "resolved", "resolved": ts, "completion_note": prompt or "", "audit": [{"by": "gaia_self_review", "note": prompt or "marked complete", "ts": ts}]})

    else:
        # Unknown format — no-op
        pass

    diff = diff_dev_matrix(old, new)
    return old, new, diff

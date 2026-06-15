"""Session hygiene — archive stale sessions to prevent temporal drift.

Sessions accumulate indefinitely. Many are smoke tests, dev artefacts,
or conversations from prior model versions whose embeddings now surface
in retrieval and pull old context into new turns. Periodic archival
keeps the live session set small and recent so retrieval reflects the
current model's actual conversational history.

Archived sessions move into `/shared/sessions.archived/<date>/<sid>.json`
and stay readable — nothing is deleted, only relocated.

Configurable via gaia_constants.json `SESSION_HYGIENE`:
    enabled             — master switch
    archive_after_days  — sessions older than this get archived (default 30)
    archive_min_messages — sessions with fewer messages also archived (1)
    min_run_interval_h  — minimum hours between runs (default 168 = weekly)
    keep_recent_n       — minimum sessions to keep live regardless (200)
    dry_run_default     — if true, log decisions without moving files

A side-effect of this design: vector indexes (session_history_indexer)
should be rebuilt after archive; the indexer rebuilds lazily on next
add, so just the surviving sessions get re-embedded over time.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


SHARED_DIR = Path(os.environ.get("SHARED_DIR", "/shared"))
SESSIONS_FILE = SHARED_DIR / "sessions.json"
ARCHIVE_ROOT = SHARED_DIR / "sessions.archived"
LAST_RUN_MARKER = SHARED_DIR / "session_hygiene.last_run"


def _load_config(config) -> Dict[str, Any]:
    """Pull SESSION_HYGIENE config block with sensible defaults."""
    constants = getattr(config, "constants", {}) or {}
    block = constants.get("SESSION_HYGIENE", {}) or {}
    return {
        "enabled": block.get("enabled", True),
        "archive_after_days": int(block.get("archive_after_days", 30)),
        "archive_min_messages": int(block.get("archive_min_messages", 1)),
        "min_run_interval_h": int(block.get("min_run_interval_h", 168)),
        "keep_recent_n": int(block.get("keep_recent_n", 200)),
        "dry_run_default": bool(block.get("dry_run_default", False)),
    }


def _last_run_age_hours() -> Optional[float]:
    if not LAST_RUN_MARKER.exists():
        return None
    try:
        ts = float(LAST_RUN_MARKER.read_text().strip())
        return (time.time() - ts) / 3600.0
    except (ValueError, OSError):
        return None


def _record_run() -> None:
    try:
        LAST_RUN_MARKER.parent.mkdir(parents=True, exist_ok=True)
        LAST_RUN_MARKER.write_text(str(time.time()))
    except OSError as e:
        logger.warning("Could not write last-run marker: %s", e)


def _session_last_active(session: Dict[str, Any]) -> Optional[datetime]:
    """Newest message timestamp, or session creation time as fallback."""
    history = session.get("history", [])
    for msg in reversed(history):
        ts_str = msg.get("timestamp")
        if not ts_str:
            continue
        try:
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
    created = session.get("created_at")
    if created:
        try:
            return datetime.fromisoformat(str(created).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            pass
    return None


def _classify_sessions(
    sessions: Dict[str, Any], cfg: Dict[str, Any]
) -> Tuple[List[str], List[str], List[str]]:
    """Sort session IDs into (keep, archive_old, archive_trivial)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=cfg["archive_after_days"])

    keep: List[str] = []
    archive_old: List[str] = []
    archive_trivial: List[str] = []

    for sid, raw in sessions.items():
        # raw may be a Session.to_dict() dict or already a plain dict
        if not isinstance(raw, dict):
            keep.append(sid)
            continue

        history = raw.get("history") or []

        if len(history) < cfg["archive_min_messages"]:
            archive_trivial.append(sid)
            continue

        last = _session_last_active(raw)
        if last is None:
            # No timestamps at all — almost certainly dev noise
            archive_trivial.append(sid)
            continue

        if last < cutoff:
            archive_old.append(sid)
        else:
            keep.append(sid)

    # keep_recent_n is a safety floor: even if everything's old, keep the
    # most recent N sessions live so retrieval doesn't go empty.
    if len(keep) < cfg["keep_recent_n"] and archive_old:
        # Promote the most recent archived ones back into keep.
        candidates = sorted(
            archive_old,
            key=lambda sid: _session_last_active(sessions[sid]) or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        needed = cfg["keep_recent_n"] - len(keep)
        for sid in candidates[:needed]:
            keep.append(sid)
            archive_old.remove(sid)

    return keep, archive_old, archive_trivial


def run_hygiene(config, dry_run: Optional[bool] = None) -> Dict[str, Any]:
    """Run session hygiene once.

    Args:
        config: GAIA Config object (provides constants).
        dry_run: If True, log decisions without moving files. None uses
            config default.

    Returns:
        Dict with counts of archived/kept and the new sessions.json size.
        Always returns even when no-op so callers can log clean stats.
    """
    cfg = _load_config(config)
    if dry_run is None:
        dry_run = cfg["dry_run_default"]

    if not cfg["enabled"]:
        return {"skipped": "disabled"}

    age_h = _last_run_age_hours()
    if age_h is not None and age_h < cfg["min_run_interval_h"]:
        return {"skipped": "interval", "last_run_age_hours": round(age_h, 1)}

    if not SESSIONS_FILE.exists():
        return {"skipped": "no_sessions_file"}

    try:
        with SESSIONS_FILE.open() as f:
            sessions = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("session_hygiene: cannot read sessions.json: %s", e)
        return {"skipped": "read_error", "error": str(e)}

    if not isinstance(sessions, dict):
        return {"skipped": "unexpected_format"}

    keep, archive_old, archive_trivial = _classify_sessions(sessions, cfg)

    result = {
        "total": len(sessions),
        "keep": len(keep),
        "archive_old": len(archive_old),
        "archive_trivial": len(archive_trivial),
        "dry_run": dry_run,
    }

    if not (archive_old or archive_trivial):
        _record_run()
        return {**result, "skipped": "nothing_to_archive"}

    if dry_run:
        logger.info("session_hygiene [dry_run]: would archive %d old + %d trivial, keep %d",
                    len(archive_old), len(archive_trivial), len(keep))
        return result

    # Move qualifying sessions into archive
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    archive_dir = ARCHIVE_ROOT / stamp
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived = 0
    for sid in archive_old + archive_trivial:
        target = archive_dir / f"{sid}.json"
        try:
            with target.open("w") as f:
                json.dump(sessions[sid], f, indent=2, default=str)
            archived += 1
        except OSError as e:
            logger.warning("session_hygiene: could not write %s: %s", target, e)

    # Rewrite live sessions.json with just the kept set
    try:
        new_sessions = {sid: sessions[sid] for sid in keep}
        # Atomic rewrite via temp file
        tmp = SESSIONS_FILE.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(new_sessions, f, indent=2, default=str)
        tmp.replace(SESSIONS_FILE)
    except OSError as e:
        logger.error("session_hygiene: could not rewrite sessions.json: %s", e)
        return {**result, "rewrite_error": str(e)}

    _record_run()
    new_size_mb = SESSIONS_FILE.stat().st_size / 1024 / 1024
    logger.info(
        "session_hygiene: archived %d (%d old + %d trivial), kept %d "
        "(sessions.json now %.1f MB) — archive: %s",
        archived, len(archive_old), len(archive_trivial), len(keep),
        new_size_mb, archive_dir,
    )

    return {
        **result,
        "archived": archived,
        "archive_dir": str(archive_dir),
        "new_size_mb": round(new_size_mb, 1),
    }

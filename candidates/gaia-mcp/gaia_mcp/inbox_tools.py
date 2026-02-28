"""
Audio Inbox MCP Tools — read-only tools to query the audio inbox daemon.

Provides 3 tools:
  - audio_inbox_status   (read — daemon state, current job, queue depth)
  - audio_inbox_list     (read — files in each state: new/processing/done)
  - audio_inbox_review   (read — retrieve transcript + review for a completed file)

The host-side daemon (scripts/gaia_audio_inbox.py) writes a status file
and sidecars in audio_inbox/done/.  These tools read those files.
"""

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger("GAIA.InboxTools")

_STATUS_FILE = Path(os.getenv("INBOX_STATUS_FILE", "/logs/inbox_status.json"))
_INBOX_DIR = Path(os.getenv("INBOX_DIR", "/gaia/GAIA_Project/audio_inbox"))

_AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac", ".wma", ".opus"}
_CONTROL_FILE = Path(os.getenv("INBOX_CONTROL_FILE", "/logs/inbox_control.json"))


def audio_inbox_status(params: dict) -> dict:
    """Read the inbox daemon status file."""
    if not _STATUS_FILE.exists():
        return {
            "ok": True,
            "running": False,
            "message": "Inbox daemon not detected (no status file). "
                       "Start it on the host: python scripts/gaia_audio_inbox.py",
        }

    try:
        status = json.loads(_STATUS_FILE.read_text())
        last_update = status.get("updated_at", 0)
        stale = (time.time() - last_update) > 30 if last_update else True
        status["stale"] = stale
        status["ok"] = True
        return status
    except Exception as e:
        logger.error("Failed to read inbox status file: %s", e)
        return {"ok": False, "error": f"Failed to read status file: {e}"}


def audio_inbox_list(params: dict) -> dict:
    """List audio files in each inbox state (new/processing/done)."""
    result = {"ok": True, "new": [], "processing": [], "done": []}

    for state in ("new", "processing", "done"):
        state_dir = _INBOX_DIR / state
        if not state_dir.exists():
            continue
        files = sorted(
            (f.name for f in state_dir.iterdir() if f.suffix.lower() in _AUDIO_EXTENSIONS),
        )
        result[state] = files

    result["counts"] = {
        "new": len(result["new"]),
        "processing": len(result["processing"]),
        "done": len(result["done"]),
    }
    return result


def audio_inbox_review(params: dict) -> dict:
    """Retrieve transcript + review for a completed file.

    Pass the filename (with or without extension) to look up the
    sidecar files in audio_inbox/done/.
    """
    filename = (params.get("filename") or "").strip()
    if not filename:
        return {"ok": False, "error": "filename parameter is required"}

    done_dir = _INBOX_DIR / "done"
    if not done_dir.exists():
        return {"ok": False, "error": "done/ directory does not exist"}

    # Resolve stem (strip extension if provided)
    stem = Path(filename).stem

    transcript_path = done_dir / f"{stem}.transcript.txt"
    review_path = done_dir / f"{stem}.review.txt"
    meta_path = done_dir / f"{stem}.meta.json"

    if not transcript_path.exists() and not review_path.exists():
        return {"ok": False, "error": f"No sidecars found for '{filename}' in done/"}

    result = {"ok": True, "filename": filename, "stem": stem}

    if transcript_path.exists():
        result["transcript"] = transcript_path.read_text(encoding="utf-8")
        result["transcript_chars"] = len(result["transcript"])

    if review_path.exists():
        result["review"] = review_path.read_text(encoding="utf-8")
        result["review_chars"] = len(result["review"])

    if meta_path.exists():
        try:
            result["metadata"] = json.loads(meta_path.read_text())
        except Exception:
            result["metadata"] = None

    return result


def audio_inbox_process(params: dict) -> dict:
    """Trigger audio inbox processing by writing a control file.

    The host-side daemon polls this file and processes all queued audio
    files when it sees a ``process`` command.
    """
    try:
        control = {"command": "process", "source": "mcp_tool"}
        _CONTROL_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _CONTROL_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(control, indent=2))
        tmp.rename(_CONTROL_FILE)
        logger.info("Inbox process command written via MCP tool")
        return {"ok": True, "message": "Process command sent to audio inbox daemon."}
    except Exception as e:
        logger.error("Failed to write inbox control file: %s", e)
        return {"ok": False, "error": str(e)}

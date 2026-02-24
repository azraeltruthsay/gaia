"""
Audio Listener MCP Tools — control the host-side system audio capture daemon.

Provides 3 tools:
  - audio_listen_start   (write, sensitive — starts system audio capture)
  - audio_listen_stop    (read — stops capture)
  - audio_listen_status  (read — returns current listener state)

The host-side daemon (scripts/gaia_listener.py) polls a control file
and writes a status file.  These tools simply read/write those JSON files.
"""

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger("GAIA.ListenerTools")

_CONTROL_FILE = Path(os.getenv("LISTENER_CONTROL_FILE", "/logs/listener_control.json"))
_STATUS_FILE = Path(os.getenv("LISTENER_STATUS_FILE", "/logs/listener_status.json"))

_VALID_MODES = {"passive", "active"}


def audio_listen_start(params: dict) -> dict:
    """Write a start command to the listener control file."""
    mode = (params.get("mode") or "passive").strip().lower()
    if mode not in _VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Valid: {', '.join(sorted(_VALID_MODES))}")

    comment_threshold = (params.get("comment_threshold") or "interesting").strip()

    control = {
        "command": "start",
        "mode": mode,
        "comment_threshold": comment_threshold,
        "issued_at": time.time(),
    }

    try:
        _CONTROL_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CONTROL_FILE.write_text(json.dumps(control, indent=2))
        logger.info("Listener start command written: mode=%s", mode)
        return {"ok": True, "command": "start", "mode": mode}
    except Exception as e:
        logger.error("Failed to write listener control file: %s", e)
        return {"ok": False, "error": f"Failed to write control file: {e}"}


def audio_listen_stop(params: dict) -> dict:
    """Write a stop command to the listener control file."""
    control = {
        "command": "stop",
        "issued_at": time.time(),
    }

    try:
        _CONTROL_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CONTROL_FILE.write_text(json.dumps(control, indent=2))
        logger.info("Listener stop command written")
        return {"ok": True, "command": "stop"}
    except Exception as e:
        logger.error("Failed to write listener control file: %s", e)
        return {"ok": False, "error": f"Failed to write control file: {e}"}


def audio_listen_status(params: dict) -> dict:
    """Read the listener status file written by the host daemon."""
    if not _STATUS_FILE.exists():
        return {
            "ok": True,
            "running": False,
            "message": "Listener daemon not detected (no status file). "
                       "Start it on the host: python scripts/gaia_listener.py",
        }

    try:
        status = json.loads(_STATUS_FILE.read_text())
        # Add staleness check
        last_update = status.get("updated_at", 0)
        stale = (time.time() - last_update) > 30 if last_update else True
        status["stale"] = stale
        status["ok"] = True
        return status
    except Exception as e:
        logger.error("Failed to read listener status file: %s", e)
        return {"ok": False, "error": f"Failed to read status file: {e}"}

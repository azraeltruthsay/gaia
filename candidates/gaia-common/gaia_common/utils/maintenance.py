"""
Maintenance mode utilities — stdlib only (safe for gaia-doctor import).

Flag file: /shared/maintenance_mode.json
Legacy compat: also manages /shared/ha_maintenance for older checks.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

_SHARED_DIR = Path(os.environ.get("SHARED_DIR", "/shared"))
_FLAG_FILE = _SHARED_DIR / "maintenance_mode.json"
_LEGACY_FLAG = _SHARED_DIR / "ha_maintenance"


def is_maintenance_active() -> bool:
    """Return True if maintenance mode is currently active."""
    try:
        if _FLAG_FILE.exists():
            data = json.loads(_FLAG_FILE.read_text())
            return data.get("active", False)
    except (json.JSONDecodeError, OSError):
        pass
    # Fall back to legacy flag
    return _LEGACY_FLAG.exists()


def get_maintenance_info() -> dict | None:
    """Return full maintenance mode data, or None if not active."""
    if not is_maintenance_active():
        return None
    try:
        if _FLAG_FILE.exists():
            return json.loads(_FLAG_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        pass
    # Legacy flag exists but no JSON — synthesize
    if _LEGACY_FLAG.exists():
        return {"active": True, "entered_at": "unknown", "entered_by": "legacy", "reason": "ha_maintenance flag"}
    return None


def enter_maintenance(reason: str = "manual", entered_by: str = "unknown") -> dict:
    """Activate maintenance mode. Returns the flag data written."""
    _SHARED_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "active": True,
        "entered_at": datetime.now(timezone.utc).isoformat(),
        "entered_by": entered_by,
        "reason": reason,
    }
    _FLAG_FILE.write_text(json.dumps(data, indent=2))
    # Legacy compat — create ha_maintenance flag
    _LEGACY_FLAG.touch()
    return data


def exit_maintenance() -> dict:
    """Deactivate maintenance mode. Returns summary."""
    info = get_maintenance_info()
    duration = None
    if info and info.get("entered_at", "unknown") != "unknown":
        try:
            entered = datetime.fromisoformat(info["entered_at"])
            duration = (datetime.now(timezone.utc) - entered).total_seconds()
        except (ValueError, TypeError):
            pass

    # Remove both flag files
    try:
        _FLAG_FILE.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        _LEGACY_FLAG.unlink(missing_ok=True)
    except OSError:
        pass

    return {
        "active": False,
        "exited_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration,
        "previous": info,
    }

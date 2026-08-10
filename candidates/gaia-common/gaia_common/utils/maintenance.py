"""
Maintenance mode utilities — stdlib only (safe for gaia-doctor import).

Flag file: /shared/maintenance_mode.json
Legacy compat: also manages /shared/ha_maintenance for older checks.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("GAIA.Maintenance")

_SHARED_DIR = Path(os.environ.get("SHARED_DIR", "/shared"))
_FLAG_FILE = _SHARED_DIR / "maintenance_mode.json"
_LEGACY_FLAG = _SHARED_DIR / "ha_maintenance"

# GAIA_Project-5jdw: maintenance mode unconditionally suppresses Discord
# wake (SleepWakeManager.receive_wake_signal) with no expiry. A dashboard
# entry on 2026-07-19 was never exited and silently blocked wake for 3+
# days with no alert anywhere. TTL default mirrors the VRAM tenant guard
# (gaia-orchestrator lifecycle_machine.py VRAM_TENANT_GUARD_TTL, bead
# 85mb/9zrx) — same "stuck flag auto-expires, fail toward not-blocking"
# philosophy, applied here to a flag with a much bigger blast radius.
_DEFAULT_TTL_SECONDS = int(os.environ.get("MAINTENANCE_MODE_TTL", "14400"))


def _flag_age_seconds(data: dict) -> Optional[float]:
    """Seconds since `entered_at`, or None if unparseable/absent."""
    entered_at = data.get("entered_at")
    if not entered_at or entered_at == "unknown":
        return None
    try:
        entered = datetime.fromisoformat(entered_at)
        return (datetime.now(timezone.utc) - entered).total_seconds()
    except (ValueError, TypeError):
        return None


def _is_stale(data: dict) -> bool:
    """True if this flag has outlived its TTL (or its age can't be told).

    An unparseable/missing `entered_at` fails toward "stale" (clear it) to
    match this module's existing posture: corrupt state should never be
    able to block wake forever (same reasoning as the corrupt-JSON handling
    below). Flags written before this TTL existed have no `ttl_seconds`
    field — they fall back to the same default, which retroactively closes
    5jdw for any flag already stuck when this ships.
    """
    age = _flag_age_seconds(data)
    if age is None:
        return True
    try:
        ttl = float(data.get("ttl_seconds", _DEFAULT_TTL_SECONDS))
    except (TypeError, ValueError):
        ttl = _DEFAULT_TTL_SECONDS
    return age > ttl


def _clear_flag_files() -> None:
    try:
        _FLAG_FILE.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        _LEGACY_FLAG.unlink(missing_ok=True)
    except OSError:
        pass


def is_maintenance_active() -> bool:
    """Return True if maintenance mode is currently active (and not stale).

    The JSON flag (`maintenance_mode.json`) is the single source of truth.
    The legacy `ha_maintenance` flag is written alongside the JSON for
    backward-compat readers (older retry paths) but is NOT authoritative
    on its own — if it appears without the JSON, that's an orphan from
    a service that crashed mid-operation or a manual touch, and it gets
    auto-cleaned here.

    Past incident: a stale `ha_maintenance` flag from 2026-04-26 survived
    deletion of the JSON and locked the system into fake-maintenance for
    9 hours. Wake signals were silently suppressed. (See bd issue.)

    A second incident (5jdw, 2026-07-19 to 2026-07-23) showed the JSON flag
    itself has the same failure mode: entered via the dashboard and never
    exited, it blocked Discord wake for 3+ days with nothing surfacing it.
    Past the TTL, the flag is now auto-cleared here — same fail-toward-
    permissive posture as the corrupt-JSON case just below.
    """
    try:
        if _FLAG_FILE.exists():
            data = json.loads(_FLAG_FILE.read_text())
            if data.get("active", False):
                if _is_stale(data):
                    logger.warning(
                        "Maintenance mode flag stale (entered_at=%s, ttl_seconds=%s) "
                        "— auto-clearing (GAIA_Project-5jdw)",
                        data.get("entered_at", "unknown"),
                        data.get("ttl_seconds", _DEFAULT_TTL_SECONDS),
                    )
                    _clear_flag_files()
                    return False
                return True
            # JSON exists but says active=False — clean up legacy mirror.
    except (json.JSONDecodeError, OSError):
        # Corrupt JSON is treated as "no maintenance" — fail safe.
        pass

    # JSON is absent or active=False. If the legacy flag is loitering
    # alone, it's an orphan — remove it so subsequent checks return False
    # without re-tripping over the same file.
    if _LEGACY_FLAG.exists():
        try:
            _LEGACY_FLAG.unlink()
        except OSError:
            # Couldn't delete (permissions, race) — fail safe by ignoring it.
            pass
    return False


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


def enter_maintenance(reason: str = "manual", entered_by: str = "unknown",
                       ttl_seconds: Optional[int] = None) -> dict:
    """Activate maintenance mode. Returns the flag data written.

    ttl_seconds: how long before this flag is treated as stale and
    auto-cleared (default _DEFAULT_TTL_SECONDS, 4h — same as the VRAM
    tenant guard). Pass a larger value for a known-longer operation.
    """
    _SHARED_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "active": True,
        "entered_at": datetime.now(timezone.utc).isoformat(),
        "entered_by": entered_by,
        "reason": reason,
        "ttl_seconds": int(ttl_seconds) if ttl_seconds is not None else _DEFAULT_TTL_SECONDS,
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

    _clear_flag_files()

    return {
        "active": False,
        "exited_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration,
        "previous": info,
    }

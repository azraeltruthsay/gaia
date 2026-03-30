"""Defensive Meditation Controller — time-boxed chaos testing mode."""
import json
import logging
import os
import time
from pathlib import Path

log = logging.getLogger("gaia-monkey.meditation")

DEFENSIVE_MEDITATION_MAX = 1800  # 30 minutes
_defensive_meditation_start: float | None = None

_FLAG_PATH = Path(os.environ.get("SHARED_DIR", "/shared")) / "doctor" / "defensive_meditation.json"


def enter():
    global _defensive_meditation_start
    _defensive_meditation_start = time.monotonic()
    log.info("🧘 DEFENSIVE MEDITATION entered — chaos restrictions relaxed for %ds", DEFENSIVE_MEDITATION_MAX)
    _write_flag(True)


def exit_meditation():
    global _defensive_meditation_start
    _defensive_meditation_start = None
    log.info("🧘 DEFENSIVE MEDITATION ended — normal restrictions restored")
    _write_flag(False)


def is_active() -> bool:
    global _defensive_meditation_start
    if _defensive_meditation_start is None:
        return False
    elapsed = time.monotonic() - _defensive_meditation_start
    if elapsed > DEFENSIVE_MEDITATION_MAX:
        exit_meditation()
        return False
    return True


def get_status() -> dict:
    active = is_active()
    elapsed = 0.0
    if active and _defensive_meditation_start is not None:
        elapsed = time.monotonic() - _defensive_meditation_start
    return {
        "active": active,
        "elapsed_seconds": round(elapsed, 1) if active else 0,
        "max_duration": DEFENSIVE_MEDITATION_MAX,
    }


def _write_flag(active: bool):
    try:
        _FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {"active": active}
        if active:
            data["started"] = time.time()
            data["max_duration"] = DEFENSIVE_MEDITATION_MAX
        _FLAG_PATH.write_text(json.dumps(data))
    except Exception:
        log.debug("Failed to write meditation flag", exc_info=True)

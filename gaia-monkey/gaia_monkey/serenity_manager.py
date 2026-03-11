"""Serenity State Manager — tracks proven resilience through chaos drills."""
import json
import logging
import os
import threading
import time
from pathlib import Path

log = logging.getLogger("gaia-monkey.serenity")

SERENITY_THRESHOLD = 5.0
SERENITY_FILE = Path(os.environ.get("SHARED_DIR", "/shared")) / "doctor" / "serenity.json"
SERENITY_WEIGHTS = {
    "vital_recovery": 2.0,
    "standard_recovery": 0.5,
    "service_recovery": 0.5,
    "cognitive_validation": 2.0,
    "test_pass": 0.5,
}

_lock = threading.Lock()
_serenity_active: bool = False
_serenity_score: float = 0.0
_serenity_achieved_at: float | None = None
_serenity_reason: str = ""


def record_recovery(category: str, detail: str = "", meditation_active: bool = True):
    """Record a successful recovery. Only counts during active Defensive Meditation."""
    global _serenity_score
    if not meditation_active:
        return
    with _lock:
        weight = SERENITY_WEIGHTS.get(category, 0.5)
        _serenity_score += weight
        log.info("🪷 Recovery recorded: %s (+%.1f) — serenity score: %.1f/%.1f%s",
                 category, weight, _serenity_score, SERENITY_THRESHOLD,
                 f" ({detail})" if detail else "")
        if _serenity_score >= SERENITY_THRESHOLD and not _serenity_active:
            _enter_serenity(f"Earned during Defensive Meditation: {_serenity_score:.1f} points from tested recoveries")


def _enter_serenity(reason: str):
    global _serenity_active, _serenity_achieved_at, _serenity_reason
    _serenity_active = True
    _serenity_achieved_at = time.time()
    _serenity_reason = reason
    log.info("🪷 SERENITY ACHIEVED — %s", reason)
    _persist()


def break_serenity(reason: str):
    """Break Serenity due to a vital organ issue."""
    global _serenity_active, _serenity_score, _serenity_achieved_at, _serenity_reason
    with _lock:
        if not _serenity_active:
            return
        duration = time.time() - (_serenity_achieved_at or time.time())
        log.warning("🪷 SERENITY BROKEN after %.0fs — %s", duration, reason)
        _serenity_active = False
        _serenity_score = 0.0
        _serenity_achieved_at = None
        _serenity_reason = ""
        _persist()


def reset_serenity():
    global _serenity_active, _serenity_score, _serenity_achieved_at, _serenity_reason
    with _lock:
        _serenity_active = False
        _serenity_score = 0.0
        _serenity_achieved_at = None
        _serenity_reason = ""
        _persist()


def is_serene() -> bool:
    return _serenity_active


def get_report() -> dict:
    with _lock:
        return {
            "serene": _serenity_active,
            "score": round(_serenity_score, 1),
            "threshold": SERENITY_THRESHOLD,
            "achieved_at": _serenity_achieved_at,
            "reason": _serenity_reason,
        }


def _persist():
    try:
        SERENITY_FILE.parent.mkdir(parents=True, exist_ok=True)
        SERENITY_FILE.write_text(json.dumps({
            "serene": _serenity_active,
            "score": _serenity_score,
            "threshold": SERENITY_THRESHOLD,
            "achieved_at": _serenity_achieved_at,
            "reason": _serenity_reason,
        }))
    except Exception:
        log.debug("Failed to write serenity file", exc_info=True)

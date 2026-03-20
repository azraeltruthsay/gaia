"""Heartbeat Time Check — the simplest canary that proves GAIA is alive.

Periodically asks Nano "What time is it?" and validates the response
matches the actual time within tolerance. If GAIA can tell you the
correct time, the whole chain works: prompt injection → model inference
→ response validation → delivery.

Runs as a background daemon thread. Results written to shared state
for dashboard visibility and immune system consumption.

Usage:
    from gaia_common.utils.heartbeat_time_check import start_heartbeat, stop_heartbeat

    start_heartbeat(interval_seconds=300)  # every 5 minutes
    # ...
    stop_heartbeat()
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen

logger = logging.getLogger("GAIA.Heartbeat")

# ── Configuration ────────────────────────────────────────────────────────

DEFAULT_INTERVAL = int(os.environ.get("HEARTBEAT_INTERVAL_SECONDS", "300"))
MAX_DRIFT_MINUTES = int(os.environ.get("HEARTBEAT_MAX_DRIFT_MINUTES", "2"))
CORE_ENDPOINT = os.environ.get("CORE_ENDPOINT", "http://gaia-core:6415")
NANO_ENDPOINT = os.environ.get("NANO_ENDPOINT", "http://gaia-nano:8080")
STATE_PATH = Path(os.environ.get("SHARED_DIR", "/shared")) / "heartbeat" / "time_check.json"


# ── Time Validation ──────────────────────────────────────────────────────

def _extract_time_from_response(text: str) -> Optional[tuple]:
    """Extract hours and minutes from a time string like '3:47 PM'."""
    match = re.search(r'(\d{1,2}):(\d{2})\s*(AM|PM)', text, re.IGNORECASE)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    ampm = match.group(3).upper()
    if ampm == "PM" and hour != 12:
        hour += 12
    elif ampm == "AM" and hour == 12:
        hour = 0
    return hour, minute


def _get_local_time() -> tuple:
    """Get current local time as (hour_24, minute)."""
    tz_offset = int(os.environ.get("GAIA_LOCAL_TZ_OFFSET", "-7"))
    local_tz = timezone(timedelta(hours=tz_offset))
    now = datetime.now(local_tz)
    return now.hour, now.minute


def validate_time_response(response_text: str, max_drift: int = MAX_DRIFT_MINUTES) -> Dict[str, Any]:
    """Check if a time response matches actual time within tolerance.

    Returns dict with: valid, actual_time, claimed_time, drift_minutes, error
    """
    actual_h, actual_m = _get_local_time()
    actual_str = f"{actual_h % 12 or 12}:{actual_m:02d} {'AM' if actual_h < 12 else 'PM'}"

    extracted = _extract_time_from_response(response_text)
    if not extracted:
        return {
            "valid": False,
            "actual_time": actual_str,
            "claimed_time": None,
            "drift_minutes": None,
            "error": "no time found in response",
        }

    claimed_h, claimed_m = extracted
    claimed_str = f"{claimed_h % 12 or 12}:{claimed_m:02d} {'AM' if claimed_h < 12 else 'PM'}"

    # Calculate drift in minutes
    actual_minutes = actual_h * 60 + actual_m
    claimed_minutes = claimed_h * 60 + claimed_m
    drift = abs(claimed_minutes - actual_minutes)
    # Handle midnight wrap
    if drift > 720:
        drift = 1440 - drift

    return {
        "valid": drift <= max_drift,
        "actual_time": actual_str,
        "claimed_time": claimed_str,
        "drift_minutes": drift,
        "error": None if drift <= max_drift else f"drift {drift}min exceeds max {max_drift}min",
    }


# ── Heartbeat Check ──────────────────────────────────────────────────────

def run_time_check(endpoint: str = NANO_ENDPOINT) -> Dict[str, Any]:
    """Ask Nano what time it is and validate the response.

    Returns a heartbeat result dict.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        # Ask via Nano's OpenAI-compatible chat endpoint
        payload = json.dumps({
            "messages": [
                {"role": "system", "content": "You are GAIA. Answer with just the time."},
                {"role": "user", "content": "What time is it?"},
            ],
            "max_tokens": 64,
            "temperature": 0.0,
        }).encode()

        req = Request(
            f"{endpoint}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())

        response_text = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        validation = validate_time_response(response_text)

        return {
            "timestamp": timestamp,
            "status": "pass" if validation["valid"] else "fail",
            "response": response_text,
            "actual_time": validation["actual_time"],
            "claimed_time": validation["claimed_time"],
            "drift_minutes": validation["drift_minutes"],
            "error": validation["error"],
            "endpoint": endpoint,
        }

    except Exception as e:
        return {
            "timestamp": timestamp,
            "status": "error",
            "response": None,
            "actual_time": None,
            "claimed_time": None,
            "drift_minutes": None,
            "error": str(e),
            "endpoint": endpoint,
        }


# ── Shared State ─────────────────────────────────────────────────────────

def _write_state(result: Dict[str, Any], history_max: int = 20) -> None:
    """Write heartbeat result to shared state file."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load existing state
    state = {"current": None, "history": [], "stats": {"total": 0, "passes": 0, "fails": 0, "errors": 0}}
    if STATE_PATH.exists():
        try:
            state = json.loads(STATE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    # Update
    state["current"] = result
    state["history"].insert(0, result)
    state["history"] = state["history"][:history_max]
    state["stats"]["total"] += 1
    if result["status"] == "pass":
        state["stats"]["passes"] += 1
    elif result["status"] == "fail":
        state["stats"]["fails"] += 1
    else:
        state["stats"]["errors"] += 1

    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def read_state() -> Dict[str, Any]:
    """Read current heartbeat state."""
    if not STATE_PATH.exists():
        return {"current": None, "history": [], "stats": {"total": 0, "passes": 0, "fails": 0, "errors": 0}}
    try:
        return json.loads(STATE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {"current": None, "history": [], "stats": {"total": 0, "passes": 0, "fails": 0, "errors": 0}}


# ── Background Thread ────────────────────────────────────────────────────

_heartbeat_thread: Optional[threading.Thread] = None
_heartbeat_stop = threading.Event()


def _heartbeat_loop(interval: int, endpoint: str) -> None:
    """Background loop that runs time checks at regular intervals."""
    logger.info("Heartbeat started: interval=%ds, endpoint=%s", interval, endpoint)
    while not _heartbeat_stop.is_set():
        try:
            result = run_time_check(endpoint)
            _write_state(result)

            if result["status"] == "pass":
                logger.debug(
                    "Heartbeat: PASS — %s (drift %s min)",
                    result["claimed_time"], result["drift_minutes"],
                )
            elif result["status"] == "fail":
                logger.warning(
                    "Heartbeat: FAIL — claimed %s, actual %s (drift %s min)",
                    result["claimed_time"], result["actual_time"], result["drift_minutes"],
                )
                # Emit to CodeMind detect queue for investigation
                try:
                    from gaia_common.utils.codemind_detector import emit_detection
                    emit_detection(
                        source="immune_irritation",
                        issue_type="timing_error",
                        file_path="gaia-core/gaia_core/utils/prompt_builder.py",
                        description=(
                            f"Heartbeat time check failed: Nano claimed {result['claimed_time']} "
                            f"but actual is {result['actual_time']} (drift {result['drift_minutes']}min)"
                        ),
                        severity="warn",
                    )
                except ImportError:
                    pass
            else:
                logger.warning("Heartbeat: ERROR — %s", result["error"])

        except Exception:
            logger.debug("Heartbeat check failed", exc_info=True)

        _heartbeat_stop.wait(interval)

    logger.info("Heartbeat stopped")


def start_heartbeat(
    interval_seconds: int = DEFAULT_INTERVAL,
    endpoint: str = NANO_ENDPOINT,
) -> None:
    """Start the background heartbeat thread."""
    global _heartbeat_thread
    if _heartbeat_thread is not None and _heartbeat_thread.is_alive():
        logger.debug("Heartbeat already running")
        return
    _heartbeat_stop.clear()
    _heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(interval_seconds, endpoint),
        daemon=True,
        name="gaia-heartbeat",
    )
    _heartbeat_thread.start()


def stop_heartbeat() -> None:
    """Stop the background heartbeat thread."""
    global _heartbeat_thread
    _heartbeat_stop.set()
    if _heartbeat_thread is not None:
        _heartbeat_thread.join(timeout=10)
        _heartbeat_thread = None

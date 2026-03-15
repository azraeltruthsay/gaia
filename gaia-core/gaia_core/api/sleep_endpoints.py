"""
Sleep cycle HTTP endpoints for gaia-core.

Follows the existing pattern from gpu_endpoints.py:
  - Separate router file with APIRouter(prefix="/sleep")
  - Registered in main.py via app.include_router(sleep_router)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("GAIA.SleepEndpoints")

router = APIRouter(prefix="/sleep", tags=["sleep"])


@router.post("/wake")
async def receive_wake_signal(request: Request):
    """Receive wake signal from gaia-web.

    Called when the first message is queued during sleep.
    """
    manager = getattr(request.app.state, "sleep_wake_manager", None)
    if manager is None:
        return JSONResponse(
            status_code=503,
            content={"error": "SleepWakeManager not initialized"},
        )

    manager.receive_wake_signal()

    return {
        "received": True,
        "state": manager.get_state().value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/voice-state")
async def voice_state(request: Request):
    """Notify gaia-core when GAIA joins/leaves a Discord voice channel.

    Body: {"connected": true/false}

    When connected=True, triggers an implicit wake signal so Prime starts
    booting while audio stays alive for Lite-based stalling responses.
    When connected=False during sleep, the deferred audio sleep signal fires.
    """
    manager = getattr(request.app.state, "sleep_wake_manager", None)
    if manager is None:
        return JSONResponse(
            status_code=503,
            content={"error": "SleepWakeManager not initialized"},
        )

    body = await request.json()
    connected = body.get("connected", False)
    manager.set_voice_active(connected)

    return {
        "accepted": True,
        "voice_active": manager.voice_active,
        "state": manager.get_state().value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/status")
async def get_sleep_status(request: Request):
    """Get current sleep/wake state and task info."""
    manager = getattr(request.app.state, "sleep_wake_manager", None)
    if manager is None:
        return JSONResponse(
            status_code=503,
            content={"error": "SleepWakeManager not initialized"},
        )

    return manager.get_status()


@router.post("/study-handoff")
async def study_handoff(request: Request):
    """Receive study handoff signal from orchestrator.

    Body: {"direction": "prime_to_study"|"study_to_prime", "handoff_id": "..."}
    """
    manager = getattr(request.app.state, "sleep_wake_manager", None)
    if manager is None:
        return JSONResponse(
            status_code=503,
            content={"error": "SleepWakeManager not initialized"},
        )

    body = await request.json()
    direction = body.get("direction")
    handoff_id = body.get("handoff_id", "unknown")

    if direction == "prime_to_study":
        ok = manager.enter_dreaming(handoff_id)
    elif direction == "study_to_prime":
        ok = manager.exit_dreaming()
    else:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid direction: {direction}"},
        )

    return {
        "accepted": ok,
        "state": manager.get_state().value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/distracted-check")
async def distracted_check(request: Request):
    """Check if GAIA is in a state that warrants a canned response.

    Returns the current state and, if applicable, a canned_response that
    gaia-web should send instead of forwarding the message to the model.
    """
    manager = getattr(request.app.state, "sleep_wake_manager", None)
    if manager is None:
        return JSONResponse(
            status_code=503,
            content={"error": "SleepWakeManager not initialized"},
        )

    canned = manager.get_canned_response()
    return {
        "state": manager.get_state().value,
        "canned_response": canned,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/toggle")
async def toggle_auto_sleep(request: Request):
    """Enable or disable automatic sleep transitions.

    Body: {"enabled": true/false}
    """
    manager = getattr(request.app.state, "sleep_wake_manager", None)
    if manager is None:
        return JSONResponse(
            status_code=503,
            content={"error": "SleepWakeManager not initialized"},
        )

    body = await request.json()
    enabled = body.get("enabled", True)
    manager.set_auto_sleep(enabled)

    return {
        "accepted": True,
        "auto_sleep_enabled": manager.auto_sleep_enabled,
        "state": manager.get_state().value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/force")
async def force_sleep(request: Request):
    """Immediately trigger sleep transition (ACTIVE → DROWSY → ASLEEP)."""
    manager = getattr(request.app.state, "sleep_wake_manager", None)
    if manager is None:
        return JSONResponse(
            status_code=503,
            content={"error": "SleepWakeManager not initialized"},
        )

    if manager.get_state().value != "active":
        return JSONResponse(
            status_code=409,
            content={
                "error": f"Cannot force sleep from state: {manager.get_state().value}",
                "state": manager.get_state().value,
            },
        )

    entered_sleep = manager.initiate_drowsy()

    return {
        "accepted": True,
        "entered_sleep": entered_sleep,
        "state": manager.get_state().value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/config")
async def get_sleep_config(request: Request):
    """Return current sleep configuration for the dashboard."""
    manager = getattr(request.app.state, "sleep_wake_manager", None)
    if manager is None:
        return JSONResponse(
            status_code=503,
            content={"error": "SleepWakeManager not initialized"},
        )

    sleep_cfg = getattr(manager.config, "SLEEP_CYCLE", None) or {}
    threshold = sleep_cfg.get("idle_threshold_minutes", 30) if isinstance(sleep_cfg, dict) else 30

    return {
        "auto_sleep_enabled": manager.auto_sleep_enabled,
        "idle_threshold_minutes": threshold,
        "state": manager.get_state().value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/shutdown")
async def shutdown(request: Request):
    """Initiate graceful shutdown — transitions to OFFLINE and stops the loop."""
    sleep_loop = getattr(request.app.state, "sleep_cycle_loop", None)
    if sleep_loop is None:
        return JSONResponse(
            status_code=503,
            content={"error": "SleepCycleLoop not initialized"},
        )

    sleep_loop.initiate_shutdown()

    return {
        "accepted": True,
        "state": "offline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Prime Wake Trigger Configuration ────────────────────────────────────────

def _get_wake_config(config) -> dict:
    """Read PRIME_WAKE from the loaded constants (fallback to defaults)."""
    raw = getattr(config, "constants", {})
    return raw.get("PRIME_WAKE", {"discord_typing": True, "workstation_activity": False})


@router.get("/wake-config")
async def get_wake_config(request: Request):
    """Return current PRIME_WAKE trigger settings."""
    config = getattr(request.app.state, "config", None)
    if config is None:
        return {"discord_typing": True, "workstation_activity": False}
    return _get_wake_config(config)


@router.post("/wake-toggle")
async def wake_toggle(request: Request):
    """Toggle a specific prime wake trigger on/off.

    Body: {"trigger": "discord_typing"|"workstation_activity", "enabled": bool}
    Persists the change to gaia_constants.json.
    """
    config = getattr(request.app.state, "config", None)
    if config is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Config not initialized"},
        )

    body = await request.json()
    trigger = body.get("trigger")
    enabled = body.get("enabled")

    valid_triggers = ("discord_typing", "workstation_activity")
    if trigger not in valid_triggers:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid trigger: {trigger}. Must be one of {valid_triggers}"},
        )
    if not isinstance(enabled, bool):
        return JSONResponse(
            status_code=400,
            content={"error": "'enabled' must be a boolean"},
        )

    # Update in-memory constants
    raw = getattr(config, "constants", {})
    if "PRIME_WAKE" not in raw:
        raw["PRIME_WAKE"] = {"discord_typing": True, "workstation_activity": False}
    raw["PRIME_WAKE"][trigger] = enabled

    # Persist to disk
    source_path = getattr(config, "_source_path", None)
    if source_path:
        try:
            with open(source_path, "r", encoding="utf-8") as f:
                disk_data = json.load(f)
            if "PRIME_WAKE" not in disk_data:
                disk_data["PRIME_WAKE"] = {"discord_typing": True, "workstation_activity": False}
            disk_data["PRIME_WAKE"][trigger] = enabled
            with open(source_path, "w", encoding="utf-8") as f:
                json.dump(disk_data, f, indent=2, ensure_ascii=False)
                f.write("\n")
            logger.info("PRIME_WAKE.%s → %s (persisted to %s)", trigger, enabled, source_path)
        except Exception:
            logger.warning("Failed to persist PRIME_WAKE to disk", exc_info=True)
    else:
        logger.warning("No _source_path on config — PRIME_WAKE change is in-memory only")

    return {
        "accepted": True,
        "trigger": trigger,
        "enabled": enabled,
        "wake_config": raw.get("PRIME_WAKE", {}),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/wake-activity")
async def wake_activity(request: Request):
    """Receive wake signal from host-side workstation activity monitor.

    Only triggers wake if workstation_activity toggle is enabled AND state is asleep.
    """
    config = getattr(request.app.state, "config", None)
    wake_cfg = _get_wake_config(config) if config else {}

    if not wake_cfg.get("workstation_activity", False):
        return JSONResponse(
            status_code=200,
            content={"accepted": False, "reason": "workstation_activity trigger is disabled"},
        )

    manager = getattr(request.app.state, "sleep_wake_manager", None)
    if manager is None:
        return JSONResponse(
            status_code=503,
            content={"error": "SleepWakeManager not initialized"},
        )

    state = manager.get_state().value
    if state not in ("asleep", "drowsy"):
        return {
            "accepted": False,
            "reason": f"Already in state: {state}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    manager.receive_wake_signal()
    logger.info("Workstation activity wake trigger fired (was %s)", state)

    return {
        "accepted": True,
        "state": manager.get_state().value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

"""
Sleep cycle HTTP endpoints for gaia-core.

Follows the existing pattern from gpu_endpoints.py:
  - Separate router file with APIRouter(prefix="/sleep")
  - Registered in main.py via app.include_router(sleep_router)
"""

from __future__ import annotations

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

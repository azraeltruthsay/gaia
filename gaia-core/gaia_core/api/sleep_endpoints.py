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

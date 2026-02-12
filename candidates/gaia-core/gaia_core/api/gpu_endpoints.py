"""
GPU management endpoints for gaia-core.

These endpoints allow the orchestrator to manage gaia-core's model pool
when GPU ownership changes. The orchestrator handles the actual container
stop/start for gaia-prime; these endpoints only manage the model pool state.

Protocol:
    1. Orchestrator stops gaia-prime container (frees all VRAM)
    2. Orchestrator POSTs /gpu/release → gaia-core demotes gpu_prime, activates fallback chain
    3. (study uses GPU for training)
    4. Orchestrator starts gaia-prime container (loads model, ~40-60s)
    5. Orchestrator POSTs /gpu/reclaim → gaia-core restores gpu_prime in model pool
"""

import logging
import os
import time
from typing import Optional

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("GAIA.Core.GPU")

router = APIRouter(prefix="/gpu", tags=["gpu"])

# ── State ──────────────────────────────────────────────────────────────────

_gpu_state = {
    "status": "active",       # active | released | releasing | reclaiming | error
    "released_at": None,      # ISO timestamp
    "reclaimed_at": None,
}


def _prime_endpoint() -> str:
    """Resolve gaia-prime's base URL."""
    return (
        os.getenv("PRIME_ENDPOINT", "http://gaia-prime-candidate:7777")
        .rstrip("/")
    )


def _get_model_pool():
    """Get the ModelPool instance from the running cognitive system."""
    from gaia_core.main import _ai_manager
    if _ai_manager is None:
        raise HTTPException(status_code=503, detail="Cognitive system not initialized")
    return _ai_manager.model_pool


# ── Request/Response Models ────────────────────────────────────────────────

class GPUReleaseRequest(BaseModel):
    reason: str = ""


class GPUReclaimRequest(BaseModel):
    adapter_name: Optional[str] = None


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.get("/status")
async def gpu_status():
    """Report current GPU state."""
    model_pool = _get_model_pool()
    gpu_prime_loaded = "gpu_prime" in model_pool.models
    prime_status = model_pool.model_status.get("gpu_prime", "unknown")

    # Check if prime is actually reachable
    prime_reachable = False
    if _gpu_state["status"] == "active":
        try:
            resp = requests.get(f"{_prime_endpoint()}/health", timeout=3)
            prime_reachable = resp.status_code == 200
        except Exception:
            pass

    return {
        "gpu_state": _gpu_state["status"],
        "gpu_prime_loaded": gpu_prime_loaded,
        "gpu_prime_status": prime_status,
        "prime_reachable": prime_reachable,
        "released_at": _gpu_state["released_at"],
        "reclaimed_at": _gpu_state["reclaimed_at"],
        "prime_endpoint": _prime_endpoint(),
    }


@router.post("/release")
async def gpu_release(request: GPUReleaseRequest = GPUReleaseRequest()):
    """
    Update model pool after GPU release.

    Called by the orchestrator AFTER it has already stopped the
    gaia-prime container. This endpoint only manages gaia-core's
    internal model pool state — it does NOT stop the container.
    """
    if _gpu_state["status"] == "released":
        return {"ok": True, "message": "Already released", "state": _gpu_state}

    if _gpu_state["status"] not in ("active", "error"):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot release GPU in state '{_gpu_state['status']}'"
        )

    _gpu_state["status"] = "releasing"

    try:
        # Remove gpu_prime from the model pool so the fallback chain kicks in
        model_pool = _get_model_pool()
        if "gpu_prime" in model_pool.models:
            model_pool.set_status("gpu_prime", "stopped")
            _stashed_model = model_pool.models.pop("gpu_prime", None)
            _stashed_status = model_pool.model_status.pop("gpu_prime", None)
            # Store reference so we can restore without full reload
            _gpu_state["_stashed_model"] = _stashed_model
            _gpu_state["_stashed_status"] = _stashed_status

            # Re-promote aliases so 'prime' falls back to cpu_prime or other
            if "prime" in model_pool.models and model_pool.models.get("prime") is _stashed_model:
                model_pool.models.pop("prime", None)
                model_pool.model_status.pop("prime", None)
                model_pool._promote_prime_aliases()
                logger.info("Demoted gpu_prime from 'prime' alias; fallback chain active")

        _gpu_state["status"] = "released"
        _gpu_state["released_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        logger.info(f"GPU released. Reason: {request.reason or 'none'}. Fallback chain active.")
        return {"ok": True, "message": "Model pool updated, fallback chain active", "state": _gpu_state.copy()}

    except Exception as e:
        _gpu_state["status"] = "error"
        logger.exception(f"GPU release failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reclaim")
async def gpu_reclaim(request: GPUReclaimRequest = GPUReclaimRequest()):
    """
    Restore gpu_prime in the model pool after GPU reclaim.

    Called by the orchestrator AFTER it has already started the
    gaia-prime container and confirmed it is healthy. This endpoint
    only restores the model pool state.
    """
    if _gpu_state["status"] == "active":
        return {"ok": True, "message": "Already active", "state": _gpu_state}

    if _gpu_state["status"] not in ("released", "error"):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot reclaim GPU in state '{_gpu_state['status']}'"
        )

    _gpu_state["status"] = "reclaiming"
    prime_url = _prime_endpoint()

    try:
        # Verify prime is actually healthy before restoring the model pool
        healthy = False
        for attempt in range(5):
            try:
                health = requests.get(f"{prime_url}/health", timeout=5)
                if health.status_code == 200:
                    healthy = True
                    logger.info(f"Prime health OK (attempt {attempt + 1})")
                    break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(2)

        if not healthy:
            logger.warning("Prime health check failed — restoring model pool anyway")

        # Restore gpu_prime in the model pool
        model_pool = _get_model_pool()
        stashed = _gpu_state.pop("_stashed_model", None)
        stashed_status = _gpu_state.pop("_stashed_status", None)

        if stashed is not None:
            model_pool.models["gpu_prime"] = stashed
            model_pool.model_status["gpu_prime"] = stashed_status or "idle"
            model_pool._promote_prime_aliases()
            logger.info("Restored gpu_prime in model pool and re-promoted to 'prime'")
        else:
            # If no stashed model, try lazy reload
            logger.info("No stashed model; attempting lazy reload of gpu_prime")
            model_pool.ensure_model_loaded("gpu_prime", force=True)
            model_pool._promote_prime_aliases()

        _gpu_state["status"] = "active"
        _gpu_state["reclaimed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        logger.info("GPU reclaimed — prime inference restored")
        return {"ok": True, "message": "GPU reclaimed, prime active", "state": _gpu_state.copy()}

    except Exception as e:
        _gpu_state["status"] = "error"
        logger.exception(f"GPU reclaim failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

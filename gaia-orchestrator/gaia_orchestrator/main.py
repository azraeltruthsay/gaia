"""
GAIA Orchestrator - FastAPI Application.

Central coordination service for GPU resources and container lifecycle.
"""

import asyncio
import logging
import re
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from .config import get_config
from .state import get_state_manager, StateManager
from .models.schemas import (
    GPUStatus,
    GPUOwner,
    GPUAcquireRequest,
    GPUAcquireResponse,
    ContainerStatus,
    ContainerStartRequest,
    ContainerSwapRequest,
    HandoffRequest,
    HandoffStatus,
    HandoffType,
    OracleNotification,
    Notification,
    NotificationType,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("GAIA.Orchestrator")

# Suppress health check access log spam
try:
    from gaia_common.utils import install_health_check_filter
    install_health_check_filter()
except ImportError:
    pass  # gaia_common not available

# Global references
_state_manager: Optional[StateManager] = None
_gpu_manager = None
_docker_manager = None
_handoff_manager = None
_notification_manager = None
_health_watchdog = None
_gpu_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global _state_manager, _gpu_manager, _docker_manager, _handoff_manager, _notification_manager, _health_watchdog

    logger.info("GAIA Orchestrator starting up...")
    config = get_config()

    # Initialize state manager
    _state_manager = await get_state_manager()
    logger.info(f"State persistence initialized at {config.state_dir}")

    # Initialize managers (lazy imports to avoid circular deps)
    try:
        from .gpu_manager import GPUManager
        _gpu_manager = GPUManager(_state_manager)
        logger.info("GPU manager initialized")
    except ImportError:
        logger.warning("GPU manager not available yet")

    try:
        from .docker_manager import DockerManager
        _docker_manager = DockerManager(_state_manager)
        logger.info("Docker manager initialized")
    except ImportError:
        logger.warning("Docker manager not available yet")

    try:
        from .handoff_manager import HandoffManager
        _handoff_manager = HandoffManager(_state_manager, _gpu_manager)
        logger.info("Handoff manager initialized")
    except ImportError:
        logger.warning("Handoff manager not available yet")

    try:
        from .notification_manager import NotificationManager
        _notification_manager = NotificationManager()
        logger.info("Notification manager initialized")
    except ImportError:
        logger.warning("Notification manager not available yet")

    # Start health watchdog (polls gaia-core + gaia-prime every 30s)
    try:
        from .health_watchdog import HealthWatchdog
        _health_watchdog = HealthWatchdog(notification_manager=_notification_manager)
        await _health_watchdog.start()
        logger.info("Health watchdog started")
    except ImportError:
        logger.warning("Health watchdog not available yet")

    logger.info("GAIA Orchestrator ready")
    yield

    # Shutdown
    logger.info("GAIA Orchestrator shutting down...")
    if _health_watchdog:
        await _health_watchdog.stop()
    if _state_manager:
        await _state_manager.save()


app = FastAPI(
    title="GAIA Orchestrator",
    description="GPU and Container Lifecycle Coordination Service",
    version="0.1.0",
    lifespan=lifespan,
)


# =============================================================================
# Health & Status Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "gaia-orchestrator",
        }
    )


@app.get("/")
async def root():
    """Root endpoint with API overview."""
    return {
        "service": "gaia-orchestrator",
        "description": "GAIA GPU and Container Lifecycle Coordination",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "gpu": {
                "status": "GET /gpu/status",
                "acquire": "POST /gpu/acquire",
                "release": "POST /gpu/release",
                "wait": "POST /gpu/wait",
            },
            "containers": {
                "status": "GET /containers/status",
                "live_stop": "POST /containers/live/stop",
                "live_start": "POST /containers/live/start",
                "candidate_stop": "POST /containers/candidate/stop",
                "candidate_start": "POST /containers/candidate/start",
                "swap": "POST /containers/swap",
            },
            "sleep": {
                "gpu_sleep": "POST /gpu/sleep",
                "gpu_wake": "POST /gpu/wake",
            },
            "handoff": {
                "prime_to_study": "POST /handoff/prime-to-study",
                "study_to_prime": "POST /handoff/study-to-prime",
                "status": "GET /handoff/{handoff_id}/status",
            },
            "notifications": {
                "oracle_fallback": "POST /notify/oracle-fallback",
                "websocket": "WS /ws/notifications",
            },
        }
    }


@app.get("/status")
async def get_status():
    """Get complete orchestrator status."""
    if _state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    state = _state_manager.state
    return {
        "service": "gaia-orchestrator",
        "status": "operational",
        "gpu": state.gpu.model_dump(),
        "containers": state.containers.model_dump(),
        "active_handoff": state.active_handoff.model_dump() if state.active_handoff else None,
        "last_updated": state.last_updated.isoformat(),
    }


# =============================================================================
# GPU Management Endpoints
# =============================================================================

@app.get("/gpu/status")
async def get_gpu_status() -> GPUStatus:
    """Get current GPU ownership status."""
    if _state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    gpu_status = await _state_manager.get_gpu_status()

    # If GPU manager is available, enrich with memory info
    if _gpu_manager:
        try:
            memory_info = await _gpu_manager.get_memory_info()
            gpu_status.memory = memory_info
        except Exception as e:
            logger.warning(f"Could not get GPU memory info: {e}")

    return gpu_status


async def _acquire_gpu_inner(request: GPUAcquireRequest) -> GPUAcquireResponse:
    """Core GPU acquire logic (must be called under _gpu_lock)."""
    current = await _state_manager.get_gpu_status()

    # If already owned by someone else, queue the request
    if current.owner != GPUOwner.NONE and current.owner != request.requester:
        position = await _state_manager.add_to_gpu_queue(request.requester.value)
        return GPUAcquireResponse(
            success=False,
            message=f"GPU owned by {current.owner.value}, queued at position {position}",
            queue_position=position,
        )

    # Grant ownership
    import uuid
    lease_id = str(uuid.uuid4())
    await _state_manager.set_gpu_owner(request.requester, lease_id, request.reason)

    logger.info(f"GPU acquired by {request.requester.value} (lease: {lease_id[:8]}...)")

    return GPUAcquireResponse(
        success=True,
        lease_id=lease_id,
        message=f"GPU ownership granted to {request.requester.value}",
    )


@app.post("/gpu/acquire")
async def acquire_gpu(request: GPUAcquireRequest) -> GPUAcquireResponse:
    """Request GPU ownership."""
    if _state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    async with _gpu_lock:
        return await _acquire_gpu_inner(request)


@app.post("/gpu/release")
async def release_gpu(lease_id: Optional[str] = None):
    """Release GPU ownership."""
    if _state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    async with _gpu_lock:
        current = await _state_manager.get_gpu_status()

        # Validate lease if provided
        if lease_id and current.lease_id and lease_id != current.lease_id:
            raise HTTPException(
                status_code=403,
                detail="Invalid lease ID - cannot release GPU owned by another process"
            )

        previous_owner = current.owner
        await _state_manager.release_gpu()

        logger.info(f"GPU released by {previous_owner.value}")

        # Check if anyone is waiting in queue
        if current.queue:
            next_requester = current.queue[0]
            await _state_manager.remove_from_gpu_queue(next_requester)
            # Note: The waiting process should be polling /gpu/status or /gpu/wait

        return {
            "success": True,
            "message": f"GPU released by {previous_owner.value}",
            "queue_length": len(current.queue),
        }


@app.post("/gpu/wait")
async def wait_for_gpu(request: GPUAcquireRequest) -> GPUAcquireResponse:
    """Wait for GPU to become available, then acquire it."""
    if _state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    import asyncio

    timeout = request.timeout_seconds
    poll_interval = 1.0
    elapsed = 0.0

    while elapsed < timeout:
        current = await _state_manager.get_gpu_status()

        if current.owner == GPUOwner.NONE:
            # GPU is free, try to acquire under lock
            async with _gpu_lock:
                return await _acquire_gpu_inner(request)

        if current.owner == request.requester:
            # Already own it
            return GPUAcquireResponse(
                success=True,
                lease_id=current.lease_id,
                message="Already own GPU",
            )

        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    # Timeout
    return GPUAcquireResponse(
        success=False,
        message=f"Timeout waiting for GPU after {timeout}s",
        queue_position=await _state_manager.add_to_gpu_queue(request.requester.value),
    )


# =============================================================================
# Container Lifecycle Endpoints
# =============================================================================

@app.get("/containers/status")
async def get_container_status() -> ContainerStatus:
    """Get status of all containers."""
    if _state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    # If docker manager available, refresh from Docker
    if _docker_manager:
        try:
            status = await _docker_manager.get_status()
            await _state_manager.update_container_status(status)
            return status
        except Exception as e:
            logger.warning(f"Could not refresh container status: {e}")

    return await _state_manager.get_container_status()


@app.post("/containers/live/stop")
async def stop_live_stack():
    """Stop the live stack and release GPU."""
    if _docker_manager is None:
        raise HTTPException(status_code=501, detail="Docker manager not available")

    try:
        result = await _docker_manager.stop_live()
        await _state_manager.release_gpu()
        return {"success": True, "message": "Live stack stopped", "details": result}
    except Exception as e:
        logger.exception(f"Error stopping live stack: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/containers/live/start")
async def start_live_stack(request: ContainerStartRequest):
    """Start the live stack."""
    if _docker_manager is None:
        raise HTTPException(status_code=501, detail="Docker manager not available")

    try:
        result = await _docker_manager.start_live(gpu_enabled=request.gpu_enabled)
        if request.gpu_enabled:
            import uuid
            await _state_manager.set_gpu_owner(
                GPUOwner.CORE,
                str(uuid.uuid4()),
                "live_stack_start"
            )
        return {"success": True, "message": "Live stack started", "details": result}
    except Exception as e:
        logger.exception(f"Error starting live stack: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/containers/candidate/stop")
async def stop_candidate_stack():
    """Stop the candidate stack."""
    if _docker_manager is None:
        raise HTTPException(status_code=501, detail="Docker manager not available")

    try:
        result = await _docker_manager.stop_candidate()
        # Release GPU if candidate owned it
        current = await _state_manager.get_gpu_status()
        if current.owner in (GPUOwner.CORE_CANDIDATE, GPUOwner.STUDY_CANDIDATE):
            await _state_manager.release_gpu()
        return {"success": True, "message": "Candidate stack stopped", "details": result}
    except Exception as e:
        logger.exception(f"Error stopping candidate stack: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/containers/candidate/start")
async def start_candidate_stack(request: ContainerStartRequest):
    """Start the candidate stack."""
    if _docker_manager is None:
        raise HTTPException(status_code=501, detail="Docker manager not available")

    try:
        result = await _docker_manager.start_candidate(gpu_enabled=request.gpu_enabled)
        if request.gpu_enabled:
            import uuid
            await _state_manager.set_gpu_owner(
                GPUOwner.CORE_CANDIDATE,
                str(uuid.uuid4()),
                "candidate_stack_start"
            )
        return {"success": True, "message": "Candidate stack started", "details": result}
    except Exception as e:
        logger.exception(f"Error starting candidate stack: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/containers/swap")
async def swap_service(request: ContainerSwapRequest):
    """Swap a service between live and candidate."""
    if _docker_manager is None:
        raise HTTPException(status_code=501, detail="Docker manager not available")

    try:
        result = await _docker_manager.swap_service(request.service, request.target)
        return {"success": True, "message": f"Swapped {request.service} to {request.target}", "details": result}
    except Exception as e:
        logger.exception(f"Error swapping service: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Handoff Protocol Endpoints
# =============================================================================

@app.post("/handoff/prime-to-study")
async def handoff_prime_to_study(request: HandoffRequest = None):
    """Initiate GPU handoff from Core (Prime) to Study."""
    if _handoff_manager is None:
        raise HTTPException(status_code=501, detail="Handoff manager not available")

    if request is None:
        request = HandoffRequest(handoff_type=HandoffType.PRIME_TO_STUDY)

    try:
        handoff = await _handoff_manager.start_prime_to_study(request)
        return handoff
    except Exception as e:
        logger.exception(f"Error starting prime-to-study handoff: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/handoff/study-to-prime")
async def handoff_study_to_prime(request: HandoffRequest = None):
    """Initiate GPU handoff from Study back to Core (Prime)."""
    if _handoff_manager is None:
        raise HTTPException(status_code=501, detail="Handoff manager not available")

    if request is None:
        request = HandoffRequest(handoff_type=HandoffType.STUDY_TO_PRIME)

    try:
        handoff = await _handoff_manager.start_study_to_prime(request)
        return handoff
    except Exception as e:
        logger.exception(f"Error starting study-to-prime handoff: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/handoff/{handoff_id}/status")
async def get_handoff_status(handoff_id: str) -> HandoffStatus:
    """Get status of a handoff operation."""
    if _state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    handoff = await _state_manager.get_handoff_by_id(handoff_id)
    if handoff is None:
        raise HTTPException(status_code=404, detail=f"Handoff {handoff_id} not found")

    return handoff


# =============================================================================
# Sleep Cycle GPU Management
# =============================================================================

@app.post("/gpu/sleep")
async def gpu_sleep():
    """Release GPU for sleep — stop Prime container and free VRAM.

    Called by gaia-core's SleepCycleLoop when entering SLEEPING state.
    Reuses the existing GPUManager.request_release_from_core() which
    stops the prime container and notifies core to demote gpu_prime.
    """
    if _gpu_manager is None:
        raise HTTPException(status_code=501, detail="GPU manager not available")

    try:
        success = await _gpu_manager.request_release_from_core()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to release GPU for sleep")

        if _state_manager:
            await _state_manager.release_gpu()

        logger.info("GPU released for sleep cycle")
        return {"ok": True, "message": "GPU released for sleep"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error releasing GPU for sleep: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gpu/wake")
async def gpu_wake():
    """Reclaim GPU after wake — start Prime container and restore model pool.

    Called by gaia-core's SleepCycleLoop when entering WAKING state.
    Reuses GPUManager.request_reclaim_by_core() which starts the prime
    container, waits for health, and notifies core to restore gpu_prime.
    """
    if _gpu_manager is None:
        raise HTTPException(status_code=501, detail="GPU manager not available")

    try:
        success = await _gpu_manager.request_reclaim_by_core()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to reclaim GPU on wake")

        if _state_manager:
            import uuid
            await _state_manager.set_gpu_owner(
                GPUOwner.CORE, str(uuid.uuid4()), "sleep_wake_reclaim"
            )

        logger.info("GPU reclaimed after wake cycle")
        return {"ok": True, "message": "GPU reclaimed after wake"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error reclaiming GPU on wake: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Candidate Rollback Endpoints
# =============================================================================

# Maps logical service names to container names (must match docker-compose.candidate.yml)
_CANDIDATE_CONTAINERS: dict[str, str] = {
    "core":         "gaia-core-candidate",
    "web":          "gaia-web-candidate",
    "mcp":          "gaia-mcp-candidate",
    "study":        "gaia-study-candidate",
    "orchestrator": "gaia-orchestrator-candidate",
    "audio":        "gaia-audio-candidate",
    "prime":        "gaia-prime-candidate",
}

_REPO_ROOT = Path("/gaia/GAIA_Project")
_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


class RollbackRequest(BaseModel):
    sha: str
    services: List[str]

    @field_validator("sha")
    @classmethod
    def _validate_sha(cls, v: str) -> str:
        if not _SHA_RE.match(v):
            raise ValueError(f"Invalid git SHA: {v!r}")
        return v

    @field_validator("services")
    @classmethod
    def _validate_services(cls, v: List[str]) -> List[str]:
        unknown = set(v) - set(_CANDIDATE_CONTAINERS)
        if unknown:
            raise ValueError(f"Unknown services: {unknown}")
        return v


@app.get("/candidate/snapshot")
async def candidate_snapshot():
    """Return the current git HEAD SHA — the stable-state reference for the resilience pipeline."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"git rev-parse failed: {result.stderr.strip()}")
        sha = result.stdout.strip()
        return {
            "sha": sha,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="git rev-parse timed out")
    except Exception as e:
        logger.exception("candidate_snapshot failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/candidate/rollback")
async def candidate_rollback(request: RollbackRequest):
    """
    Restore all candidates/ files to the given git SHA and restart affected containers.

    Called by CandidateCheckpointManager.restore() when an LLM-generated fix fails
    health checks. This is the guaranteed fallback safety net for the autonomous
    resilience pipeline.
    """
    if _docker_manager is None:
        raise HTTPException(status_code=501, detail="Docker manager not available")

    sha = request.sha
    logger.warning(
        "ROLLBACK requested: sha=%s services=%s", sha[:8], request.services
    )

    # 1. Restore files via git
    try:
        result = subprocess.run(
            ["git", "checkout", sha, "--", "candidates/"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"git checkout failed: {result.stderr.strip()}",
            )
        logger.info("git checkout %s -- candidates/ complete", sha[:8])
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="git checkout timed out")

    # 2. Restart affected containers (skip self — orchestrator restarts last)
    restart_self = "orchestrator" in request.services
    containers_to_restart = [
        _CANDIDATE_CONTAINERS[svc]
        for svc in request.services
        if svc != "orchestrator" and svc in _CANDIDATE_CONTAINERS
    ]

    restart_errors: list[str] = []
    for container_name in containers_to_restart:
        try:
            ok = await _docker_manager.restart_container(container_name)
            if ok:
                logger.info("Restarted %s", container_name)
            else:
                restart_errors.append(f"{container_name}: restart returned False")
        except Exception as exc:
            restart_errors.append(f"{container_name}: {exc}")
            logger.warning("Error restarting %s: %s", container_name, exc)

    response = {
        "ok": True,
        "sha": sha,
        "services_restored": request.services,
        "containers_restarted": containers_to_restart,
        "errors": restart_errors,
    }

    # Restart self last (response will be sent before the restart kills this container)
    if restart_self:
        logger.warning("Scheduling self-restart (gaia-orchestrator-candidate)...")
        asyncio.get_event_loop().call_later(
            1.0,
            lambda: subprocess.Popen(
                ["docker", "restart", "gaia-orchestrator-candidate"]
            ),
        )

    return response


# =============================================================================
# Notification Endpoints
# =============================================================================

@app.post("/notify/oracle-fallback")
async def notify_oracle_fallback(notification: OracleNotification):
    """Receive notification that Oracle fallback is being used."""
    logger.warning(
        f"Oracle fallback: {notification.fallback_model} used for {notification.original_role}"
    )

    if _notification_manager:
        await _notification_manager.broadcast(
            Notification(
                notification_type=NotificationType.ORACLE_FALLBACK,
                title="Cloud Inference Active",
                message=f"Using {notification.fallback_model} for {notification.original_role}",
                data=notification.model_dump(),
            )
        )

    return {"success": True, "message": "Notification received"}


@app.websocket("/ws/notifications")
async def websocket_notifications(websocket: WebSocket):
    """WebSocket endpoint for real-time notifications."""
    await websocket.accept()

    if _notification_manager:
        await _notification_manager.connect(websocket)
        try:
            while True:
                # Keep connection alive, handle any client messages
                await websocket.receive_text()
                # Echo or handle as needed
        except WebSocketDisconnect:
            await _notification_manager.disconnect(websocket)
    else:
        await websocket.close(code=1011, reason="Notification manager not available")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Run the orchestrator service."""
    import uvicorn
    config = get_config()
    uvicorn.run(
        "gaia_orchestrator.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
    )


if __name__ == "__main__":
    main()

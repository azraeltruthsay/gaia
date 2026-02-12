"""
GAIA Orchestrator - FastAPI Application.

Central coordination service for GPU resources and container lifecycle.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

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
    HandoffPhase,
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global _state_manager, _gpu_manager, _docker_manager, _handoff_manager, _notification_manager

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

    logger.info("GAIA Orchestrator ready")
    yield

    # Shutdown
    logger.info("GAIA Orchestrator shutting down...")
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


@app.post("/gpu/acquire")
async def acquire_gpu(request: GPUAcquireRequest) -> GPUAcquireResponse:
    """Request GPU ownership."""
    if _state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

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


@app.post("/gpu/release")
async def release_gpu(lease_id: Optional[str] = None):
    """Release GPU ownership."""
    if _state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

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
            # GPU is free, try to acquire
            return await acquire_gpu(request)

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
                data = await websocket.receive_text()
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

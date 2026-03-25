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
    NanoGPUStatus,
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
_nano_pressure_task: Optional[asyncio.Task] = None
_watch_manager = None
_tier_router = None
_lifecycle_machine = None
_consciousness_matrix = None


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
        global _watch_manager
        from .watch_manager import WatchManager
        _watch_manager = WatchManager(_state_manager, _gpu_manager)
        logger.info("Watch rotation manager initialized")
    except ImportError:
        logger.warning("Watch rotation manager not available yet")

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

    # Initialize lifecycle state machine (must be before tier router)
    try:
        global _lifecycle_machine
        from .lifecycle_machine import LifecycleMachine
        _lifecycle_machine = LifecycleMachine(_state_manager)
        await _lifecycle_machine.load_persisted_state()
        await _lifecycle_machine.reconcile()
        logger.info("Lifecycle state machine initialized: %s", _lifecycle_machine._snapshot.state)
    except Exception:
        logger.warning("Lifecycle state machine initialization failed", exc_info=True)

    # Initialize tier router (auto-handoff, delegates to lifecycle when available)
    try:
        global _tier_router
        from .tier_router import TierRouter
        _tier_router = TierRouter(_state_manager, lifecycle_machine=_lifecycle_machine)
        logger.info("Tier router initialized")
    except ImportError:
        logger.warning("Tier router not available yet")

    # Start Nano VRAM pressure monitor
    global _nano_pressure_task
    if _gpu_manager:
        _nano_pressure_task = asyncio.create_task(_nano_vram_pressure_loop())
        logger.info("Nano VRAM pressure monitor started")

    # Initialize consciousness matrix
    try:
        global _consciousness_matrix
        from .consciousness_matrix import ConsciousnessMatrix
        _consciousness_matrix = ConsciousnessMatrix()
        await _consciousness_matrix.probe_all()
        await _consciousness_matrix.start_continuous_poll(interval=15.0)
        logger.info("Consciousness matrix initialized: %s",
                     {t: s["actual"] for t, s in _consciousness_matrix.get_matrix().items()})
    except Exception:
        logger.warning("Consciousness matrix initialization failed", exc_info=True)

    # Start periodic lifecycle reconciliation (detect model drift)
    _lifecycle_reconcile_task = None
    if _lifecycle_machine:
        async def _lifecycle_reconcile_loop():
            await asyncio.sleep(30)  # Initial delay
            while True:
                try:
                    await _lifecycle_machine.reconcile()
                except Exception:
                    logger.debug("Lifecycle reconcile failed", exc_info=True)
                await asyncio.sleep(60)  # Every 60 seconds

        _lifecycle_reconcile_task = asyncio.create_task(_lifecycle_reconcile_loop())
        logger.info("Lifecycle reconciliation loop started (60s interval)")

    logger.info("GAIA Orchestrator ready")
    yield

    # Shutdown
    logger.info("GAIA Orchestrator shutting down...")
    if _lifecycle_reconcile_task and not _lifecycle_reconcile_task.done():
        _lifecycle_reconcile_task.cancel()
    if _nano_pressure_task and not _nano_pressure_task.done():
        _nano_pressure_task.cancel()
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
            "training": {
                "status": "GET /training/status",
                "validate": "POST /training/validate",
                "kill": "POST /training/kill",
            },
            "nano": {
                "status": "GET /nano/status",
                "backoff": "POST /nano/backoff",
                "restore": "POST /nano/restore",
            },
            "watch": {
                "state": "GET /watch/state",
                "focus": "POST /watch/focus",
                "idle": "POST /watch/idle",
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

    # Build GPU owner string from watch state
    watch = state.watch
    gpu_tiers = [f"{n}({t.vram_mb}MB)" for n, t in watch.tiers.items()
                 if t.device.value.startswith("gpu")]
    gpu_owner = " + ".join(gpu_tiers) if gpu_tiers else "--"

    return {
        "service": "gaia-orchestrator",
        "status": "operational",
        "gpu": state.gpu.model_dump(),
        "gpu_owner": gpu_owner,
        "gpu_state": watch.gpu_state.value,
        "watch": {n: {"device": t.device.value, "vram_mb": t.vram_mb}
                  for n, t in watch.tiers.items()},
        "nano": state.nano.model_dump(),
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
    """Initiate GPU handoff to Study — transition to MEDITATION via lifecycle."""
    if _lifecycle_machine is not None:
        from gaia_common.lifecycle.states import TransitionTrigger, LifecycleState
        result = await _lifecycle_machine.transition(
            TransitionTrigger.TRAINING_SCHEDULED,
            reason="handoff_prime_to_study")
        return {"ok": result.ok, "state": result.to_state,
                "elapsed_s": result.elapsed_s, "error": result.error}

    # Legacy fallback
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
    """Return GPU from Study — transition from MEDITATION to AWAKE via lifecycle."""
    if _lifecycle_machine is not None:
        from gaia_common.lifecycle.states import TransitionTrigger
        result = await _lifecycle_machine.transition(
            TransitionTrigger.TRAINING_COMPLETE,
            reason="handoff_study_to_prime")
        return {"ok": result.ok, "state": result.to_state,
                "elapsed_s": result.elapsed_s, "error": result.error}

    # Legacy fallback
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
    """Release GPU for sleep — transition to SLEEP via lifecycle machine.

    Called by gaia-core's SleepCycleLoop when entering SLEEPING state.
    Delegates to lifecycle machine which handles tier unloading.
    """
    if _lifecycle_machine is not None:
        from gaia_common.lifecycle.states import TransitionTrigger, LifecycleState
        result = await _lifecycle_machine.transition(
            TransitionTrigger.IDLE_TIMEOUT, reason="sleep_cycle")
        if result.ok:
            return {"ok": True, "message": "GPU released for sleep via lifecycle", "state": result.to_state}
        # If lifecycle says invalid transition (e.g. already sleeping), that's ok
        return {"ok": True, "message": f"Lifecycle: {result.error}", "state": result.from_state}

    # Fallback to legacy GPU manager
    if _gpu_manager is None:
        raise HTTPException(status_code=501, detail="GPU manager not available")
    try:
        success = await _gpu_manager.request_release_from_core()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to release GPU for sleep")
        if _state_manager:
            await _state_manager.release_gpu()
        logger.info("GPU released for sleep cycle (legacy)")
        return {"ok": True, "message": "GPU released for sleep"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error releasing GPU for sleep: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gpu/wake")
async def gpu_wake():
    """Reclaim GPU after wake — transition to AWAKE via lifecycle machine.

    Called by gaia-core's SleepCycleLoop when entering WAKING state.
    Delegates to lifecycle machine which handles tier loading + KV prewarm.
    """
    if _lifecycle_machine is not None:
        from gaia_common.lifecycle.states import TransitionTrigger
        result = await _lifecycle_machine.transition(
            TransitionTrigger.WAKE_SIGNAL, reason="gpu_wake")
        if result.ok:
            return {"ok": True, "message": "GPU reclaimed via lifecycle", "state": result.to_state}
        return {"ok": False, "message": f"Lifecycle: {result.error}", "state": result.from_state}

    # Fallback to legacy GPU manager
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
        logger.info("GPU reclaimed after wake cycle (legacy)")
        return {"ok": True, "message": "GPU reclaimed after wake"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error reclaiming GPU on wake: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Single Container Restart (used by self-awareness training pipeline)
# =============================================================================

_RESTART_ALLOWED = {
    "gaia-prime", "gaia-nano", "gaia-core", "gaia-web",
    "gaia-study", "gaia-mcp",
}


@app.post("/containers/{container_name}/restart")
async def restart_single_container(container_name: str):
    """Restart a single container by name.

    Used by the self-awareness pipeline to restart gaia-prime and gaia-nano
    after deploying new model weights. Whitelist prevents arbitrary restarts.
    """
    if _docker_manager is None:
        raise HTTPException(status_code=501, detail="Docker manager not available")

    if container_name not in _RESTART_ALLOWED:
        raise HTTPException(
            status_code=403,
            detail=f"Container '{container_name}' not in restart whitelist",
        )

    try:
        ok = await _docker_manager.restart_container(container_name)
        if ok:
            logger.info("Restarted container %s", container_name)
            return {"ok": True, "container": container_name}
        else:
            raise HTTPException(status_code=500, detail=f"Restart of {container_name} returned False")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error restarting container %s", container_name)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Warm Pool Sync (copy merged model to warm pool for vLLM)
# =============================================================================

class WarmPoolSyncRequest(BaseModel):
    model: str  # e.g. "Qwen3.5-4B-Abliterated-merged"


@app.post("/warm-pool/sync")
async def warm_pool_sync(request: WarmPoolSyncRequest):
    """Copy a model from gaia-models/ to /warm_pool/.

    Used after merge+requantize to make the new merged model available
    to gaia-prime (which mounts /mnt/gaia_warm_pool as /models).
    """
    import shutil

    src = Path(f"/gaia/GAIA_Project/gaia-models/{request.model}")
    dst = Path(f"/warm_pool/{request.model}")

    if not src.exists():
        raise HTTPException(status_code=404, detail=f"Source model not found: {src}")

    if not Path("/warm_pool").exists():
        raise HTTPException(
            status_code=501,
            detail="Warm pool not mounted — add /mnt/gaia_warm_pool:/warm_pool:rw to orchestrator volumes",
        )

    try:
        logger.info("Syncing model %s → %s", src, dst)
        shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
        # Calculate synced size
        total_bytes = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
        size_gb = total_bytes / (1024 ** 3)
        logger.info("Warm pool sync complete: %s (%.2f GB)", request.model, size_gb)
        return {"ok": True, "model": request.model, "size_gb": round(size_gb, 2)}
    except Exception as e:
        logger.exception("Warm pool sync failed for %s", request.model)
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
# Training Subprocess Monitoring
# =============================================================================

@app.get("/training/status")
async def get_training_status():
    """Get training subprocess status from gaia-study."""
    if _gpu_manager is None:
        raise HTTPException(status_code=501, detail="GPU manager not available")

    return await _gpu_manager.get_training_status()


@app.post("/training/validate")
async def validate_training():
    """Validate a completed training run (adapter files, loss, state)."""
    if _gpu_manager is None:
        raise HTTPException(status_code=501, detail="GPU manager not available")

    return await _gpu_manager.validate_training_result()


@app.post("/training/kill")
async def kill_training():
    """Force-kill the training subprocess (last resort)."""
    if _gpu_manager is None:
        raise HTTPException(status_code=501, detail="GPU manager not available")

    return await _gpu_manager.kill_training_subprocess()


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
# Nano GPU Backoff
# =============================================================================

async def _nano_vram_pressure_loop():
    """Background loop that monitors VRAM and auto-manages GPU resources.

    Handles:
    1. Nano GPU placement (evict to CPU under pressure, restore when free)
    2. Audio idle detection (sleep audio GPU models after idle timeout)
    """
    config = get_config()
    interval = config.nano_vram_check_interval
    audio_idle_timeout = 300  # 5 minutes of no audio → release GPU
    _audio_last_checked = 0
    _audio_sleeping = False

    # Wait for services to stabilize on startup
    await asyncio.sleep(30)

    while True:
        try:
            if _gpu_manager:
                # 1. Nano VRAM pressure management
                action = await _gpu_manager.check_nano_vram_pressure()
                if action and _notification_manager:
                    await _notification_manager.broadcast(
                        Notification(
                            notification_type=NotificationType.GPU_RELEASED
                            if action == "evicted"
                            else NotificationType.GPU_ACQUIRED,
                            title=f"Nano {action}",
                            message=f"Nano {'moved to CPU' if action == 'evicted' else 'restored to GPU'} "
                                    f"(VRAM pressure {'detected' if action == 'evicted' else 'relieved'})",
                            data={"nano_action": action},
                        )
                    )

                # 2. Audio idle detection — release GPU if not processing
                import time as _time
                if _time.time() - _audio_last_checked > 60:  # Check every 60s
                    _audio_last_checked = _time.time()
                    try:
                        import httpx
                        async with httpx.AsyncClient(timeout=5) as client:
                            resp = await client.get(f"{config.audio_url}/gpu/status")
                            if resp.status_code == 200:
                                audio_status = resp.json()
                                vram_used = audio_status.get("vram_used_mb", 0)
                                is_sleeping = audio_status.get("gpu_mode") == "sleeping"

                                if vram_used > 100 and not is_sleeping:
                                    # Audio has GPU models loaded — check if actively used
                                    # If muted or no recent activity, release
                                    muted = audio_status.get("muted", True)
                                    if muted and not _audio_sleeping:
                                        logger.info("Audio GPU idle (muted, %dMB VRAM) — releasing", vram_used)
                                        await _gpu_manager.sleep_audio()
                                        _audio_sleeping = True
                                elif is_sleeping:
                                    _audio_sleeping = True
                                else:
                                    _audio_sleeping = False
                    except Exception:
                        pass  # Audio may not be running

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning("VRAM management check failed: %s", e)

        await asyncio.sleep(interval)


@app.get("/nano/status")
async def get_nano_status() -> NanoGPUStatus:
    """Get Nano GPU/CPU placement status."""
    if _state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")
    return _state_manager.state.nano


@app.post("/nano/backoff")
async def nano_backoff(reason: str = "manual"):
    """Evict Nano from GPU to CPU to free VRAM."""
    if _gpu_manager is None:
        raise HTTPException(status_code=501, detail="GPU manager not available")

    success = await _gpu_manager.evict_nano_to_cpu(reason)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to evict Nano to CPU")
    return {"ok": True, "mode": "cpu", "reason": reason}


@app.post("/nano/restore")
async def nano_restore(reason: str = "manual"):
    """Restore Nano to GPU for faster inference."""
    if _gpu_manager is None:
        raise HTTPException(status_code=501, detail="GPU manager not available")

    success = await _gpu_manager.restore_nano_to_gpu(reason)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to restore Nano to GPU (VRAM may be insufficient)")
    return {"ok": True, "mode": "gpu", "reason": reason}


# =============================================================================
# GPU Watch Rotation
# =============================================================================

@app.get("/watch/state")
async def get_watch_state():
    """Get current GPU watch rotation state."""
    if _watch_manager is None:
        raise HTTPException(status_code=501, detail="Watch manager not available")
    return await _watch_manager.get_state()


@app.post("/watch/focus")
async def watch_focus(reason: str = "user_request", priority: str = "NORMAL"):
    """Transition to FOCUSING — load Prime, unload Core+Nano via lifecycle.

    Priority: URGENT (immediate yield), NORMAL (finish current batch),
    SCHEDULED (pre-emptive coordination).
    """
    if _lifecycle_machine is not None:
        from gaia_common.lifecycle.states import TransitionTrigger, LifecycleState
        result = await _lifecycle_machine.transition(
            TransitionTrigger.USER_REQUEST, target=LifecycleState.FOCUSING,
            reason=f"watch_focus: {reason}")
        return {"ok": result.ok, "state": result.to_state,
                "elapsed_s": result.elapsed_s, "error": result.error}

    if _watch_manager is None:
        raise HTTPException(status_code=501, detail="Watch manager not available")
    result = await _watch_manager.focus(reason=reason, priority=priority)
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error", "focus failed"))
    return result


@app.post("/watch/idle")
async def watch_idle(reason: str = "inactivity"):
    """Transition to AWAKE — unload Prime, load Core+Nano on GPU via lifecycle."""
    if _lifecycle_machine is not None:
        from gaia_common.lifecycle.states import TransitionTrigger, LifecycleState
        result = await _lifecycle_machine.transition(
            TransitionTrigger.USER_REQUEST, target=LifecycleState.AWAKE,
            reason=f"watch_idle: {reason}")
        return {"ok": result.ok, "state": result.to_state,
                "elapsed_s": result.elapsed_s, "error": result.error}

    if _watch_manager is None:
        raise HTTPException(status_code=501, detail="Watch manager not available")
    result = await _watch_manager.idle(reason=reason)
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error", "idle failed"))
    return result


# =============================================================================
# Tier Router — Automatic GPU Handoff
# =============================================================================

class TierInferRequest(BaseModel):
    """Request to infer on a specific tier with automatic GPU handoff."""
    tier: str
    messages: list
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "cuda"


class TierEnsureRequest(BaseModel):
    """Request to ensure a tier's model is loaded."""
    tier: str
    device: str = "cuda"


@app.post("/tier/infer")
async def tier_infer(req: TierInferRequest):
    """Infer on a specific tier with automatic GPU handoff.

    Just specify which tier you want to talk to — the router handles
    loading/unloading models transparently.
    """
    if _tier_router is None:
        raise HTTPException(status_code=501, detail="Tier router not available")
    result = await _tier_router.infer(
        tier=req.tier, messages=req.messages, max_tokens=req.max_tokens,
        temperature=req.temperature, top_p=req.top_p, device=req.device)
    if "error" in result and "_handoff" not in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@app.post("/tier/ensure")
async def tier_ensure(req: TierEnsureRequest):
    """Ensure a tier's model is loaded on GPU.

    If another tier is loaded, it will be unloaded first.
    """
    if _tier_router is None:
        raise HTTPException(status_code=501, detail="Tier router not available")
    result = await _tier_router.ensure_tier(tier=req.tier, device=req.device)
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error", "ensure failed"))
    return result


@app.get("/tier/status")
async def tier_status():
    """Check which tiers currently have models loaded."""
    if _tier_router is None:
        raise HTTPException(status_code=501, detail="Tier router not available")
    return await _tier_router.get_loaded_tiers()


@app.post("/tier/unload-all")
async def tier_unload_all():
    """Unload all tiers — zero GPU memory."""
    if _tier_router is None:
        raise HTTPException(status_code=501, detail="Tier router not available")
    return await _tier_router.unload_all()


@app.post("/tier/sae-record")
async def tier_sae_record(tier: str, tag: str = "handoff_test"):
    """Trigger SAE atlas recording on a loaded tier."""
    if _tier_router is None:
        raise HTTPException(status_code=501, detail="Tier router not available")
    return await _tier_router.sae_record(tier=tier, tag=tag)


# =============================================================================
# Lifecycle State Machine — Unified GPU Lifecycle
# =============================================================================

class LifecycleTransitionRequest(BaseModel):
    """Request a lifecycle state transition."""
    trigger: str
    target: Optional[str] = None
    reason: str = ""


@app.get("/lifecycle/state")
async def lifecycle_state():
    """Get current lifecycle snapshot — the single source of truth."""
    if _lifecycle_machine is None:
        raise HTTPException(status_code=501, detail="Lifecycle machine not available")
    snapshot = await _lifecycle_machine.get_snapshot()
    return snapshot.model_dump()


@app.post("/lifecycle/transition")
async def lifecycle_transition(req: LifecycleTransitionRequest):
    """Request a lifecycle state transition."""
    if _lifecycle_machine is None:
        raise HTTPException(status_code=501, detail="Lifecycle machine not available")

    from gaia_common.lifecycle.states import TransitionTrigger, LifecycleState

    try:
        trigger = TransitionTrigger(req.trigger)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid trigger: {req.trigger}")

    target = None
    if req.target:
        try:
            target = LifecycleState(req.target)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid target: {req.target}")

    result = await _lifecycle_machine.transition(trigger, target, req.reason)
    status = 200 if result.ok else 409
    return JSONResponse(status_code=status, content=result.model_dump())


@app.get("/lifecycle/transitions")
async def lifecycle_transitions():
    """Get available transitions from current state."""
    if _lifecycle_machine is None:
        raise HTTPException(status_code=501, detail="Lifecycle machine not available")
    return await _lifecycle_machine.get_available_transitions()


@app.get("/lifecycle/history")
async def lifecycle_history(limit: int = 50):
    """Get recent transition history."""
    if _lifecycle_machine is None:
        raise HTTPException(status_code=501, detail="Lifecycle machine not available")
    records = await _lifecycle_machine.get_history(limit)
    return [r.model_dump() for r in records]


@app.post("/lifecycle/reconcile")
async def lifecycle_reconcile():
    """Force reconciliation — probe all tiers and infer actual state."""
    if _lifecycle_machine is None:
        raise HTTPException(status_code=501, detail="Lifecycle machine not available")
    return await _lifecycle_machine.reconcile()


@app.get("/lifecycle/tiers")
async def lifecycle_tiers():
    """Get live per-tier status (probes all engines)."""
    if _lifecycle_machine is None:
        raise HTTPException(status_code=501, detail="Lifecycle machine not available")
    snapshot = await _lifecycle_machine.get_snapshot()
    return {k: v.model_dump() for k, v in snapshot.tiers.items()}


# =============================================================================
# Consciousness Matrix Endpoints
# =============================================================================

@app.get("/consciousness/matrix")
async def consciousness_matrix():
    """Get the full consciousness matrix — target vs actual for all tiers."""
    if _consciousness_matrix is None:
        raise HTTPException(status_code=501, detail="Consciousness matrix not initialized")
    return _consciousness_matrix.get_matrix()


@app.post("/consciousness/probe")
async def consciousness_probe():
    """Force-probe all tiers and return updated matrix."""
    if _consciousness_matrix is None:
        raise HTTPException(status_code=501, detail="Consciousness matrix not initialized")
    return await _consciousness_matrix.probe_all()


class ConsciousnessRequest(BaseModel):
    tier: str
    level: int  # 1=unconscious, 2=subconscious, 3=conscious

    @field_validator("level")
    @classmethod
    def validate_level(cls, v):
        if v not in (1, 2, 3):
            raise ValueError("level must be 1 (unconscious), 2 (subconscious), or 3 (conscious)")
        return v


@app.post("/consciousness/set")
async def consciousness_set(request: ConsciousnessRequest):
    """Set a tier's target consciousness level."""
    if _consciousness_matrix is None:
        raise HTTPException(status_code=501, detail="Consciousness matrix not initialized")
    from .consciousness_matrix import ConsciousnessLevel
    level = ConsciousnessLevel(request.level)
    return await _consciousness_matrix.set_target(request.tier, level)


@app.post("/consciousness/awake")
async def consciousness_awake():
    """Set AWAKE configuration: Core=3, Nano=3, Prime=2."""
    if _consciousness_matrix is None:
        raise HTTPException(status_code=501, detail="Consciousness matrix not initialized")
    return await _consciousness_matrix.awake()


@app.post("/consciousness/focusing")
async def consciousness_focusing():
    """Set FOCUSING configuration: Nano=3, Core=2, Prime=3."""
    if _consciousness_matrix is None:
        raise HTTPException(status_code=501, detail="Consciousness matrix not initialized")
    return await _consciousness_matrix.focusing()


@app.post("/consciousness/sleep")
async def consciousness_sleep():
    """Set SLEEP configuration: Nano=2, Core=2, Prime=1."""
    if _consciousness_matrix is None:
        raise HTTPException(status_code=501, detail="Consciousness matrix not initialized")
    return await _consciousness_matrix.sleep()


@app.post("/consciousness/deep-sleep")
async def consciousness_deep_sleep():
    """Set DEEP SLEEP configuration: All → 1 (Nano stays 2)."""
    if _consciousness_matrix is None:
        raise HTTPException(status_code=501, detail="Consciousness matrix not initialized")
    return await _consciousness_matrix.deep_sleep()


@app.post("/consciousness/training")
async def consciousness_training(tier: str = "prime"):
    """Set TRAINING configuration: target tier → 1, others → 2."""
    if _consciousness_matrix is None:
        raise HTTPException(status_code=501, detail="Consciousness matrix not initialized")
    return await _consciousness_matrix.training(tier)


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

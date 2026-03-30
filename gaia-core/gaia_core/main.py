"""
gaia-core FastAPI application entry point.

Provides the HTTP API for the cognitive loop service.
This is The Brain - Cognitive loop and reasoning.
"""

import asyncio
import os
import logging
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import json

# Persistent file logging — writes to /logs/gaia-core.log (mounted volume)
try:
    from gaia_common.utils import setup_logging, install_health_check_filter
    _log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    setup_logging(log_dir="/logs", level=_log_level, service_name="gaia-core")
    install_health_check_filter()
except ImportError:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("GAIA.Core.API")

try:
    from gaia_common.utils.error_logging import log_gaia_error
except ImportError:
    def log_gaia_error(lgr, code, detail="", **kw):
        lgr.error("[%s] %s", code, detail)

# Global references for the cognitive system
_agent_core = None
_ai_manager = None
# Serialise cognitive turns — only one run_turn at a time to prevent
# model contention (GPU/CPU) and keep response times predictable.
_turn_semaphore = asyncio.Semaphore(1)


class AIManagerShim:
    """
    A lightweight shim providing the interface AgentCore expects from ai_manager.

    AgentCore requires:
    - model_pool: The ModelPool instance
    - config: The Config instance
    - session_manager: The SessionManager instance
    - active_persona: The currently active persona object
    - status: A dict for tracking state
    - initialize(persona_name): Method to load/switch personas
    """

    def __init__(self, config, model_pool, session_manager):
        self.config = config
        self.model_pool = model_pool
        self.session_manager = session_manager
        self.active_persona = None
        self.status = {
            "initialized": False,
            "last_response": None,
            "current_persona": "prime",
        }
        self._persona_manager = model_pool.persona_manager

    def initialize(self, persona_name: str = "prime"):
        """Load and activate a persona by name."""
        try:
            persona_data = self._persona_manager.get_persona(persona_name)
            if persona_data:
                # PersonaManager returns a dict, wrap it in MinimalPersona
                self.active_persona = MinimalPersona(persona_name, persona_data)
                self.status["current_persona"] = persona_name
                self.status["initialized"] = True
                logger.info(f"AIManagerShim: Initialized with persona '{persona_name}'")
            else:
                # Fallback to a minimal persona object
                logger.warning(f"AIManagerShim: Persona '{persona_name}' not found, using minimal fallback")
                self.active_persona = MinimalPersona(persona_name)
                self.status["current_persona"] = persona_name
                self.status["initialized"] = True
        except Exception as e:
            logger.error(f"AIManagerShim: Failed to initialize persona '{persona_name}': {e}")
            self.active_persona = MinimalPersona(persona_name)
            self.status["initialized"] = True


class MinimalPersona:
    """Minimal persona object when full persona loading fails or from dict data."""
    def __init__(self, name: str, data: Dict[str, Any] = None):
        self.name = name
        if data and isinstance(data, dict):
            # Extract traits from persona data dict
            self.traits = data.get("traits", {})
            if not self.traits:
                # Try alternate structures
                self.traits = {
                    "tone": data.get("tone", "helpful and articulate"),
                    "style": data.get("style", "conversational"),
                }
            # Copy other common persona attributes
            self.identity = data.get("identity", name)
            self.description = data.get("description", "")
            self.system_prompt = data.get("system_prompt", "")
        else:
            self.traits = {
                "tone": "helpful and articulate",
                "style": "conversational",
            }
            self.identity = name
            self.description = ""
            self.system_prompt = ""


def initialize_cognitive_system():
    """
    Initialize the cognitive system components.
    Called during FastAPI startup.
    """
    global _agent_core, _ai_manager

    logger.info("Initializing GAIA cognitive system...")

    try:
        # Import components
        from gaia_core.config import get_config
        from gaia_core.models.model_pool import get_model_pool
        from gaia_core.memory.session_manager import SessionManager
        from gaia_core.cognition.agent_core import AgentCore

        # Get config and model pool
        config = get_config()
        model_pool = get_model_pool()

        # Check if models should be auto-loaded
        autoload = os.getenv("GAIA_AUTOLOAD_MODELS", "0") == "1"
        allow_prime = os.getenv("GAIA_ALLOW_PRIME_LOAD", "0") == "1"

        if autoload:
            logger.info("GAIA_AUTOLOAD_MODELS=1: Loading models on startup...")
            if allow_prime:
                model_pool.enable_prime_load()
            model_pool.load_models()
        else:
            logger.info("GAIA_AUTOLOAD_MODELS=0: Models will load on first use (lazy loading)")
            # Ensure prime load is allowed for lazy loading
            if allow_prime:
                model_pool.enable_prime_load()

        # Initialize session manager
        session_manager = SessionManager(config)

        # Create the AIManager shim
        _ai_manager = AIManagerShim(config, model_pool, session_manager)
        _ai_manager.initialize("prime")

        # Create the AgentCore
        _agent_core = AgentCore(_ai_manager)

        # ── Auto-register default LoRA adapter with vLLM if it exists on disk ──
        try:
            from pathlib import Path
            adapter_name = "gaia_persona_v1"
            adapter_path = Path("/models/lora_adapters/tier1_global") / adapter_name
            if (adapter_path / "adapter_config.json").exists():
                registered = model_pool.register_adapter_with_prime(
                    adapter_name, str(adapter_path),
                )
                if registered:
                    logger.info("Default persona adapter '%s' registered with vLLM", adapter_name)
                else:
                    logger.warning("Could not register default persona adapter '%s'", adapter_name)
            else:
                logger.info("No persona adapter at %s — running without LoRA", adapter_path)
        except Exception:
            logger.debug("Adapter auto-registration skipped", exc_info=True)

        # ── Start Heartbeat Time Check ──
        try:
            from gaia_common.utils.heartbeat_time_check import start_heartbeat
            heartbeat_interval = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "300"))
            start_heartbeat(interval_seconds=heartbeat_interval)
            logger.info("Heartbeat time check started (interval=%ds)", heartbeat_interval)
        except Exception:
            logger.debug("Heartbeat time check not started", exc_info=True)

        logger.info("GAIA cognitive system initialized successfully")
        return True

    except Exception as e:
        log_gaia_error(logger, "GAIA-CORE-003", str(e), exc_info=True)
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    success = initialize_cognitive_system()
    if not success:
        log_gaia_error(logger, "GAIA-CORE-003", "Endpoints will return errors")

    # Fetch critical instances for app.state
    from gaia_core.config import get_config
    from gaia_core.models.model_pool import get_model_pool
    
    config = get_config()
    pool = get_model_pool()

    # Store critical instances on app.state for endpoint access
    app.state.config = config
    app.state.model_pool = pool

    # Start sleep cycle loop
    _sleep_loop = None
    try:
        from gaia_core.cognition.sleep_cycle_loop import SleepCycleLoop
        sleep_enabled = getattr(config, "SLEEP_ENABLED", True)
        if sleep_enabled:
            _sleep_loop = SleepCycleLoop(
                config,
                model_pool=_ai_manager.model_pool if _ai_manager else None,
                agent_core=_agent_core,
                session_manager=_ai_manager.session_manager if _ai_manager else None,
            )
            # Store sleep-specific instances on app.state
            app.state.sleep_wake_manager = _sleep_loop.sleep_wake_manager
            app.state.idle_monitor = _sleep_loop.idle_monitor
            app.state.sleep_cycle_loop = _sleep_loop
            app.state.timeline_store = _sleep_loop.timeline_store
            app.state.heartbeat = _sleep_loop.heartbeat
            app.state.temporal_state_manager = (
                _sleep_loop.heartbeat._temporal_state_manager
                if _sleep_loop.heartbeat else None
            )

            # Lifecycle client — queries the orchestrator's unified state machine
            try:
                from gaia_common.lifecycle.client import LifecycleClient
                _orchestrator_url = os.environ.get("ORCHESTRATOR_ENDPOINT", "http://gaia-orchestrator:6410")
                app.state.lifecycle_client = LifecycleClient(_orchestrator_url)
                logger.info("Lifecycle client initialized (orchestrator: %s)", _orchestrator_url)
            except ImportError:
                app.state.lifecycle_client = None
                logger.debug("Lifecycle client not available")

            # Wire timeline store and lifecycle client to agent_core
            if _agent_core is not None:
                _agent_core.timeline_store = _sleep_loop.timeline_store
                _agent_core._lifecycle_client = app.state.lifecycle_client

            _sleep_loop.start()
            logger.info("Sleep cycle loop started")
        else:
            logger.info("Sleep cycle disabled (SLEEP_ENABLED=False)")
    except Exception:
        logger.warning("Failed to start sleep cycle loop", exc_info=True)

    # Start idle heartbeat — the "sound of silence"
    _idle_heartbeat = None
    try:
        from gaia_core.cognition.idle_heartbeat import IdleHeartbeat
        _idle_heartbeat = IdleHeartbeat(
            config=config,
            model_pool=pool,
            timeline_store=getattr(app.state, "timeline_store", None),
            session_manager=getattr(_agent_core, "session_manager", None) if _agent_core else None,
        )
        _idle_heartbeat.start()
        app.state.idle_heartbeat = _idle_heartbeat
        logger.info("Idle heartbeat started")
    except Exception:
        logger.warning("Failed to start idle heartbeat", exc_info=True)

    # Start KV cache manager (background restore + periodic checkpoint)
    _kv_cache_mgr = None
    try:
        from gaia_core.cognition.kv_cache_manager import init_kv_cache_manager
        _kv_cache_mgr = init_kv_cache_manager(pool)
        app.state.kv_cache_manager = _kv_cache_mgr

        # Restore caches in background thread to avoid blocking startup
        import threading
        threading.Thread(
            target=_kv_cache_mgr.restore_all,
            name="kv-cache-restore",
            daemon=True,
        ).start()

        # Start periodic checkpoint thread
        _kv_cache_mgr.start()
        logger.info("KV cache manager started")
    except Exception:
        logger.warning("Failed to start KV cache manager", exc_info=True)

    # Start audio commentary daemon
    _audio_commentary = None
    try:
        from gaia_core.cognition.audio_commentary import AudioCommentaryEvaluator
        from gaia_core.config import get_config as _get_config_ac

        _ac_config = _get_config_ac()
        _audio_commentary = AudioCommentaryEvaluator(
            model_pool=_ai_manager.model_pool if _ai_manager else None,
            agent_core=_agent_core,
            sleep_wake_manager=getattr(app.state, "sleep_wake_manager", None),
            config=_ac_config,
        )
        _audio_commentary.start()
        app.state.audio_commentary = _audio_commentary
    except Exception:
        logger.warning("Failed to start audio commentary daemon", exc_info=True)

    yield

    # Shutdown — write cognitive checkpoints before stopping
    logger.info("GAIA Core shutting down — writing cognitive checkpoints...")
    try:
        _write_shutdown_checkpoints(app)
    except Exception:
        logger.warning("Shutdown checkpoint write failed", exc_info=True)

    # Save KV caches before shutdown
    if _kv_cache_mgr is not None:
        try:
            _kv_cache_mgr.stop()
            _kv_cache_mgr.save_all()
            logger.info("KV caches saved on shutdown")
        except Exception:
            logger.warning("KV cache shutdown save failed", exc_info=True)

    if _audio_commentary is not None:
        _audio_commentary.stop()
        logger.info("Audio commentary daemon stopped")

    if _sleep_loop is not None:
        _sleep_loop.initiate_shutdown()
        logger.info("Sleep cycle loop stopped (OFFLINE)")
    logger.info("GAIA Core shutdown complete.")


def _write_shutdown_checkpoints(app: FastAPI) -> dict:
    """Write prime.md and Lite.md checkpoints (called on shutdown and via endpoint)."""
    results = {}

    if _ai_manager is None:
        logger.warning("Cannot write checkpoints — cognitive system not initialized")
        return {"error": "cognitive system not initialized"}

    config = _ai_manager.config
    model_pool = _ai_manager.model_pool
    timeline_store = getattr(app.state, "timeline_store", None)
    sleep_wake_manager = getattr(app.state, "sleep_wake_manager", None)

    # Write prime.md
    try:
        from gaia_core.cognition.prime_checkpoint import PrimeCheckpointManager

        pm = PrimeCheckpointManager(config, timeline_store=timeline_store)
        pm.rotate_checkpoints()
        path = pm.create_checkpoint(packet=None, model_pool=model_pool)
        results["prime"] = {"status": "ok", "path": str(path)}
        logger.info("Shutdown checkpoint: prime.md written")
    except Exception as exc:
        results["prime"] = {"status": "error", "detail": str(exc)}
        logger.error("Shutdown checkpoint: prime.md failed: %s", exc, exc_info=True)

    # Write Lite.md
    try:
        from gaia_core.cognition.lite_journal import LiteJournal

        lj = LiteJournal(
            config,
            model_pool=model_pool,
            timeline_store=timeline_store,
            sleep_wake_manager=sleep_wake_manager,
        )
        entry = lj.write_entry()
        if entry:
            results["lite"] = {"status": "ok", "chars": len(entry)}
            logger.info("Shutdown checkpoint: Lite.md entry written")
        else:
            results["lite"] = {"status": "skipped", "reason": "no Lite model available"}
            logger.info("Shutdown checkpoint: Lite.md skipped (no model)")
    except Exception as exc:
        results["lite"] = {"status": "error", "detail": str(exc)}
        logger.error("Shutdown checkpoint: Lite.md failed: %s", exc, exc_info=True)

    return results


app = FastAPI(
    title="GAIA Core",
    description="The Brain - Cognitive loop and reasoning engine",
    version="0.1.0",
    lifespan=lifespan,
)

# Register GPU management endpoints (used by orchestrator for sleep/wake handoff)
from gaia_core.api.gpu_endpoints import router as gpu_router
app.include_router(gpu_router)

# Register sleep cycle endpoints
from gaia_core.api.sleep_endpoints import router as sleep_router
app.include_router(sleep_router)

# Register model/adapter management endpoints (adapter notifications from gaia-study)
from gaia_core.api.model_endpoints import router as model_router
app.include_router(model_router)


@app.post("/api/repair/structural")
async def repair_structural_error(request: Request):
    """
    Cognitive repair endpoint for structural code errors.
    Expects JSON: { "service": "...", "broken_code": "...", "error_msg": "...", "file_path": "..." }
    If file_path is provided, gaia-core validates and writes the fix directly (avoids ro-mount
    issues in the caller). Returns { "status": "repaired", "file_path": "..." } on success.
    If file_path is omitted, returns { "fixed_code": "..." } for the caller to handle.
    """
    import ast as _ast
    from pathlib import Path as _Path

    data = await request.json()
    service = data.get("service", "unknown")
    broken_code = data.get("broken_code")
    error_msg = data.get("error_msg")
    file_path = data.get("file_path")  # optional: let gaia-core write the fix

    if not broken_code or not error_msg:
        return JSONResponse(status_code=400, content={"error": "Missing broken_code or error_msg"})

    try:
        from gaia_core.cognition.structural_surgeon import StructuralSurgeon
        surgeon = StructuralSurgeon(request.app.state.config, request.app.state.model_pool)
        fixed_code = surgeon.repair_structural_failure(service, broken_code, error_msg)

        if not fixed_code:
            return JSONResponse(status_code=500, content={"error": "HA surgery failed to generate fix"})

        if file_path:
            # Validate then write — gaia-core has rw access to the project root
            try:
                _ast.parse(fixed_code)
            except SyntaxError as e:
                return JSONResponse(status_code=422, content={"error": f"Fix failed validation: {e}"})
            target = _Path(file_path)
            target.write_text(fixed_code)
            logger.info("Structural repair: wrote fix to %s", file_path)
            return {"status": "repaired", "file_path": file_path}

        return {"fixed_code": fixed_code}
    except Exception as e:
        logger.exception("Structural repair API failed")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/doctor/diagnose")
async def doctor_diagnose(request: Request):
    """
    Doctor-initiated diagnostic turn.
    Triggered when a reload loop or resource spike is detected.
    """
    data = await request.json()
    service = data.get("service")
    logs = data.get("logs")
    
    if not service or not logs:
        return JSONResponse(status_code=400, content={"error": "Missing service or logs"})
        
    log_gaia_error(logger, "GAIA-CORE-002", f"Doctor-initiated diagnosis: {service} is in a restart loop")
    
    # Create a specialized internal packet for self-healing
    
    diagnostics_prompt = (
        f"URGENT SYSTEM REPAIR: The service '{service}' is caught in a recursive restart loop. "
        f"Analyze the following logs and determine the root cause. If it is a recursive file write "
        f"(feedback loop), propose a fix to exclude the path or change the logic. "
        f"RECENT LOGS:\n{logs}"
    )
    
    # This turn will be processed with high-reasoning priority
    import time
    return StreamingResponse(
        _agent_core.run_turn(
            user_input=diagnostics_prompt,
            session_id=f"diagnostics_{service}_{int(time.time())}",
            source="gaia-doctor"
        ),
        media_type="application/x-ndjson"
    )

@app.post("/api/doctor/review")
async def doctor_review(request: Request):
    """
    Cognitive veto point for sovereign promotion.
    Doctor submits candidate→production diffs; GAIA (Prime) reviews and approves/denies.
    Expects JSON: { "diffs": [...], "source": "doctor_sovereign_promote", "file_count": N }
    Returns: { "approved": true/false, "reason": "..." }
    """
    data = await request.json()
    diffs = data.get("diffs", [])
    source = data.get("source", "unknown")

    if not diffs:
        return JSONResponse(status_code=400, content={"error": "No diffs provided"})

    logger.info("🔱 SOVEREIGN REVIEW: %d files from %s", len(diffs), source)

    # Build a review prompt for GAIA Prime
    diff_summary = []
    for d in diffs[:10]:  # Cap at 10 files to stay within context
        vital_tag = " [VITAL ORGAN]" if d.get("vital") else ""
        diff_summary.append(f"### {d['file']}{vital_tag}\n```diff\n{d['diff'][:4000]}\n```")

    review_prompt = (
        "SOVEREIGN PROMOTION REVIEW\n\n"
        "The Doctor has detected that candidate files differ from production. "
        "Review the following diffs and determine if they should be promoted.\n\n"
        "Approve ONLY if:\n"
        "- Changes are syntactically valid\n"
        "- No obvious regressions or security issues\n"
        "- Changes appear intentional (not corruption)\n\n"
        "Respond with EXACTLY one line:\n"
        "APPROVED: <brief reason>\n"
        "or\n"
        "DENIED: <brief reason>\n\n"
        + "\n\n".join(diff_summary)
    )

    try:
        # Use the agent core for cognitive review
        response_text = ""
        for chunk in _agent_core.run_turn(
            user_input=review_prompt,
            session_id=f"sovereign_review_{int(_time.time())}",
            source="gaia-doctor",
        ):
            if chunk.get("type") == "token":
                response_text += chunk.get("value", "")

        # Parse the response
        response_upper = response_text.upper()
        if "APPROVED" in response_upper:
            reason = response_text.strip().split(":", 1)[-1].strip() if ":" in response_text else "Approved by GAIA"
            logger.info("🔱 SOVEREIGN REVIEW: APPROVED — %s", reason[:200])
            return {"approved": True, "reason": reason[:500]}
        else:
            reason = response_text.strip().split(":", 1)[-1].strip() if ":" in response_text else "Denied by GAIA"
            logger.warning("🔱 SOVEREIGN REVIEW: DENIED — %s", reason[:200])
            return {"approved": False, "reason": reason[:500]}

    except Exception as e:
        logger.exception("Sovereign review failed")
        return JSONResponse(status_code=500, content={"approved": False, "reason": str(e)})


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration.

    Returns 200 with inference_ok=true if the Core inference backend
    is reachable, or 200 with inference_ok=false + degraded status
    if inference is down but the API is alive. Doctor uses this to
    distinguish between service-down and inference-down.
    """
    import os
    core_endpoint = os.environ.get("CORE_CPU_ENDPOINT", "http://localhost:8092")
    inference_ok = False
    inference_detail = ""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{core_endpoint}/health")
            inference_ok = resp.status_code == 200
            if not inference_ok:
                inference_detail = f"status={resp.status_code}"
    except Exception as _inf_exc:
        inference_detail = str(_inf_exc)[:100]

    status = "healthy" if inference_ok else "degraded"
    return JSONResponse(
        status_code=200,
        content={
            "status": status,
            "service": "gaia-core",
            "inference_ok": inference_ok,
            "inference_detail": inference_detail or "ok",
        }
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "gaia-core",
        "description": "GAIA Cognitive Loop Service",
        "endpoints": {
            "/health": "Health check",
            "/": "This endpoint",
            "/process_packet": "Process a CognitionPacket through the cognitive loop",
            "/status": "Get cognitive system status",
            "/gpu/status": "GPU state (active/sleeping)",
            "/gpu/release": "Put gaia-prime to sleep, free GPU",
            "/gpu/reclaim": "Wake gaia-prime, restore GPU inference",
            "/sleep/status": "Sleep cycle state machine status",
            "/sleep/wake": "Send wake signal (POST)",
            "/sleep/study-handoff": "Study handoff notification (POST)",
            "/sleep/distracted-check": "Check for canned response (GET)",
            "/sleep/shutdown": "Graceful shutdown (POST)",
            "/cognition/checkpoint": "Write cognitive checkpoints (POST)",
            "/api/sessions": "List all sessions (GET) / Create session (POST)",
            "/api/sessions/{id}/history": "Get session message history (GET)",
            "/api/sessions/{id}/summary": "Generate session summary (POST)",
            "/api/sessions/{id}/meta": "Update session metadata (PUT)",
        }
    }


@app.get("/status")
async def get_status():
    """Get the status of the cognitive system."""
    global _agent_core, _ai_manager

    if _agent_core is None or _ai_manager is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_initialized",
                "service": "gaia-core",
                "message": "Cognitive system not initialized",
            }
        )

    # Get model pool status
    model_pool = _ai_manager.model_pool
    available_models = list(model_pool.models.keys()) if hasattr(model_pool, 'models') else []
    model_status = getattr(model_pool, 'model_status', {})

    return {
        "status": "operational",
        "service": "gaia-core",
        "ai_manager": {
            "initialized": _ai_manager.status.get("initialized", False),
            "current_persona": _ai_manager.status.get("current_persona"),
        },
        "models": {
            "available": available_models,
            "status": model_status,
        }
    }


@app.post("/cognition/checkpoint")
async def cognition_checkpoint():
    """Write cognitive checkpoints (prime.md + Lite.md).

    Called by graceful_checkpoint.sh before container shutdown, or manually
    to persist cognitive state.  Safe to call multiple times.
    """
    results = _write_shutdown_checkpoints(app)

    if "error" in results:
        return JSONResponse(status_code=503, content=results)

    return JSONResponse(status_code=200, content=results)


@app.post("/api/kv-cache/save")
async def kv_cache_save():
    """Trigger an immediate KV cache save for all roles."""
    from gaia_core.cognition.kv_cache_manager import get_kv_cache_manager
    mgr = get_kv_cache_manager()
    if mgr is None:
        return JSONResponse(status_code=503, content={"error": "KV cache manager not initialized"})
    results = mgr.save_all()
    return {"status": "ok", "results": results}


@app.post("/api/kv-cache/restore/{role}")
async def kv_cache_restore(role: str):
    """Restore KV cache for a specific role (e.g., 'reflex' or 'core')."""
    from gaia_core.cognition.kv_cache_manager import get_kv_cache_manager, _CHECKPOINT_FILENAMES
    mgr = get_kv_cache_manager()
    if mgr is None:
        return JSONResponse(status_code=503, content={"error": "KV cache manager not initialized"})
    if role not in _CHECKPOINT_FILENAMES:
        return JSONResponse(status_code=400, content={"error": f"Unknown role: {role}. Valid: {list(_CHECKPOINT_FILENAMES.keys())}"})
    model = mgr._get_model_for_role(role)
    if model is None:
        return JSONResponse(status_code=404, content={"error": f"No model found for role '{role}'"})
    filename = _CHECKPOINT_FILENAMES[role]
    ok = model.restore_kv_cache(filename)
    return {"status": "ok" if ok else "failed", "role": role, "filename": filename}


@app.get("/api/kv-cache/pressure")
async def kv_cache_pressure():
    """Return KV cache pressure for all roles and recent compaction log."""
    from gaia_core.cognition.kv_cache_manager import get_kv_cache_manager
    mgr = get_kv_cache_manager()
    if mgr is None:
        return JSONResponse(status_code=503, content={"error": "KV cache manager not initialized"})
    return {
        "pressures": mgr.get_all_pressures(),
        "compaction_log": mgr.get_compaction_log(),
    }


@app.post("/api/kv-cache/compact/{role}")
async def kv_cache_compact(role: str):
    """Manually trigger KV cache compaction for a specific role."""
    from gaia_core.cognition.kv_cache_manager import get_kv_cache_manager, _CHECKPOINT_FILENAMES
    mgr = get_kv_cache_manager()
    if mgr is None:
        return JSONResponse(status_code=503, content={"error": "KV cache manager not initialized"})
    if role not in _CHECKPOINT_FILENAMES:
        return JSONResponse(status_code=400, content={
            "error": f"Unknown role: {role}. Valid: {list(_CHECKPOINT_FILENAMES.keys())}"
        })
    ok = mgr.compact(role, reason="manual API call")
    return {
        "status": "compacted" if ok else "failed",
        "role": role,
        "pressure_after": mgr.get_cache_pressure(role),
    }


# ── Session / Conversation Management ──────────────────────────────────────

@app.get("/api/sessions")
async def list_sessions():
    """List all sessions with metadata."""
    if _ai_manager is None:
        return JSONResponse(status_code=503, content={"error": "Cognitive system not initialized"})

    sm = _ai_manager.session_manager
    sessions_out = []
    for sid, session in sm.sessions.items():
        last_ts = session.last_message_timestamp()
        sessions_out.append({
            "session_id": sid,
            "title": session.meta.get("title", sid),
            "message_count": len(session.history),
            "created_at": session.created_at.isoformat(),
            "updated_at": last_ts.isoformat() if last_ts else session.created_at.isoformat(),
        })
    # Sort by most recently updated first
    sessions_out.sort(key=lambda s: s["updated_at"], reverse=True)
    return {"sessions": sessions_out}


@app.post("/api/sessions")
async def create_session(request: Request):
    """Create a new conversation session."""
    if _ai_manager is None:
        return JSONResponse(status_code=503, content={"error": "Cognitive system not initialized"})

    data = await request.json()
    session_id = data.get("session_id", f"web_{__import__('uuid').uuid4().hex[:8]}")
    title = data.get("title", "New conversation")

    sm = _ai_manager.session_manager
    session = sm.get_or_create_session(session_id)
    session.meta["title"] = title
    sm._save_state()

    return {
        "session_id": session_id,
        "title": title,
        "created_at": session.created_at.isoformat(),
    }


@app.get("/api/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get full message history for a session."""
    if _ai_manager is None:
        return JSONResponse(status_code=503, content={"error": "Cognitive system not initialized"})

    sm = _ai_manager.session_manager
    if session_id not in sm.sessions:
        return JSONResponse(status_code=404, content={"error": f"Session '{session_id}' not found"})

    history = sm.get_history(session_id)
    session = sm.sessions[session_id]
    return {
        "session_id": session_id,
        "title": session.meta.get("title", session_id),
        "messages": history,
    }


@app.post("/api/sessions/{session_id}/summary")
async def generate_session_summary(session_id: str):
    """Generate a summary of the conversation for the context pool."""
    if _ai_manager is None:
        return JSONResponse(status_code=503, content={"error": "Cognitive system not initialized"})

    sm = _ai_manager.session_manager
    if session_id not in sm.sessions:
        return JSONResponse(status_code=404, content={"error": f"Session '{session_id}' not found"})

    session = sm.sessions[session_id]
    if not session.history:
        return {"session_id": session_id, "summary": "", "keywords": []}

    try:
        summary = sm.summarizer.generate_summary(session.history, packet=None)
        keywords = sm.keyword_extractor.extract_keywords(session.history)
    except Exception as e:
        logger.warning("Summary generation failed for %s: %s", session_id, e)
        # Fallback: take first and last messages as a crude summary
        first_msg = session.history[0].get("content", "")[:200]
        last_msg = session.history[-1].get("content", "")[:200] if len(session.history) > 1 else ""
        summary = f"Conversation starting with: {first_msg}"
        if last_msg:
            summary += f" ... ending with: {last_msg}"
        keywords = []

    return {
        "session_id": session_id,
        "title": session.meta.get("title", session_id),
        "summary": summary,
        "keywords": keywords,
    }


@app.put("/api/sessions/{session_id}/meta")
async def update_session_meta(request: Request, session_id: str):
    """Update session metadata (title, etc.)."""
    if _ai_manager is None:
        return JSONResponse(status_code=503, content={"error": "Cognitive system not initialized"})

    sm = _ai_manager.session_manager
    if session_id not in sm.sessions:
        return JSONResponse(status_code=404, content={"error": f"Session '{session_id}' not found"})

    data = await request.json()
    session = sm.sessions[session_id]
    for key, value in data.items():
        session.meta[key] = value
    sm._save_state()

    return {"session_id": session_id, "meta": session.meta}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Archive and remove a session."""
    if _ai_manager is None:
        return JSONResponse(status_code=503, content={"error": "Cognitive system not initialized"})

    sm = _ai_manager.session_manager
    if session_id not in sm.sessions:
        return JSONResponse(status_code=404, content={"error": f"Session '{session_id}' not found"})

    session = sm.sessions[session_id]

    # Archive before deletion if there is history
    if session.history:
        try:
            sm.summarize_and_archive(session_id)
        except Exception as e:
            logger.warning("Pre-delete archive failed for %s: %s", session_id, e)

    # Remove from active sessions
    sm.reset_session(session_id)

    return {"status": "deleted", "session_id": session_id}


@app.post("/process_packet")
async def process_packet(packet_data: Dict[str, Any]):
    """
    Process a CognitionPacket through the cognitive loop with streaming support.
    Yields chunks of data: tokens as they are generated, and finally the completed packet.
    """
    global _agent_core, _ai_manager

    # Mark system active for sleep cycle idle tracking
    idle_monitor = getattr(app.state, "idle_monitor", None)
    if idle_monitor:
        idle_monitor.mark_active()

    if _agent_core is None or _ai_manager is None:
        raise HTTPException(
            status_code=503,
            detail="[GAIA-CORE-003] Cognitive system not initialized. Check logs for startup errors."
        )

    async def _run_loop():
        logger.info("Main: _run_loop generator started")
        async with _turn_semaphore:
            logger.info("Main: acquired turn semaphore")
            async for chunk in _run_loop_inner():
                yield chunk

    async def _run_loop_inner():
        try:
            # 1. Deserialize the packet correctly
            try:
                from gaia_common.protocols.cognition_packet import CognitionPacket
                packet = CognitionPacket.from_dict(packet_data)
            except Exception as e:
                log_gaia_error(logger, "GAIA-CORE-015", f"Failed to deserialize packet: {e}")
                yield json.dumps({"type": "error", "value": f"Invalid packet structure: {str(e)}"}) + "\n"
                return

            # Extract routing information
            user_input = packet.content.original_prompt
            session_id = packet.header.session_id

            # Determine source and destination from output_routing
            source = "web"
            destination = "web"
            metadata = {}

            if packet.header.output_routing:
                routing = packet.header.output_routing
                if routing.source_destination:
                    source = routing.source_destination.value if hasattr(routing.source_destination, 'value') else str(routing.source_destination)
                if routing.primary:
                    dest = routing.primary.destination
                    destination = dest.value if hasattr(dest, 'value') else str(dest)
                    metadata = {
                        "channel_id": routing.primary.channel_id,
                        "user_id": routing.primary.user_id,
                        "reply_to_message_id": routing.primary.reply_to_message_id,
                        "is_dm": routing.primary.metadata.get("is_dm", False) if routing.primary.metadata else False,
                    }

            logger.info(f"Processing packet {packet.header.packet_id}: '{user_input[:50]}...' from {source}")

            # Log to event buffer for episodic memory
            try:
                from gaia_common.event_buffer import log_event
                user_name = metadata.get("user_id", "unknown")
                log_event("conversation",
                          f"Message from {source}: \"{user_input[:80]}\"",
                          source="cognitive_pipeline")
            except Exception:
                pass

            # --- PRE-FLIGHT: Speculative Reflex ---
            # Trigger this BEFORE the heavy run_turn loop starts
            loop = asyncio.get_event_loop()
            reflex_text = ""
            history = _ai_manager.session_manager.get_history(session_id)
            if _agent_core.is_eligible_for_reflex(packet, history):
                logger.info("Main: Triggering instant speculative Reflex...")
                _reflex_t0 = _time.perf_counter()
                reflex_text = await loop.run_in_executor(
                    None, _agent_core.generate_instant_reflex, packet
                )
                if reflex_text:
                    # Uncertainty check: if Nano hedges or describes process
                    # without an actual answer, DON'T return — let Core handle it
                    user_input = packet.content.original_prompt or ""
                    if _agent_core._should_escalate_for_uncertainty(reflex_text, user_input):
                        logger.info("Main: Nano reflex uncertain — escalating to full pipeline")
                        reflex_text = ""  # Clear so full pipeline runs
                    else:
                        # Log reflex generation to stream
                        try:
                            from gaia_core.utils.generation_stream_logger import get_logger as _get_gen_logger
                            _gl = _get_gen_logger()
                            _reflex_elapsed = int((_time.perf_counter() - _reflex_t0) * 1000)
                            _gid = _gl.start_generation("reflex-0.5B", "nano", "response")
                            _gl.log_token(_gid, reflex_text)
                            _gl.end_generation(_gid)
                        except Exception:
                            pass
                        formatted_reflex = f"⚡ **[(Reflex) Nano]**\n{reflex_text}"
                        yield json.dumps({"type": "token", "value": formatted_reflex + "\n\n---\n\n"}) + "\n"
                        yield json.dumps({"type": "flush"}) + "\n"

                        # FINALIZATION: Skip run_turn if reflex already provided the answer
                        from gaia_common.protocols.cognition_packet import PacketState
                        packet.status.state = PacketState.COMPLETED
                        packet.response.candidate = reflex_text
                        yield json.dumps({"type": "packet", "value": packet.to_serializable_dict()}) + "\n"
                        return

            # Run the cognitive loop
            response_pieces = []
            final_packet_dict = None

            import re as _re
            _think_tag_re = _re.compile(r'</?(?:think|thinking)[^>]*>')
            _think_block_re = _re.compile(r'<(?:think|thinking)>.*?</(?:think|thinking)>\s*', _re.DOTALL)
            _think_unclosed_re = _re.compile(r'<(?:think|thinking)>.*$', _re.DOTALL)

            def _strip_think_token(text: str) -> str:
                """Strip think tags from a streaming token WITHOUT stripping whitespace."""
                if not text or "<" not in text:
                    return text
                result = _think_block_re.sub('', text)
                result = _think_unclosed_re.sub('', result)
                result = _think_tag_re.sub('', result)
                return result

            # AgentCore.run_turn is a synchronous generator. Each next()
            # call may block for seconds during llama_cpp inference.
            # Running in a thread executor prevents blocking the uvicorn
            # event loop, keeping /health and other endpoints responsive.
            loop = asyncio.get_event_loop()
            logger.info("Main: creating run_turn generator (agent_core=%s)", type(_agent_core).__name__ if _agent_core else "None")
            gen = _agent_core.run_turn(
                user_input=user_input,
                session_id=session_id,
                destination=destination,
                source=source,
                metadata=metadata,
                reflex_text=reflex_text
            )
            logger.info("Main: run_turn generator created, calling first next()")

            def _next_event():
                try:
                    return next(gen)
                except StopIteration:
                    return None
                except Exception as _gen_exc:
                    logger.error("run_turn generator exception: %s", _gen_exc, exc_info=True)
                    return None

            while True:
                event = await loop.run_in_executor(None, _next_event)
                if event is None:
                    break
                if isinstance(event, dict):
                    if event.get("type") == "token":
                        val = event.get("value", "")
                        val = _strip_think_token(val)
                        if not val:
                            continue
                        response_pieces.append(val)
                        # Yield token immediately for real-time UI updates
                        yield json.dumps({"type": "token", "value": val}) + "\n"
                    elif event.get("type") == "flush":
                        # Signal to front-ends to flush their buffers
                        yield json.dumps(event) + "\n"
                    elif event.get("type") == "packet":
                        # Store the final packet to yield at the very end
                        final_packet_dict = event.get("value")

            # Ensure a flush is always emitted so front-ends send accumulated text
            if response_pieces:
                yield json.dumps({"type": "flush"}) + "\n"

            # Finalize response processing
            full_response = "".join(response_pieces)
            from gaia_core.utils.output_router import _strip_think_tags_robust
            full_response = _strip_think_tags_robust(full_response)

            if final_packet_dict:
                # Ensure the final response in the packet is clean (no think tags)
                if "response" in final_packet_dict and "candidate" in final_packet_dict["response"]:
                    final_packet_dict["response"]["candidate"] = full_response
                
                yield json.dumps({"type": "packet", "value": final_packet_dict}) + "\n"

            # Reset idle timer after response completes
            if idle_monitor:
                idle_monitor.mark_active()

        except Exception as e:
            logger.exception(f"Error in streaming turn loop: {e}")
            yield json.dumps({"type": "error", "value": str(e)}) + "\n"

    return StreamingResponse(_run_loop(), media_type="application/x-ndjson")


# ── Audio Context Ingest ─────────────────────────────────────────────

from collections import deque
import time as _time

_audio_context_buffer: deque = deque(maxlen=30)  # ~15 minutes at 30s chunks
_audio_listening_active: bool = False  # Only feed buffer to prompt builder when True


class AudioIngestRequest(BaseModel):
    transcript: str
    mode: str = "passive"
    context_markers: list = []
    timestamp: str | None = None


@app.post("/audio/ingest")
async def audio_ingest(req: AudioIngestRequest):
    """Ingest audio transcript as ambient context (no cognitive loop).

    Stores transcripts in a ring buffer that the prompt builder can
    reference during the next cognitive turn.
    """
    if not req.transcript.strip():
        return {"status": "skipped", "reason": "empty transcript"}

    entry = {
        "text": req.transcript.strip(),
        "mode": req.mode,
        "context_markers": req.context_markers,
        "timestamp": req.timestamp or _time.strftime("%H:%M:%S"),
        "ingested_at": _time.time(),
    }
    _audio_context_buffer.append(entry)

    logger.info("Audio ingest: %d chars, markers=%s, buffer=%d/%d",
                len(req.transcript), req.context_markers,
                len(_audio_context_buffer), _audio_context_buffer.maxlen)

    # Mark system active for sleep cycle idle tracking
    idle_monitor = getattr(app.state, "idle_monitor", None)
    if idle_monitor:
        idle_monitor.mark_active()

    return {
        "status": "ingested",
        "buffer_size": len(_audio_context_buffer),
        "buffer_max": _audio_context_buffer.maxlen,
    }


@app.post("/audio/listen")
async def audio_listen_toggle(enabled: bool = True):
    """Enable or disable feeding audio buffer into the prompt builder.

    When enabled, GAIA will see recent audio transcripts as ambient context
    during cognitive turns. When disabled, the buffer still accumulates
    but is not injected into prompts.
    """
    global _audio_listening_active
    _audio_listening_active = enabled
    state = "active" if enabled else "paused"
    logger.info("Audio listening %s (buffer has %d entries)", state, len(_audio_context_buffer))
    return {"listening": _audio_listening_active, "buffer_size": len(_audio_context_buffer)}


@app.get("/audio/context")
async def audio_context():
    """Return the current audio context buffer and listening state."""
    return {
        "listening": _audio_listening_active,
        "entries": list(_audio_context_buffer),
        "count": len(_audio_context_buffer),
        "max": _audio_context_buffer.maxlen,
    }


# ── Cognitive Similarity Endpoint ─────────────────────────────────────
# Used by gaia-doctor's cognitive test battery for open-ended validation.

class CognitiveQueryRequest(BaseModel):
    prompt: str
    system: str = "You are GAIA, a sovereign AI agent. Answer concisely and accurately."
    max_tokens: int = 512
    temperature: float = 0.3
    target: str = "core"  # "core" (CPU llama-server), "prime" (GPU vLLM), "nano"
    no_think: bool = False  # Suppress <think> reasoning blocks (Qwen3+)


class SimilarityRequest(BaseModel):
    text: str
    reference: str


@app.post("/api/cognitive/similarity")
async def cognitive_similarity(req: SimilarityRequest):
    """Rate semantic similarity between two texts using the Nano/Lite model.

    Returns a score from 0.0 to 1.0.
    """
    prompt = (
        "Rate the semantic similarity between these two texts on a scale of 0.0 to 1.0.\n"
        "Reply with ONLY a JSON object: {\"score\": 0.XX}\n\n"
        f"Text A: {req.text[:500]}\n\n"
        f"Text B: {req.reference[:500]}"
    )
    try:
        if _agent_core and hasattr(_agent_core, 'model_pool'):
            pool = _agent_core.model_pool
            result = await pool.generate(
                prompt=prompt,
                role="nano",
                max_tokens=32,
                temperature=0.1,
            )
            # Parse score from response
            import re as _re
            text = result if isinstance(result, str) else str(result)
            match = _re.search(r'"score"\s*:\s*([\d.]+)', text)
            if match:
                score = min(max(float(match.group(1)), 0.0), 1.0)
                return {"score": score}
        # Fallback: basic token overlap
        t_tokens = set(req.text.lower().split())
        r_tokens = set(req.reference.lower().split())
        if not r_tokens:
            return {"score": 1.0}
        overlap = len(t_tokens & r_tokens) / max(len(r_tokens), 1)
        return {"score": round(min(overlap, 1.0), 4)}
    except Exception as e:
        logger.warning("Similarity endpoint error: %s", e)
        # Fallback
        t_tokens = set(req.text.lower().split())
        r_tokens = set(req.reference.lower().split())
        overlap = len(t_tokens & r_tokens) / max(len(r_tokens), 1)
        return {"score": round(min(overlap, 1.0), 4)}


@app.post("/api/cognitive/query")
async def cognitive_query(req: CognitiveQueryRequest):
    """Lightweight LLM query endpoint for cognitive testing.

    Bypasses the full 20-stage cognitive pipeline — sends the prompt
    directly to a model backend. Much faster (~5s vs ~90s).

    target options:
      - "core" (default): embedded llama-server (CPU, GGUF)
      - "prime": gaia-prime vLLM (GPU, merged model)
      - "nano": embedded nano llama-server (CPU, small GGUF)
    """
    import httpx

    target_endpoints = {
        "core": os.environ.get("CORE_CPU_ENDPOINT", "http://localhost:8092"),
        "prime": os.environ.get("PRIME_ENDPOINT", "http://gaia-prime:7777"),
        "nano": os.environ.get("NANO_ENDPOINT", "http://localhost:8093"),
    }
    base = target_endpoints.get(req.target, target_endpoints["core"])
    url = f"{base.rstrip('/')}/v1/chat/completions"
    timeout = 60.0 if req.target == "prime" else 45.0

    system_content = req.system

    # Inject world state + architecture awareness so the model has live context.
    # Operational facts (ports, services, pipeline stages) belong in context,
    # NOT in weights. This is the Curriculum Split principle.
    try:
        from gaia_common.utils.world_state import format_world_state_snapshot
        world_state = format_world_state_snapshot(max_lines=6)
        if world_state:
            system_content += f"\n\nCurrent System State:\n{world_state}"
    except Exception:
        pass

    try:
        from pathlib import Path
        awareness_path = Path(os.environ.get("KNOWLEDGE_DIR", "/knowledge")) / "awareness" / "operational" / "architecture_facts.md"
        if awareness_path.exists():
            facts = awareness_path.read_text(encoding="utf-8").strip()
            # Truncate to keep prompt tight — VRAM is limited
            if len(facts) > 1200:
                facts = facts[:1200] + "\n..."
            system_content += f"\n\nArchitecture Reference:\n{facts}"
    except Exception:
        pass

    if req.no_think:
        system_content += " /no_think"

    payload = {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": req.prompt},
        ],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
    }
    # Qwen3 chat_template: enable_thinking=false suppresses <think> blocks
    if req.no_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return {"content": text.strip(), "target": req.target}
    except Exception as e:
        logger.warning("Cognitive query error (target=%s): %s", req.target, e)
        return {"content": "", "error": str(e), "target": req.target}


# ── Embedded llama-server Management ─────────────────────────────────
# Used by the self-awareness training pipeline to release/reload the
# Core CPU model without restarting the gaia-core container.

from gaia_core.model_server import get_model_server as _get_model_server


class ModelReloadRequest(BaseModel):
    model_path: str | None = None


@app.post("/model/release")
async def model_release():
    """Stop the embedded llama-server and free RAM.

    Called by the training pipeline before GGUF overwrite.
    """
    ms = _get_model_server()
    result = ms.release()
    status_code = 200 if result.get("ok") else 500
    return JSONResponse(status_code=status_code, content=result)


@app.post("/model/reload")
async def model_reload(req: ModelReloadRequest = None):
    """Start the embedded llama-server (optionally with a new GGUF).

    Called by the training pipeline after deploying a new Core model.
    """
    model_path = req.model_path if req else None
    ms = _get_model_server()
    result = ms.reload(model_path=model_path)
    status_code = 200 if result.get("ok") else 500
    return JSONResponse(status_code=status_code, content=result)


@app.get("/model/status")
async def model_status():
    """Return the current embedded llama-server status."""
    ms = _get_model_server()
    return ms.status()


# ── Cognitive Status Aliases ───────────────────────────────────────────
# Convenience endpoints expected by test plans and gaia-doctor.
# These delegate to existing functionality.

@app.get("/cognitive/status")
async def cognitive_status():
    """Cognitive system status — alias of /status with cognitive framing."""
    global _agent_core, _ai_manager
    if _agent_core is None or _ai_manager is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_initialized", "cognitive_ready": False},
        )
    model_pool = _ai_manager.model_pool
    return {
        "status": "operational",
        "cognitive_ready": True,
        "current_persona": _ai_manager.status.get("current_persona"),
        "models_loaded": list(model_pool.models.keys()) if hasattr(model_pool, "models") else [],
    }


@app.get("/cognitive/monitor")
async def cognitive_monitor():
    """Lightweight cognitive health monitor for dashboards and test plans."""
    global _agent_core, _ai_manager
    initialized = _agent_core is not None and _ai_manager is not None
    info: dict = {
        "cognitive_initialized": initialized,
        "service": "gaia-core",
    }
    if initialized:
        info["current_persona"] = _ai_manager.status.get("current_persona")
        info["turn_semaphore_locked"] = _turn_semaphore.locked()
    return info


def get_audio_context_for_prompt(max_entries: int = 10, max_chars: int = 2000) -> str | None:
    """Return formatted audio context for prompt injection.

    Called by the prompt builder. Returns None if listening is inactive
    or the buffer is empty.
    """
    if not _audio_listening_active or not _audio_context_buffer:
        return None

    lines = []
    total_chars = 0
    # Take the most recent entries, newest last
    entries = list(_audio_context_buffer)[-max_entries:]
    for entry in entries:
        markers = entry.get("context_markers", [])
        marker_str = f" [{', '.join(markers)}]" if markers else ""
        line = f"[{entry['timestamp']}]{marker_str} {entry['text']}"
        if total_chars + len(line) > max_chars:
            break
        lines.append(line)
        total_chars += len(line)

    if not lines:
        return None

    return (
        "── Ambient Audio Context (system audio transcription) ──\n"
        + "\n".join(lines)
        + "\n── End Audio Context ──"
    )

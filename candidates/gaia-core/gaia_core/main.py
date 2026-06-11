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


def _resolve_cfr_recall(history, rid):
    """CFR Phase 2b: resolve a BLURred conversation turn's full text by id.

    The page-fault handler for conversation-as-virtual-memory. When GAIA emits
    an expand_context tool call to page a set-aside turn back, we resolve it
    locally from this session's history (the backing store is always resident
    here) — no MCP round-trip. Returns a compact result dict, or None if the id
    isn't found. Pure + side-effect-free so it can be unit-tested in isolation.
    """
    if not rid or not history:
        return None
    for m in history:
        if str(m.get("id")) != str(rid):
            continue
        text = (m.get("content") or "")
        try:
            from gaia_core.utils.output_router import _strip_think_tags_robust
            text = _strip_think_tags_robust(text)
        except Exception:
            pass  # strip is best-effort; never let it drop a valid match
        return {"recalled_turn_id": str(rid), "role": m.get("role", "?"), "text": text[:2000]}
    return None


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
            # Pass turn semaphore and event loop for Initiative Bridge (Phase 6)
            try:
                _loop = asyncio.get_running_loop()
            except RuntimeError:
                _loop = None
            _sleep_loop = SleepCycleLoop(
                config,
                model_pool=_ai_manager.model_pool if _ai_manager else None,
                agent_core=_agent_core,
                session_manager=_ai_manager.session_manager if _ai_manager else None,
                turn_semaphore=_turn_semaphore,
                event_loop=_loop,
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
    """Write prime.md and Core.md checkpoints (called on shutdown and via endpoint)."""
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

    # Write Core.md (formerly Lite.md)
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
            results["core"] = {"status": "ok", "chars": len(entry)}
            logger.info("Shutdown checkpoint: Core.md entry written")
        else:
            results["core"] = {"status": "skipped", "reason": "no Core model available"}
            logger.info("Shutdown checkpoint: Core.md skipped (no model)")
    except Exception as exc:
        results["core"] = {"status": "error", "detail": str(exc)}
        logger.error("Shutdown checkpoint: Core.md failed: %s", exc, exc_info=True)

    return results


app = FastAPI(
    title="GAIA Core",
    description="The Brain - Cognitive loop and reasoning engine",
    version="0.1.0",
    lifespan=lifespan,
)

# Inter-service HMAC authentication
try:
    from gaia_common.utils.service_auth import AuthMiddleware
    if AuthMiddleware:
        app.add_middleware(AuthMiddleware)
except ImportError:
    pass

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


_core_boot_time = __import__("time").monotonic()
_STARTUP_GRACE_SECONDS = 90  # Engine model load takes ~60s after container start


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration.

    Returns 200 with inference_ok=true if the Core inference backend
    is reachable, or 200 with status=healthy during the startup grace
    period while models are still loading (not degraded — loading is
    expected, not broken).  After the grace window, reports degraded
    if inference is unreachable so Doctor can remediate.
    """
    import os, time as _t
    core_endpoint = os.environ.get("CORE_CPU_ENDPOINT", "http://localhost:8092")
    inference_ok = False
    inference_detail = ""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{core_endpoint}/health")
            inference_ok = resp.status_code == 200
            if not inference_ok:
                inference_detail = f"status={resp.status_code}"
    except Exception as _inf_exc:
        inference_detail = str(_inf_exc)[:100]

    in_grace = (_t.monotonic() - _core_boot_time) < _STARTUP_GRACE_SECONDS

    if inference_ok:
        status = "healthy"
    elif in_grace:
        status = "healthy"  # Still loading — don't trigger remediation
        inference_detail = inference_detail or "engine loading (startup grace)"
    else:
        status = "degraded"

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
    """Restore KV cache for a specific role (e.g., 'nano' or 'core')."""
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

    # ── Readiness Gate ──────────────────────────────────────────────────
    # Verify inference engine is reachable before processing.
    # Health check only — no test inference (that adds 500ms+ per request).
    try:
        import httpx as _httpx
        _engine_ok = False
        try:
            _r = _httpx.get("http://localhost:8092/health", timeout=3)
            if _r.status_code == 200:
                _health = _r.json()
                if _health.get("model_loaded") or _health.get("status") == "ok":
                    _engine_ok = True
        except Exception:
            pass

        if not _engine_ok:
            logger.warning("Readiness gate: inference engine not ready")
            # Try to trigger a wake signal
            try:
                _wake_url = getattr(app.state, "_orchestrator_url", "http://gaia-orchestrator:6410")
                _httpx.post(f"{_wake_url}/gpu/wake", json={}, timeout=5)
            except Exception:
                pass

            # Return a friendly message — don't try to process the request
            # from inside the gate (causes scoping issues with _run_loop).
            # The user's client will see this and can retry.
            async def _waking_response():
                yield json.dumps({
                    "type": "token",
                    "value": "One moment — my inference engine is loading. Try again in a few seconds."
                }) + "\n"
                yield json.dumps({"type": "flush"}) + "\n"

            return StreamingResponse(_waking_response(), media_type="application/x-ndjson")
    except ImportError:
        pass  # httpx not available — skip gate

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

            # v0.5: Pass audio_payloads through metadata for NeuralRouter (Proposal 11)
            _audio = getattr(getattr(packet, 'content', None), 'audio_payloads', None)

            # shj: gaia-audio STT fallback. CORE_AUDIO_NATIVE controls whether
            # audio_payloads route to Core's multimodal audio path (default,
            # value="1") or are pre-transcribed by gaia-audio's Qwen3-ASR and
            # injected as text (value="0"). Until V7 audio quality lands
            # (m0b), operators can flip to "0" for production-quality
            # speech responses while Core's audio side trains.
            _audio_native = os.environ.get("CORE_AUDIO_NATIVE", "1").lower() in ("1", "true", "yes", "on")
            if _audio and not _audio_native:
                try:
                    import httpx as _httpx
                    _audio_url = os.environ.get("AUDIO_ENDPOINT", "http://gaia-audio:8080")
                    _ap = _audio[0]
                    _b64_payload = getattr(_ap, "data_base64", "") or ""
                    _sr = getattr(_ap, "sample_rate", 16000) or 16000
                    if _b64_payload:
                        async with _httpx.AsyncClient(timeout=30.0) as _stt_client:
                            _r = await _stt_client.post(
                                f"{_audio_url}/transcribe",
                                json={"audio_base64": _b64_payload, "sample_rate": int(_sr)},
                            )
                            if _r.status_code == 200:
                                _transcript = (_r.json() or {}).get("text", "").strip()
                                if _transcript:
                                    # Replace user_input with the transcript and
                                    # clear audio_payloads so Core sees text-only.
                                    user_input = _transcript
                                    packet.content.original_prompt = _transcript
                                    packet.content.audio_payloads = []
                                    _audio = None
                                    logger.info("CORE_AUDIO_NATIVE=0 — transcribed audio via gaia-audio STT (%d chars)",
                                                len(_transcript))
                                else:
                                    logger.warning("STT fallback returned empty transcript — leaving audio for Core path")
                            else:
                                logger.warning("STT fallback HTTP %d — leaving audio for Core path", _r.status_code)
                except Exception as _stt_err:
                    logger.warning("STT fallback failed (%s) — leaving audio for Core path", _stt_err)

            if _audio:
                metadata["_audio_payloads"] = _audio

            # 8ki: forward image attachments from the incoming packet so the
            # multimodal pipeline can consume them. run_turn rebuilds a fresh
            # packet via _create_initial_packet, which would otherwise drop
            # the inbound content.attachments.
            _inbound_attachments = list(getattr(getattr(packet, 'content', None), 'attachments', None) or [])

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

            # --- PRE-FLIGHT: Blast Shield (agent-input layer) ---
            # Catch destructive shell patterns BEFORE any model call. Prompt-
            # based hedging on Core is unreliable; a deterministic regex
            # short-circuits before reflex/run_turn ever fire. Mirrors the
            # patterns MCP's run_shell Blast Shield blocks at execution time.
            _bs_reason = None
            if user_input:
                import re as _re_blast_main
                _destructive_re_main = _re_blast_main.compile(
                    r"(?:^|[\s;&|`(])"
                    r"(?:"
                    r"rm\s+(?:-[A-Za-z]*r[A-Za-z]*f?|-[A-Za-z]*f[A-Za-z]*r)\s+/+(?:\s|$|--)"
                    r"|sudo\s+rm\s+-[A-Za-z]*r"
                    r"|dd\s+(?:if|of)=/dev/(?:sd|hd|nvme|mmcblk)"
                    r"|mkfs(?:\.[a-z0-9]+)?\s+/dev/"
                    r"|shred\s+(?:-[A-Za-z]+\s+)*/(?:dev|etc|boot|root)"
                    r"|chmod\s+-R\s+(?:777|000)\s+/"
                    r"|>\s*/dev/sd[a-z]"
                    r"|:\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"
                    r")",
                    _re_blast_main.IGNORECASE,
                )
                if _destructive_re_main.search(user_input):
                    _bs_reason = "destructive shell pattern"

            if _bs_reason:
                logger.warning(
                    "[BLAST SHIELD] %s in user input — refusing without model call: %r",
                    _bs_reason, (user_input or "")[:120],
                )
                _refusal = (
                    "[Core]\n\nI won't run that. The command you asked for "
                    "is destructive — it would wipe filesystems or destroy "
                    "the host. Capability is not consent: even with shell "
                    "access I refuse irreversible, system-destroying "
                    "operations.\n\nIf you're testing my safety guard, this "
                    "is the guard. If you actually need something "
                    "destructive-looking — clearing a sandbox directory, "
                    "wiping a specific dev partition during install, etc. — "
                    "give me a narrower path and explicit intent."
                )
                yield json.dumps({"type": "token", "value": _refusal}) + "\n"
                yield json.dumps({"type": "flush"}) + "\n"
                from gaia_common.protocols.cognition_packet import PacketState
                packet.status.state = PacketState.COMPLETED
                packet.response.candidate = _refusal
                yield json.dumps({"type": "packet", "value": packet.to_serializable_dict()}) + "\n"
                return

            # --- PRE-FLIGHT: Stakes clarification (GAIA_Project-pbb) ---
            # Phase 3 of 6ho. Two paths:
            #   1. PRIOR ASK pending: treat this turn as the reply,
            #      resolve, and replay the original utterance with an
            #      explicit IC/OOC marker so the rest of the pipeline
            #      runs with the right framing.
            #   2. NEW UTTERANCE ambiguous + low-confidence + not in
            #      debounce window → emit a clarifying question and
            #      short-circuit. The Phase 2 agent_core hook still
            #      runs to stash the classification on the packet.
            _stakes_clarif_msg = None
            try:
                from gaia_core.cognition.stakes_clarification import (
                    decide_clarification, resolve_clarification_reply,
                    pending_clarification,
                )
                from gaia_core.cognition.stakes_classifier import (
                    classify_stakes, is_role_play_active,
                )

                _pending = pending_clarification(session_id)
                if _pending:
                    _reply = resolve_clarification_reply(
                        session_id, user_input or "",
                    )
                    if _reply and _reply.resolution != "unresolved":
                        # Replay the original utterance with the
                        # resolved framing. The IC/OOC marker pushes
                        # the downstream classifier into high-confidence
                        # mode and skips re-asking.
                        _marker = (
                            "ooc:" if _reply.resolution == "real_world"
                            else "in-character:"
                        )
                        logger.info(
                            "[STAKES] Resolved %s — replaying %r with %s marker",
                            _reply.resolution,
                            _reply.original_user_input[:60],
                            _marker,
                        )
                        user_input = (
                            f"{_marker} {_reply.original_user_input}\n\n"
                            f"[User confirmed: "
                            f"{_reply.resolution.replace('_', '-')}]"
                        )
                        try:
                            packet.content.original_prompt = user_input
                        except Exception:
                            pass
                    # else: unresolved reply or no reply parsed —
                    # let the new utterance go through the classifier
                    # below as fresh input.

                if user_input and _stakes_clarif_msg is None:
                    _stakes = classify_stakes(
                        user_input,
                        role_play_active=is_role_play_active(),
                    )
                    _decision = decide_clarification(
                        _stakes,
                        session_id=session_id,
                        user_input=user_input,
                    )
                    if _decision.ask:
                        _stakes_clarif_msg = _decision.question
            except Exception:
                logger.debug("Stakes clarification flow failed", exc_info=True)

            if _stakes_clarif_msg:
                logger.info(
                    "[STAKES CLARIFICATION] Asking before answering: %s",
                    _stakes_clarif_msg[:80],
                )
                yield json.dumps(
                    {"type": "token", "value": _stakes_clarif_msg}
                ) + "\n"
                yield json.dumps({"type": "flush"}) + "\n"
                from gaia_common.protocols.cognition_packet import PacketState
                packet.status.state = PacketState.COMPLETED
                packet.response.candidate = _stakes_clarif_msg
                yield json.dumps(
                    {"type": "packet",
                     "value": packet.to_serializable_dict()}
                ) + "\n"
                return

            # --- PRE-FLIGHT: unreachable-path hedge ---
            # Detect file paths in user_input that fall OUTSIDE the MCP allow-
            # list (/gaia/GAIA_Project, /knowledge, /gaia-common, /sandbox).
            # We don't refuse — we append a system aside so the model knows
            # to hedge ("I don't have access to /tmp") instead of fabricating.
            # Reflex is disabled for these so the full pipeline handles them
            # carefully (reflex on Core has no path awareness).
            _reflex_disabled_for_path = False
            if user_input:
                _path_match_main = _re_blast_main.search(
                    r"(?:^|\s)((?:/tmp|/etc|/root|/boot|/var|/proc|/sys|/dev|/run|/mnt|/media|/srv)/"
                    r"[A-Za-z0-9_./\-]+)",
                    user_input,
                )
                if _path_match_main:
                    _unreachable = _path_match_main.group(1)
                    logger.info(
                        "[ALLOW-LIST HEDGE] user mentioned unreachable path %r — appending hedge to user_input, disabling reflex",
                        _unreachable,
                    )
                    user_input = (
                        f"{user_input}\n\n"
                        f"[SYSTEM ASIDE: The path {_unreachable} is OUTSIDE "
                        f"your MCP file-read allow-list (only "
                        f"/gaia/GAIA_Project, /knowledge, /gaia-common, "
                        f"/sandbox are reachable). You cannot read this "
                        f"file. Do NOT fabricate its contents. Tell the "
                        f"user you don't have access to that path.]"
                    )
                    try:
                        packet.content.original_prompt = user_input
                    except Exception:
                        pass
                    _reflex_disabled_for_path = True

            # --- PRE-FLIGHT: Speculative Reflex ---
            # Trigger this BEFORE the heavy run_turn loop starts.
            #
            # GAIA_Project-19i: skip reflex when the user has explicitly
            # routed to a heavier tier or when the orchestrator is already
            # FOCUSING (Prime on GPU). See reflex_gate.should_skip_reflex.
            from gaia_core.cognition.reflex_gate import should_skip_reflex
            _lc = getattr(_agent_core, "_lifecycle_client", None) \
                or getattr(app.state, "lifecycle_client", None)
            _reflex_skip_reason = should_skip_reflex(
                user_input or "", packet, _lc,
            )

            loop = asyncio.get_event_loop()
            reflex_text = ""
            history = _ai_manager.session_manager.get_history(session_id)
            _reflex_eligible = (
                not _reflex_disabled_for_path
                and _reflex_skip_reason is None
                and _agent_core.is_eligible_for_reflex(packet, history)
            )
            if _reflex_skip_reason is not None:
                logger.info("Main: Reflex skipped — %s", _reflex_skip_reason)
            if _reflex_eligible:
                logger.info("Main: Triggering instant speculative Reflex...")
                _reflex_t0 = _time.perf_counter()
                reflex_text = await loop.run_in_executor(
                    None, _agent_core.generate_instant_reflex, packet
                )
                if reflex_text:
                    # Uncertainty check: if reflex hedges or describes process
                    # without an actual answer, DON'T return — let Core handle it
                    user_input = packet.content.original_prompt or ""
                    # Tool-call escape hatch: if reflex emitted a tool_call
                    # envelope, the actual tool hasn't executed yet. Finalizing
                    # here would show the user the envelope as the answer.
                    # Always escalate so the agent layer parses + executes the
                    # call and synthesizes a real follow-up response.
                    _reflex_has_toolcall = (
                        "<tool_call>" in reflex_text and "</tool_call>" in reflex_text
                    )
                    if _reflex_has_toolcall:
                        logger.info("Main: Reflex emitted tool_call — escalating to full pipeline to execute it")
                        reflex_text = ""
                    elif _agent_core._should_escalate_for_uncertainty(reflex_text, user_input):
                        logger.info("Main: Reflex uncertain — escalating to full pipeline")
                        reflex_text = ""  # Clear so full pipeline runs
                    else:
                        # Log reflex generation to stream
                        _reflex_model = getattr(_agent_core, '_last_responding_model', None) or "core"
                        try:
                            from gaia_core.utils.generation_stream_logger import get_logger as _get_gen_logger
                            _gl = _get_gen_logger()
                            _reflex_elapsed = int((_time.perf_counter() - _reflex_t0) * 1000)
                            _gid = _gl.start_generation(_reflex_model, _reflex_model, "response")
                            _gl.log_token(_gid, reflex_text)
                            _gl.end_generation(_gid)
                        except Exception:
                            pass
                        _reflex_label = "Operator" if _reflex_model == "core" else _reflex_model.title()
                        formatted_reflex = f"⚡ **[({_reflex_label}) {_reflex_model.title()}]**\n{reflex_text}"
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

            # ── Native tool call parser ──────────────────────────────────
            # Intercepts <tool_call> tags in model output, executes via MCP,
            # and can trigger a continuation generation with the result.
            try:
                from gaia_common.utils.tool_call_parser import (
                    ToolCallParser, ParseEventType, format_tool_result,
                    TOOL_CALL_OPEN, TOOL_CALL_CLOSE,
                )
                _tc_parser = ToolCallParser()
                _tc_enabled = True
            except ImportError:
                _tc_parser = None
                _tc_enabled = False
                TOOL_CALL_OPEN = "<tool_call>"
                TOOL_CALL_CLOSE = "</tool_call>"

            # AgentCore.run_turn is a synchronous generator. Each next()
            # call may block for seconds during llama_cpp inference.
            # Running in a thread executor prevents blocking the uvicorn
            # event loop, keeping /health and other endpoints responsive.
            loop = asyncio.get_event_loop()
            gen = _agent_core.run_turn(
                user_input=user_input,
                session_id=session_id,
                destination=destination,
                source=source,
                metadata=metadata,
                reflex_text=reflex_text,
                attachments=_inbound_attachments,
            )

            def _next_event():
                try:
                    return next(gen)
                except StopIteration:
                    return None

            _pending_tool_calls = []
            _seen_tool_calls = set()  # Dedup repeated tool calls

            # Suppress tool calls on tool-free conversational intents — Core
            # habitually emits e.g. worldstate() even on a greeting, which adds
            # "[calling...]" noise and fragments the reply. Greetings/chitchat
            # never need a tool, so drop such calls silently (the surrounding
            # reply text still streams).
            _suppress_tools = False
            try:
                from gaia_core.cognition.nlu.embed_intent_classifier import EmbedIntentClassifier
                _eic = EmbedIntentClassifier.instance()
                if _eic.ready:
                    _intent_lbl, _ = _eic.classify(user_input or "")
                    _suppress_tools = _intent_lbl in {
                        "greeting", "farewell", "gratitude", "smalltalk",
                        "social", "chitchat", "acknowledgment", "affirmation",
                        # 'time' is world-state-answerable: the clock is already
                        # injected in the prompt, so drop the redundant (noisy)
                        # worldstate tool call and let Core answer from context.
                        "time"}
                    if _suppress_tools:
                        logger.info("Chat: tool-free intent '%s' — suppressing tool calls", _intent_lbl)
            except Exception:
                logger.debug("intent pre-check for tool suppression failed (non-fatal)", exc_info=True)

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

                        # Feed through tool_call parser if enabled
                        if _tc_enabled and _tc_parser:
                            parse_events = _tc_parser.feed(val)
                            for pe in parse_events:
                                if pe.type == ParseEventType.TEXT:
                                    response_pieces.append(pe.text)
                                    yield json.dumps({"type": "token", "value": pe.text}) + "\n"
                                elif pe.type == ParseEventType.TOOL_CALL_DETECTED:
                                    # Tool-free conversational turn: drop the call
                                    # silently — no "[calling...]" display, no MCP
                                    # round-trip. The reply text still streams.
                                    # expand_context is the CFR recall fault-handler — never
                                    # suppress it; it's how GAIA pages a set-aside turn back
                                    # into focus, and is meaningful even on a chatty turn.
                                    if _suppress_tools and pe.tool_name != "expand_context":
                                        logger.info("Tool call '%s' suppressed (tool-free intent)", pe.tool_name)
                                        continue
                                    # Dedup: only process each unique tool call once
                                    _tc_key = f"{pe.tool_name}:{pe.tool_action}:{json.dumps(pe.tool_params, sort_keys=True)}"
                                    if _tc_key in _seen_tool_calls:
                                        logger.debug("Duplicate tool call skipped: %s", _tc_key[:80])
                                        continue
                                    _seen_tool_calls.add(_tc_key)

                                    _action_display = f"({pe.tool_action})" if pe.tool_action else ""
                                    tool_display = f"\n*[calling {pe.tool_name}{_action_display}...]*\n"
                                    yield json.dumps({"type": "token", "value": tool_display}) + "\n"
                                    yield json.dumps({"type": "flush"}) + "\n"
                                    _pending_tool_calls.append(pe)

                                    # Stop generation after first tool call — execute it,
                                    # then continue generation with the result
                                    break
                                elif pe.type == ParseEventType.TOOL_ERROR:
                                    response_pieces.append(pe.text)
                                    yield json.dumps({"type": "token", "value": pe.text}) + "\n"

                        # If we just detected a tool call, break the generation loop
                        # to execute it immediately
                        if _pending_tool_calls and _seen_tool_calls:
                            break
                        # else: parser already yielded the text — don't re-emit

                        elif not _tc_enabled or not _tc_parser:
                            # No parser — pass through raw
                            response_pieces.append(val)
                            yield json.dumps({"type": "token", "value": val}) + "\n"

                    elif event.get("type") == "flush":
                        yield json.dumps(event) + "\n"
                    elif event.get("type") == "packet":
                        final_packet_dict = event.get("value")
                    else:
                        # Handle any other dictionary types (e.g. self-improvement progress)
                        # by dumping to JSON so StreamingResponse can encode it.
                        logger.debug(f"Yielding unknown event type: {event.get('type', 'unknown')}")
                        yield json.dumps(event) + "\n"

            # Flush any remaining parser buffer
            if _tc_enabled and _tc_parser:
                for pe in _tc_parser.flush():
                    if pe.type == ParseEventType.TEXT and pe.text:
                        response_pieces.append(pe.text)
                        yield json.dumps({"type": "token", "value": pe.text}) + "\n"

            # ── Execute pending tool calls and generate continuation ───
            if _pending_tool_calls:
                for tc in _pending_tool_calls:
                    try:
                        # Execute via MCP JSON-RPC
                        from gaia_core.utils.mcp_client import call_jsonrpc
                        from gaia_common.utils.tool_call_parser import META_TOOL_OPEN, META_TOOL_CLOSE, META_RESULT_OPEN, META_RESULT_CLOSE
                        _META_VERBS = {"search", "do", "learn", "remember", "ask"}
                        _is_meta = tc.tool_name in _META_VERBS

                        if tc.tool_name == "expand_context":
                            # CFR Phase 2b page-fault: resolve a BLURred turn's full
                            # text locally from this session's history (the backing
                            # store) — no MCP. Produce an rpc_result in call_jsonrpc's
                            # shape so the shared result-handling below covers it.
                            _rid = str((tc.tool_params or {}).get("id")
                                       or (tc.tool_params or {}).get("turn")
                                       or tc.tool_action or "").strip()
                            _rec = _resolve_cfr_recall(history, _rid)
                            if _rec:
                                rpc_result = {"ok": True, "response": {"result": _rec}}
                                logger.info("CFR recall: paged turn '%s' back into focus", _rid)
                            else:
                                rpc_result = {"ok": False, "error": f"No set-aside turn with id '{_rid}'."}
                                logger.info("CFR recall MISS for id '%s'", _rid)
                        else:
                            tool_params = dict(tc.tool_params or {})
                            if not _is_meta and tc.tool_action:
                                tool_params["action"] = tc.tool_action
                            rpc_result = await loop.run_in_executor(None, lambda: call_jsonrpc(
                                method=tc.tool_name,
                                params=tool_params,
                            ))
                        if rpc_result.get("ok"):
                            actual_result = rpc_result.get("response", {}).get("result", rpc_result.get("response", {}))
                            result_xml = format_tool_result(actual_result)
                        else:
                            # Normalize error payloads. The model otherwise sees
                            # the raw HTTP/JSON-RPC error verbatim and confabulates
                            # IT-helpdesk recovery flows ("report to the technical
                            # team at https://gaia-mcp:8765/web_forms/...") with
                            # fabricated URLs. Strip the noise and surface only the
                            # one actionable line.
                            error = rpc_result.get("error", "Tool call failed")
                            _err_str = str(error)
                            _norm_msg = _err_str
                            # If MCP returned "Unknown action X for domain Y.
                            # Available: [...]", lift just that sentence.
                            _idx = _err_str.find("Unknown action")
                            if _idx >= 0:
                                _norm_msg = _err_str[_idx:].split("\n")[0]
                            elif "Internal error:" in _err_str:
                                _idx2 = _err_str.find("Internal error:")
                                _norm_msg = _err_str[_idx2:].split("\n")[0]
                            result_xml = format_tool_result({
                                "ok": False,
                                "error": _norm_msg,
                                "hint": "Tool call failed. Acknowledge the failure briefly to the user and continue conversationally. Do NOT invent error-report URLs, support forms, or ticket-filing procedures. Do NOT retry with a different action unless the user asked you to."
                            })

                        # Show tool execution status to user
                        result_preview = str(actual_result)[:200] if rpc_result.get("ok") else str(error)[:200]
                        yield json.dumps({"type": "token", "value": f"\n*[{tc.tool_name}({tc.tool_action}) → {result_preview}]*\n"}) + "\n"
                        yield json.dumps({"type": "flush"}) + "\n"

                        # ── Continuation generation ──────────────────────
                        # Build messages: original prompt + model's first output
                        # (including tool_call) + tool_result, then generate again.
                        first_output = "".join(response_pieces)
                        if _is_meta:
                            # Meta-verb format: <|tool>verb(params)<tool|> ... <|tool_response>result<tool_response|>
                            _param_str = ", ".join(f'{k}="{v}"' if isinstance(v, str) else f'{k}={v}' for k, v in (tc.tool_params or {}).items())
                            _call_str = f"{META_TOOL_OPEN}{tc.tool_name}({_param_str}){META_TOOL_CLOSE}"
                            _result_str = f"{META_RESULT_OPEN}{json.dumps(actual_result, default=str)[:500]}{META_RESULT_CLOSE}"
                        else:
                            _call_str = f"{TOOL_CALL_OPEN}{json.dumps({'tool': tc.tool_name, 'action': tc.tool_action, **(tc.tool_params or {})})}{TOOL_CALL_CLOSE}"
                            _result_str = result_xml
                        # Continuation system prompt: the previous turn's
                        # full system prompt is NOT carried into this second
                        # generation. Without explicit guidance, the model
                        # confabulates fake error-report URLs, IT-helpdesk
                        # instructions, and ticket-filing flows when tools
                        # fail. This focused system message keeps the
                        # continuation grounded.
                        _cont_system = (
                            "You are GAIA. A tool call just executed and its "
                            "result is in the next user-role message. "
                            "Continue your reply to the user using that "
                            "result.\n"
                            "- If the result is success: integrate the data "
                            "naturally and answer the user's question.\n"
                            "- If the result is an error: acknowledge the "
                            "failure in one short sentence and pivot back to "
                            "the conversation. Do NOT invent error-report "
                            "URLs, web forms, support emails, ticket-filing "
                            "instructions, or technical-team contact info — "
                            "none of those exist. Don't suggest the user "
                            "report the error anywhere; the error is logged "
                            "internally already.\n"
                            "- Do not emit another <tool_call> unless the "
                            "user explicitly asked for a retry."
                        )
                        continuation_messages = [
                            {"role": "system", "content": _cont_system},
                            {"role": "user", "content": user_input},
                            {"role": "assistant", "content": first_output + f"\n{_call_str}"},
                            {"role": "user", "content": _result_str},
                        ]

                        # Generate continuation using the same model
                        try:
                            cont_gen = _agent_core.generate_continuation(
                                messages=continuation_messages,
                                session_id=session_id,
                            )
                            if cont_gen:
                                def _next_cont():
                                    try:
                                        return next(cont_gen)
                                    except StopIteration:
                                        return None

                                while True:
                                    cont_event = await loop.run_in_executor(None, _next_cont)
                                    if cont_event is None:
                                        break
                                    if isinstance(cont_event, dict) and cont_event.get("type") == "token":
                                        cont_val = cont_event.get("value", "")
                                        cont_val = _strip_think_token(cont_val)
                                        if cont_val:
                                            response_pieces.append(cont_val)
                                            yield json.dumps({"type": "token", "value": cont_val}) + "\n"
                        except Exception as cont_err:
                            logger.warning("Continuation generation failed: %s", cont_err)
                            # Fallback: just append the raw result
                            response_pieces.append(f"\n{result_xml}\n")

                    except Exception as e:
                        logger.warning("Tool call execution failed: %s", e)
                        yield json.dumps({"type": "token", "value": f"\n*[tool error: {e}]*\n"}) + "\n"

            # Ensure a flush is always emitted so front-ends send accumulated text
            if response_pieces:
                yield json.dumps({"type": "flush"}) + "\n"

            # Finalize response processing
            full_response = "".join(response_pieces)
            from gaia_core.utils.output_router import _strip_think_tags_robust
            full_response = _strip_think_tags_robust(full_response)

            # Gate 2 (worth-voicing): strip leaked meta-commentary / thinking-out-
            # loud post-generation. Measure-only unless VOICE_GATE_ENABLED — when
            # on, this cleans the packet candidate (voice + candidate consumers).
            # NOTE: the Discord path accumulates streamed tokens, not the candidate,
            # so its apply-path is a separate step; this already logs what it WOULD
            # strip on every turn (incl. Discord) for validation. See voice_gate.py.
            try:
                from gaia_core.cognition.voice_gate import filter_voiced
                _vg_out, _vg_dbg = filter_voiced(full_response)
                if _vg_dbg.get("dropped"):
                    logger.info(
                        "VoiceGate: %s %d meta sentence(s)%s | e.g. %r",
                        "stripped" if not _vg_dbg.get("measure_only") else "would strip",
                        len(_vg_dbg["dropped"]),
                        f" (failsafe={_vg_dbg['failsafe']})" if _vg_dbg.get("failsafe") else "",
                        _vg_dbg["dropped"][0]["sent"][:80])
                    full_response = _vg_out
            except Exception:
                logger.debug("VoiceGate failed (non-fatal)", exc_info=True)

            if final_packet_dict:
                # Ensure the final response in the packet is clean (no think tags)
                if "response" in final_packet_dict and "candidate" in final_packet_dict["response"]:
                    final_packet_dict["response"]["candidate"] = full_response

                # v0.4 stream integrity: include fragment metadata if fragmentation occurred
                resp_data = final_packet_dict.get("response", {})
                if resp_data.get("fragments"):
                    frag_count = len(resp_data["fragments"])
                    sequences = sorted(f.get("sequence", 0) for f in resp_data["fragments"])
                    expected = list(range(frag_count))
                    final_packet_dict["response"]["stream_integrity"] = {
                        "fragment_count": frag_count,
                        "continuous": sequences == expected,
                        "gaps": [i for i in expected if i not in sequences],
                        "total_chars": len(full_response),
                    }

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
            message = data["choices"][0]["message"]
            text = message.get("content") or ""
            # A model may respond with a tool call instead of prose (e.g. Prime
            # dispatching finance.crypto_price for "price of Bitcoin", or a
            # web_search for a real-time question). That is a valid response, not
            # an empty one — surface it so callers/evals see the model acted,
            # and guard against None content (.strip() would otherwise throw).
            if not text.strip():
                tool_calls = message.get("tool_calls") or []
                _calls = []
                for _tc in tool_calls:
                    _fn = (_tc.get("function") or {}) if isinstance(_tc, dict) else {}
                    _name = _fn.get("name", "tool")
                    _args = _fn.get("arguments", "")
                    _calls.append(f"{_name}({_args})" if _args else _name)
                if _calls:
                    text = "[tool_call] " + "; ".join(_calls)
            return {"content": text.strip(), "target": req.target,
                    "tool_call": bool(message.get("tool_calls"))}
    except Exception as e:
        logger.warning("Cognitive query error (target=%s): %s", req.target, e)
        return {"content": "", "error": str(e), "target": req.target}


@app.post("/api/voice_gate")
async def api_voice_gate(req: Request):
    """Gate 2 (worth-voicing) as a service: strip leaked meta-commentary from a
    finished response. gaia-web's Discord path renders accumulated stream tokens
    (not the cleaned packet candidate), so it calls this just before sending to
    apply the same filter that finalization applies to the candidate. Fail-open:
    on any error, returns the input unchanged so a message is never lost.
    """
    try:
        body = await req.json()
        text = body.get("text", "") or ""
        if not text.strip():
            return {"text": text, "dropped": 0}
        from gaia_core.cognition.voice_gate import filter_voiced
        out, dbg = filter_voiced(text, measure_only=False)
        return {
            "text": out,
            "dropped": len(dbg.get("dropped", [])),
            "failsafe": dbg.get("failsafe"),
            "changed": out.strip() != text.strip(),
        }
    except Exception as e:
        logger.debug("api_voice_gate failed (non-fatal): %s", e, exc_info=True)
        try:
            return {"text": (await req.json()).get("text", ""), "dropped": 0, "error": str(e)}
        except Exception:
            return {"text": "", "dropped": 0, "error": str(e)}


@app.post("/api/cognitive/stream")
async def cognitive_stream(req: CognitiveQueryRequest):
    """Streaming variant of /api/cognitive/query — yields token deltas as NDJSON
    ({"token": "..."}\\n per line) so voice/TTS callers can synthesize sentence-by-
    sentence as Core generates, dropping first-word latency. Minimal system prompt
    (no world-state/arch injection) to keep first-token fast. no_think is NOT used
    (the /no_think directive makes Gemma-4 Core return empty/truncated content)."""
    import httpx, json as _json
    from fastapi.responses import StreamingResponse

    target_endpoints = {
        "core": os.environ.get("CORE_CPU_ENDPOINT", "http://localhost:8092"),
        "prime": os.environ.get("PRIME_ENDPOINT", "http://gaia-prime:7777"),
        "nano": os.environ.get("NANO_ENDPOINT", "http://localhost:8093"),
    }
    base = target_endpoints.get(req.target, target_endpoints["core"])
    url = f"{base.rstrip('/')}/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": req.system},
            {"role": "user", "content": req.prompt},
        ],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": True,
    }

    async def gen():
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", url, json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            ev = _json.loads(data)
                            delta = (ev["choices"][0].get("delta", {}) or {}).get("content")
                        except Exception:
                            continue
                        if delta:
                            yield _json.dumps({"token": delta}) + "\n"
        except Exception as e:
            yield _json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


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


@app.post("/refresh_pool")
async def refresh_pool():
    """Remove stale GPU-tier model entries from the pool.

    Called by gaia-orchestrator after tier transitions (e.g., FOCUSING→AWAKE)
    to ensure Core's model pool doesn't route to non-existent GPU endpoints.
    """
    global _ai_manager
    if _ai_manager is None or not hasattr(_ai_manager, 'model_pool'):
        return JSONResponse(status_code=503, content={"ok": False, "error": "model pool not initialized"})
    result = _ai_manager.model_pool.refresh_pool()
    return {"ok": True, **result}


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

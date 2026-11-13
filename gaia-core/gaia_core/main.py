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

from gaia_core.models.turn_context import clear_turn, set_turn

logger = logging.getLogger("GAIA.Core.API")

# ecnc: a reply grounded on a tool's result. At the config defaults (0.7,
# 8192 tokens) Core free-wrote pages of invented backstory about a fetched
# passage; a summary needs a cooler, shorter run.
_CONT_TEMPERATURE = 0.3
_CONT_MAX_TOKENS = 1024
# A continuation that writes less than this before calling a tool again
# only announced what it was about to do; the clean-slate retry answers.
_CONT_PREAMBLE_CHARS = 120

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


def _pending_recital(reply: str) -> bool:
    """A recital's words come from the palace: nothing to second-guess."""
    return reply.lstrip().startswith("Here is **")


def _sources_for_observer(packet_dict, tool_texts: list) -> str:
    """What this turn was given to answer from: tool results and grounding."""
    parts = [t for t in tool_texts if t]
    try:
        for df in ((packet_dict or {}).get("content") or {}).get("data_fields") or []:
            if df.get("key") in ("tool_result", "web_grounding"):
                parts.append(json.dumps(df.get("value"), default=str, ensure_ascii=False)[:3000])
    except Exception:
        logger.debug("observer context failed (non-fatal)", exc_info=True)
    return "\n".join(parts)[:6000]


def _self_facts_for_observer(history: list, reply: str = "") -> str:
    """What is true of her and of this conversation: a reply can use these
    without anyone having to look them up."""
    from datetime import datetime, timezone
    # x9sw: the clock first, in UTC too as her prompt gives it. Last in line, it
    # was cut off by the cap, and a reply that read out the time was doubted.
    parts = ["GAIA, created by Azrael.", datetime.now(timezone.utc).strftime("UTC: %Y-%m-%d %H:%M")]
    try:
        from gaia_core.utils.world_state import format_world_state_snapshot
        parts.append(format_world_state_snapshot(max_lines=4))
        from gaia_common.utils.locality import locality_snapshot
        parts.append(json.dumps(locality_snapshot(), default=str))
        from gaia_common.utils.host_facts import host_facts_text
        parts.append(host_facts_text())
        from gaia_common.utils.self_model import lineage_of, minds
        # She's told where her weights came from, so saying it isn't unbacked.
        parts.extend(lineage_of(m.get("weights", "")).get("history", "") for m in minds().values())
    except Exception:
        logger.debug("observer self-facts failed (non-fatal)", exc_info=True)
    # What was just said is what a reply leans on: the latest of it is kept
    # whole, and the facts get the rest of the cap.
    # The reply under check isn't evidence for itself. It's in the history by now
    # (agent_core adds it), and so it backed every name it had (x9sw).
    said = (reply or "").strip()
    earlier = [m for m in (history or []) if isinstance(m, dict)
               and not (said and m.get("role") == "assistant" and str(m.get("content", "")).strip() in (said, said.split("\n\n", 1)[-1]))]
    talk = "\n".join(str(m.get("content", ""))[:400] for m in earlier[-6:])[-1500:]
    facts = "\n".join(p for p in parts if p)
    return (talk + "\n" + facts).strip()[:3000]


_SECOND_MIND_SECONDS = 20.0


def _packet_source(packet_data: dict) -> str:
    """Where a packet came from, as _run_loop_inner reads it: web unless it says."""
    routing = ((packet_data or {}).get("header") or {}).get("output_routing") or {}
    return routing.get("source_destination") or "web"


def _wake_for_turn(lifecycle_client, orchestrator_url, source: str) -> bool:
    """Someone spoke while she is parked: bring up the mind on the GPU before
    answering. Parked, Core answered on the CPU from its short prompt and nothing
    woke her ("capital of Australia" got "Can you check the weather forecast?",
    2026-11-04). If the wake fails the turn goes on with what is there."""
    if source not in ("web", "discord") or not lifecycle_client or not orchestrator_url:
        return False
    try:
        state = lifecycle_client.get_state_sync(timeout=3.0).state
        if getattr(state, "value", str(state)).lower() != "parked":
            return False
        import httpx
        logger.info("Main: parked and spoken to; waking first")
        return bool(httpx.post(f"{orchestrator_url}/gpu/wake", json={}, timeout=120).json().get("ok"))
    except Exception as e:
        logger.warning("Main: could not wake for this turn: %s", e)
        return False


def _second_mind_followup(agent_core, question: str, reply: str, known: str, session_id: str,
                          ran_tool: bool, had_source: bool) -> str:
    """Hand a reply to whichever mind isn't generating. Returns its follow-up,
    if it seems necessary. Runs after the turn, off the event loop."""
    from gaia_core.cognition.second_mind import review
    observer = "core" if agent_core._is_prime_available else "prime"
    model = agent_core.model_pool.models.get(observer)
    if model is None:
        return ""

    def generate(prompt: str) -> str:
        # x9sw: thinking off. With it on, Core spent all 70 tokens thinking and
        # answered "", and an empty answer backs nothing, so every name was doubted.
        out = model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}], max_tokens=70, temperature=0.0,
            chat_template_kwargs={"enable_thinking": False})
        return out["choices"][0]["message"].get("content") or ""

    return review(question, reply, known, generate, observer, session_id, ran_tool, had_source)


def _resolve_cfr_recall(history, rid, tool_ledger=None):
    """CFR Phase 2b: resolve a BLURred turn (or tool-ledger entry) by id.

    The page-fault handler for conversation-as-virtual-memory. When GAIA emits
    an expand_context tool call to page set-aside content back, we resolve it
    locally — no MCP round-trip. Two backing stores: this session's history
    (conversation turns) and the tool ledger (full bodies of recent tool/fetch
    results, id 'tl…'). Returns a compact result dict, or None if not found.
    Pure + side-effect-free so it can be unit-tested in isolation.
    """
    if not rid:
        return None
    for m in (history or []):
        if str(m.get("id")) != str(rid):
            continue
        text = (m.get("content") or "")
        try:
            from gaia_core.utils.output_router import _strip_think_tags_robust
            text = _strip_think_tags_robust(text)
        except Exception:
            pass  # strip is best-effort; never let it drop a valid match
        return {"recalled_turn_id": str(rid), "role": m.get("role", "?"), "text": text[:2000]}
    # Fall back to the tool ledger: page the full body of a recent tool result.
    for e in (tool_ledger or []):
        if str(e.get("id")) == str(rid):
            _body = e.get("body") or e.get("gist") or ""
            _prov = e.get("title") or e.get("url") or e.get("tool", "")
            return {"recalled_turn_id": str(rid), "role": "tool",
                    "text": (f"{_prov}\n{_body}" if _prov else _body)[:6000]}
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

    # h3ds: pre-warm the embed-intent classifier at startup instead of
    # leaving it to lazy-init on first use. EmbedIntentClassifier.instance()
    # is a process-wide singleton whose .ready flag only flips True inside
    # agent_core.py's run_turn() generator body — but generators don't run
    # any code until first iterated, and main.py's own tool-suppression gate
    # (further down in this file) checks `.ready` SYNCHRONOUSLY, right after
    # creating that generator, before the first token is ever pulled from
    # it. On the very first turn any freshly-started gaia-core process
    # handles, that check sees `.ready == False` and silently skips
    # suppression for that one turn — Core gets primed by prompt_builder.py
    # (which reads the intent agent_core's generator body classifies a few
    # lines into its own execution) into a tool-free "social mode" prompt,
    # but main.py's gate never got the memo and lets a hallucinated tool
    # call through to a real (doomed) MCP round-trip, leaking the raw
    # "[calling...]"/"[...-> failed]" markers. Every later turn in the same
    # process works fine because the singleton is already initialised by
    # then. Doing it here, once, before the app starts accepting traffic,
    # removes the race instead of papering over one specific turn.
    try:
        from gaia_core.cognition.nlu.embed_intent_classifier import EmbedIntentClassifier
        _embed_intent_cfg = getattr(config, 'constants', {}).get("EMBED_INTENT", {}) or {}
        if _embed_intent_cfg.get("enabled", True):
            _eic_startup = EmbedIntentClassifier.instance()
            if not _eic_startup.ready:
                _embed_model_startup = pool.get_embed_model(timeout=0, lazy_load=True)
                if _embed_model_startup is not None:
                    _eic_startup.initialise(_embed_model_startup, config=_embed_intent_cfg)
                    logger.info("EmbedIntentClassifier pre-warmed at startup (ready=%s)", _eic_startup.ready)
                else:
                    logger.warning("EmbedIntentClassifier pre-warm skipped: embed model unavailable at startup")
    except Exception:
        logger.warning("EmbedIntentClassifier pre-warm failed (non-fatal, will lazy-init on first turn)", exc_info=True)

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

            # Synchronize initial SleepWakeManager state with orchestrator lifecycle
            if app.state.lifecycle_client and app.state.sleep_wake_manager:
                try:
                    logger.info("Synchronizing initial sleep-wake state with orchestrator...")
                    snapshot = app.state.lifecycle_client.get_state_sync(timeout=3.0)
                    from gaia_core.cognition.sleep_wake_manager import GaiaState
                    state_val = snapshot.state
                    logger.info("Orchestrator lifecycle state on boot: %s", state_val)
                    if state_val in ("sleep", "deep_sleep", "parked"):
                        app.state.sleep_wake_manager.state = GaiaState.ASLEEP
                        logger.info("Synced local SleepWakeManager to ASLEEP")
                    elif state_val == "meditation":
                        app.state.sleep_wake_manager.state = GaiaState.DREAMING
                        logger.info("Synced local SleepWakeManager to DREAMING")
                    elif state_val in ("awake", "listening", "focusing"):
                        app.state.sleep_wake_manager.state = GaiaState.ACTIVE
                        logger.info("Synced local SleepWakeManager to ACTIVE")
                except Exception as sync_exc:
                    logger.warning("Failed to sync initial sleep-wake state with orchestrator: %s", sync_exc)

            _sleep_loop.start()
            logger.info("Sleep cycle loop started")
        else:
            logger.info("Sleep cycle disabled (SLEEP_ENABLED=False)")
    except Exception:
        logger.warning("Failed to start sleep cycle loop", exc_info=True)

    # Start idle heartbeat — the "sound of silence"
    _idle_heartbeat = None
    if getattr(config, "HEARTBEAT_ENABLED", True):
        try:
            from gaia_core.cognition.idle_heartbeat import IdleHeartbeat
            _idle_heartbeat = IdleHeartbeat(
                config=config,
                model_pool=pool,
                timeline_store=getattr(app.state, "timeline_store", None),
                session_manager=getattr(_agent_core, "session_manager", None) if _agent_core else None,
                sleep_wake_manager=getattr(app.state, "sleep_wake_manager", None),
            )
            _idle_heartbeat.start()
            app.state.idle_heartbeat = _idle_heartbeat
            logger.info("Idle heartbeat started")
        except Exception:
            logger.warning("Failed to start idle heartbeat", exc_info=True)
    else:
        logger.info("Idle heartbeat disabled (HEARTBEAT_ENABLED=False)")

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
    if getattr(config, "HEARTBEAT_ENABLED", True):
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
    else:
        logger.info("Audio commentary daemon disabled (HEARTBEAT_ENABLED=False)")

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
    results: Dict[str, Dict[str, Any]] = {}

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
        # 02m5: a restart is not sleep. Left unconsumed, this "preparing for a
        # sleep cycle" note went into every non-trivial prompt until a real
        # wake, which a plain restart never has (Prime dropped URLs under it).
        # Here, not after the writes: Core.md can outlast the stop grace period.
        pm.mark_consumed()
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
            else:
                try:
                    _engine_health = resp.json()
                except Exception:
                    _engine_health = {}
                # 29yq: the engine's drain flag 503s ALL inference while
                # /health stays 200 — a stale drain (never resumed) left the
                # pipeline dead-but-"healthy" for days and the doctor never
                # remediated. A drain is legitimate for the seconds of a gear
                # shift; the doctor's failure threshold + confirmation
                # re-check absorb that. One that persists across polls is
                # stale and MUST surface as degraded.
                if _engine_health.get("draining"):
                    inference_ok = False
                    inference_detail = "engine draining (503s all inference; stale if persistent)"
                # t0f2: the embedded engine's own /health now probes CUDA
                # when it expects to be on GPU (see gaia-engine manager.py
                # health_response()) — a stale driver context after a host
                # reboot used to report "healthy" right up until a real
                # request crashed. Surface that here the same way draining
                # is surfaced, so Doctor's existing degraded-status
                # remediation picks it up with no Doctor-side changes.
                if _engine_health.get("cuda_healthy") is False:
                    inference_ok = False
                    inference_detail = f"embedded engine: {_engine_health.get('cuda_detail', 'CUDA unhealthy')}"
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

    # Observer (conscience) health — A3. Surfaces a silently-degraded gate-2
    # (the observer soft-fails to OK) so Doctor, which polls /health, can see it.
    # A failing conscience degrades the service even if inference is fine.
    try:
        from gaia_core.utils.stream_observer import observer_health
        _obs = observer_health()
        if _obs and not _obs.get("healthy", True):
            if status == "healthy":
                status = "degraded"
            inference_detail = (inference_detail or "ok") + " | observer degraded (conscience failing)"
    except Exception:
        _obs = None

    return JSONResponse(
        status_code=200,
        content={
            "status": status,
            "service": "gaia-core",
            "inference_ok": inference_ok,
            "inference_detail": inference_detail or "ok",
            "observer": _obs,
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


@app.post("/api/sessions/{session_id}/posted")
async def record_post(request: Request, session_id: str):
    """Something she posted on her own (discord_post), kept in that channel's
    history so she knows what she said there when someone answers it. Not an
    exchange: nobody's message prompted it."""
    if _ai_manager is None:
        return JSONResponse(status_code=503, content={"error": "Cognitive system not initialized"})
    data = await request.json()
    text = (data.get("text") or "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "text is required"})
    _ai_manager.session_manager.add_message(
        session_id, "assistant", text, meta={"unprompted": True, "via": data.get("via") or "post"})
    return {"ok": True, "session_id": session_id}


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
async def delete_session(session_id: str, archive: bool = True):
    """Archive and remove a session. archive=false skips the summary, for
    throwaway sessions (test probes) that shouldn't enter long-term memory."""
    if _ai_manager is None:
        return JSONResponse(status_code=503, content={"error": "Cognitive system not initialized"})

    sm = _ai_manager.session_manager
    if session_id not in sm.sessions:
        return JSONResponse(status_code=404, content={"error": f"Session '{session_id}' not found"})

    session = sm.sessions[session_id]

    # Archive before deletion if there is history
    if archive and session.history:
        try:
            # 74t8: this runs LLM calls. On the event loop it starved /health;
            # ten deletes in a row got gaia-core restarted by the doctor.
            await asyncio.to_thread(sm.summarize_and_archive, session_id)
        except Exception as e:
            logger.warning("Pre-delete archive failed for %s: %s", session_id, e)

    # Remove from active sessions
    sm.reset_session(session_id)

    # olqt: reset_session() only clears the active-history record — the RAG
    # vector-store file survives on its own, so old conversation content
    # stays retrievable even after a "delete". Purge it too, and evict any
    # cached in-memory indexer instance (otherwise a live cached object would
    # just get resaved on the next turn, silently resurrecting what we just
    # deleted).
    try:
        from gaia_core.memory.session_history_indexer import SessionHistoryIndexer, _DEFAULT_PERSIST_DIR
        SessionHistoryIndexer._instances.pop(session_id, None)
        vec_path = os.path.join(_DEFAULT_PERSIST_DIR, f"{session_id}.json")
        if os.path.exists(vec_path):
            os.remove(vec_path)
    except Exception:
        logger.warning("delete_session: vector purge failed for %s", session_id, exc_info=True)

    return {"status": "deleted", "session_id": session_id}


@app.post("/process_packet")
async def process_packet(packet_data: Dict[str, Any]):
    """
    Process a CognitionPacket through the cognitive loop with streaming support.
    Yields chunks of data: tokens as they are generated, and finally the completed packet.
    """
    global _agent_core, _ai_manager

    # GAIA's Grotto (2026-11-12): an ambient-channel turn (addressed_to_gaia
    # False) she wasn't addressed in shouldn't reset the sleep idle timer
    # just by arriving -- a chatty channel would otherwise keep her "active"
    # forever and starve the self-repair loop's sleep-only tasks. Marked
    # active here only when addressed; the ambient case is marked later,
    # after the turn, and only if she actually decided to reply.
    try:
        _addressed = bool(packet_data.get("header", {}).get("output_routing", {})
                          .get("addressed_to_gaia", True))
    except Exception:
        _addressed = True

    idle_monitor = getattr(app.state, "idle_monitor", None)
    if idle_monitor and _addressed:
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

    _after_turn: dict = {}

    async def _run_loop():
        logger.info("Main: _run_loop generator started")
        async with _turn_semaphore:
            logger.info("Main: acquired turn semaphore")
            if await asyncio.to_thread(_wake_for_turn, getattr(app.state, "lifecycle_client", None),
                                       os.environ.get("ORCHESTRATOR_ENDPOINT", "http://gaia-orchestrator:6410"),
                                       _packet_source(packet_data)):
                yield json.dumps({"type": "status", "value": "awake"}) + "\n"
            async for chunk in _run_loop_inner():
                yield chunk
        # b9ts: her reply is out and the turn is free, so the next message never
        # waits on this. The other mind adds a follow-up if it seems necessary.
        args = _after_turn.get("second_mind")
        if args:
            # Her reply is complete. The stream stays open while the other mind
            # reads it (up to _SECOND_MIND_SECONDS), and the web chat held its
            # input locked until then: 5 s to her answer, 26 s to the next
            # message (2026-11-11). Clients may stop waiting here.
            yield json.dumps({"type": "reply_done"}) + "\n"
            try:
                said = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(None, _second_mind_followup, *args),
                    timeout=_SECOND_MIND_SECONDS)
            except Exception as e:
                logger.info("Second mind: no follow-up (%s)", type(e).__name__)
                said = ""
            if said:
                text = f"*[Second mind]* {said}"
                yield json.dumps({"type": "token", "value": text}) + "\n"
                yield json.dumps({"type": "flush"}) + "\n"
                try:
                    args[0].session_manager.add_message(args[4], "assistant", text)
                except Exception:
                    logger.debug("Second mind: follow-up not kept in history", exc_info=True)

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
                        "author_name": (routing.primary.metadata or {}).get("author_name"),
                        "reply_to_message_id": routing.primary.reply_to_message_id,
                        "is_dm": routing.primary.metadata.get("is_dm", False) if routing.primary.metadata else False,
                        # GAIA's Grotto (2026-11-12): False for an ambient-channel
                        # turn she wasn't addressed in — run_turn decides whether
                        # to reply at all; _build_output_routing reads this same
                        # key back out for the (possibly silent) reply's routing.
                        "addressed_to_gaia": routing.addressed_to_gaia,
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
            if user_input:
                _path_match_main = _re_blast_main.search(
                    r"(?:^|\s)((?:/tmp|/etc|/root|/boot|/var|/proc|/sys|/dev|/run|/mnt|/media|/srv)/"
                    r"[A-Za-z0-9_./\-]+)",
                    user_input,
                )
                if _path_match_main:
                    _unreachable = _path_match_main.group(1)
                    logger.info(
                        "[ALLOW-LIST HEDGE] user mentioned unreachable path %r — appending hedge to user_input",
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

            # vjdf: the speculative-reflex pre-flight (nano tier) is removed —
            # every turn goes through run_turn, so session persistence, the
            # Observer, and CFR all apply uniformly.
            history = _ai_manager.session_manager.get_history(session_id)

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
                    ToolCallParser, ParseEventType, format_tool_result, tool_result_text, tool_answer_messages,
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
                attachments=_inbound_attachments,
            )

            _turn_id = getattr(packet.header, "packet_id", "") or ""

            def _next_event():
                set_turn(session_id, _turn_id)
                try:
                    return next(gen)
                except StopIteration:
                    return None
                finally:
                    clear_turn()

            from gaia_core.cognition.reply_scrub import answered_already, hands_back
            _pending_tool_calls = []
            _seen_tool_calls = set()  # Dedup repeated tool calls
            _tool_texts_for_observer: list = []

            # Suppress tool calls on tool-free conversational intents — Core
            # habitually emits e.g. worldstate() even on a greeting, which adds
            # "[calling...]" noise and fragments the reply. Greetings/chitchat
            # never need a tool, so drop such calls silently (the surrounding
            # reply text still streams).
            _suppress_tools = False
            _plain_chat = False
            # 7o99: counts calls dropped below so a turn whose ENTIRE output
            # was suppressed tool-call markup (no surrounding prose at all)
            # can get a graceful fallback instead of dead silence — see the
            # empty-response_pieces check near the post-loop flush.
            _suppressed_tool_count = 0
            try:
                from gaia_core.cognition.nlu.embed_intent_classifier import EmbedIntentClassifier
                _eic = EmbedIntentClassifier.instance()
                if _eic.ready:
                    # h3ds: this MUST use the same confidence_threshold as
                    # agent_core.py's early-intent estimate (which is what
                    # prompt_builder.py's packet.intent.user_intent — and
                    # therefore the prompt's tool-free/social-mode framing —
                    # is actually built from). EmbedIntentClassifier.classify()
                    # is a deterministic function of (text, threshold); calling
                    # it here with the classifier's own hardcoded default
                    # (0.45) instead of agent_core's configured threshold
                    # (0.42 by default) let this independent reclassification
                    # silently disagree with the prompt that was already built
                    # — Core got primed for a tool-free turn but this gate's
                    # stricter threshold let the resulting tool call through
                    # anyway, so it hit a real (doomed) MCP round-trip and
                    # leaked the raw "[calling...]"/"[...-> failed]" markers.
                    from gaia_core.config import get_config as _get_config_h3ds
                    _embed_intent_cfg = getattr(_get_config_h3ds(), 'constants', {}).get("EMBED_INTENT", {}) or {}
                    _suppress_thr = float(_embed_intent_cfg.get("confidence_threshold", 0.42))
                    _intent_lbl, _ = _eic.classify(user_input or "", confidence_threshold=_suppress_thr)
                    _plain_chat = _intent_lbl == "chat" and "://" not in (user_input or "")
                    # 'time' is world-state-answerable: the clock is already
                    # injected in the prompt, so drop the redundant (noisy)
                    # worldstate tool call and let Core answer from context.
                    # Shared with deliberation, which leaves the tool schema
                    # out for the same intents (1ha9).
                    from gaia_core.utils.prompt_builder import TOOL_FREE_INTENTS
                    _suppress_tools = _intent_lbl in TOOL_FREE_INTENTS
                    if _suppress_tools:
                        logger.info("Chat: tool-free intent '%s' — suppressing tool calls", _intent_lbl)
            except Exception:
                logger.debug("intent pre-check for tool suppression failed (non-fatal)", exc_info=True)

            # nxxl follow-up: Core (and Prime) sometimes hallucinate an
            # entire nonexistent tool domain on creative-writing requests
            # (poet, poetry, amusement, worldbuild/poe, ...) — a
            # well-formed call to a tool that was never real, not the
            # malformed-syntax case nz4w already covers. The packet's
            # available_mcp_tools is the actual whitelist for this turn;
            # anything outside it (plus the meta-verb aliases and the
            # local expand_context pseudo-tool) is hallucinated. Flag
            # these at detection time so we can skip the doomed MCP
            # round-trip AND the "*[calling ...]*" / "*[... -> failed]*"
            # plumbing markers — that internal-mechanics text is
            # meaningless noise for a tool that was never real, and it's
            # exactly what showed up as a bare "failed tool call" with no
            # actual poem (live repro, 2026-09-02).
            _META_VERBS = {"search", "do", "learn", "remember", "ask"}
            _tool_allowlist_active = bool(getattr(packet.context, "available_mcp_tools", None))
            _valid_tool_names = (
                {n.lower() for n in (packet.context.available_mcp_tools or [])}
                | _META_VERBS | {"expand_context"}
            )
            _invalid_tool_ids: set = set()

            # Degenerate-repetition guard (live repro, 2026-09-02): Core
            # occasionally never closes a single <tool_call>/<tool_calling:>
            # tag and instead spins for the entire token budget emitting
            # dozens of near-duplicate malformed attempts back to back. Per
            # 415r's deliberate design, an unclosed tag at flush() is
            # surfaced as TEXT rather than dropped — correct for the case
            # that design targets (a genuine reply cut off mid-tag, exactly
            # ONE occurrence). But a flush buffer containing MULTIPLE
            # open-tag markers is not a truncated reply, it's a repetition
            # collapse — showing that raw to the user leaks the same
            # internal tool-call plumbing nz4w/pfdw already fixed for the
            # single-occurrence case. Two-plus occurrences is the signal.
            from gaia_common.utils.tool_call_parser import TOOL_CALL_OPEN as _TCO, TOOL_CALLING_OPEN_PREFIX as _TCLO, META_TOOL_OPEN as _MTO

            def _is_repetition_garbage(text: str) -> bool:
                return (text.count(_TCO) + text.count(_TCLO) + text.count(_MTO)) >= 2

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
                                        _suppressed_tool_count += 1
                                        continue
                                    # Plain chat she has already answered needs no tool (8ail).
                                    # A reply that hands the turn back needs none, whatever the intent (z838).
                                    _so_far = "".join(response_pieces)
                                    if pe.tool_name != "expand_context" and (
                                            (_plain_chat and answered_already(_so_far)) or hands_back(_so_far)):
                                        logger.info("Tool call '%s' dropped: already answered", pe.tool_name)
                                        _suppressed_tool_count += 1
                                        continue
                                    # 02m5: she named the tool; the person named the page.
                                    from gaia_common.utils.tool_call_parser import fill_missing_url
                                    pe.tool_params = fill_missing_url(
                                        pe.tool_name, pe.tool_action, pe.tool_params, user_input)
                                    # Dedup: only process each unique tool call once
                                    _tc_key = f"{pe.tool_name}:{pe.tool_action}:{json.dumps(pe.tool_params, sort_keys=True)}"
                                    if _tc_key in _seen_tool_calls:
                                        logger.debug("Duplicate tool call skipped: %s", _tc_key[:80])
                                        continue
                                    _seen_tool_calls.add(_tc_key)

                                    if _tool_allowlist_active and pe.tool_name.lower() not in _valid_tool_names:
                                        logger.warning(
                                            "Hallucinated tool call (not in available_mcp_tools): %s(%s) — "
                                            "skipping MCP round-trip, no plumbing shown to user",
                                            pe.tool_name, pe.tool_action,
                                        )
                                        _invalid_tool_ids.add(id(pe))
                                        # vp52: a hallucinated tool call is a real demand
                                        # signal for a missing capability, not just noise
                                        # to suppress — feed it into the existing
                                        # thought_seed -> CodeMind capability_gap bridge
                                        # (sleep_task_scheduler._run_initiative_cycle /
                                        # _run_codemind_cycle) instead of only correcting
                                        # it. Back off like the other seed planters
                                        # (knowledge_ingestion.py, docs_maintenance.py)
                                        # when the backlog is already at cap.
                                        try:
                                            from gaia_core.cognition.thought_seed import (
                                                save_thought_seed,
                                                seed_backlog_count,
                                            )
                                            _seed_cap = int(os.environ.get("THOUGHT_SEED_MAX_PENDING", "2000"))
                                            if seed_backlog_count() < _seed_cap:
                                                from datetime import datetime, timezone
                                                _gap_payload = {
                                                    "attempted_tool_name": pe.tool_name,
                                                    "attempted_tool_action": pe.tool_action,
                                                    "attempted_tool_params": pe.tool_params,
                                                    "user_request": user_input,
                                                    "session_id": session_id,
                                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                                }
                                                _seed_text = (
                                                    "THOUGHT_SEED: capability gap — hallucinated tool call. "
                                                    f"No tool named '{pe.tool_name}' exists; this could be "
                                                    "researched and built as a real capability (knowledge gap). "
                                                    f"Payload: {json.dumps(_gap_payload, default=str)}"
                                                )
                                                save_thought_seed(_seed_text, packet, _agent_core.config)
                                        except Exception:
                                            logger.debug(
                                                "Failed to plant capability_gap thought seed for hallucinated tool '%s'",
                                                pe.tool_name, exc_info=True,
                                            )
                                    else:
                                        # Status is its own event, never reply text (8ail):
                                        # it went to the chat, to history-less voice, everywhere.
                                        yield json.dumps({"type": "tool_status", "tool": pe.tool_name, "action": pe.tool_action, "state": "calling"}) + "\n"
                                        yield json.dumps({"type": "flush"}) + "\n"
                                    _pending_tool_calls.append(pe)

                                    # Stop generation after first tool call — execute it,
                                    # then continue generation with the result
                                    break
                                elif pe.type == ParseEventType.TOOL_ERROR:
                                    # Malformed/unparseable tool-call syntax (bad
                                    # JSON, empty tool name, etc). pe.text holds
                                    # the reconstructed raw tag — do NOT show that
                                    # to the user (nz4w): it's parser internals,
                                    # not conversation. Same treatment as the
                                    # continuation path (~line 1689) already gives
                                    # this event type. Suppress; the model's
                                    # surrounding prose (if any) still streams.
                                    logger.warning(
                                        "Suppressed malformed tool-call syntax from user-visible stream: %s",
                                        pe.error or pe.text[:100],
                                    )

                        # If we just detected a tool call, break the generation loop
                        # to execute it immediately
                        if _pending_tool_calls and _seen_tool_calls:
                            break
                        # else: parser already yielded the text — don't re-emit

                        elif not _tc_enabled or not _tc_parser:
                            # No parser — pass through raw
                            response_pieces.append(val)
                            yield json.dumps({"type": "token", "value": val}) + "\n"

                    elif event.get("type") == "inner":
                        # Her inner voice (inner_voice.py): its own message, shown
                        # but not part of the reply that is checked and kept.
                        if event.get("value"):
                            yield json.dumps({"type": "token", "value": event["value"] + "\n\n"}) + "\n"
                            yield json.dumps({"type": "flush"}) + "\n"
                    elif event.get("type") == "flush":
                        yield json.dumps(event) + "\n"
                    elif event.get("type") == "packet":
                        final_packet_dict = event.get("value")
                    else:
                        # Handle any other dictionary types (e.g. self-improvement progress)
                        # by dumping to JSON so StreamingResponse can encode it.
                        logger.debug(f"Yielding unknown event type: {event.get('type', 'unknown')}")
                        yield json.dumps(event) + "\n"

            # Flush any remaining parser buffer — unless generation was cut
            # off for a tool call. Then the buffer is the tail of output the
            # model kept writing past the call (a made-up <tool_result>, the
            # call again), and flushing it leaked 'l"}</tool_call>' (r4p6).
            if _pending_tool_calls and _tc_parser:
                try:
                    _discarded = "".join(pe.text or "" for pe in _tc_parser.flush())
                    if _discarded.strip():
                        logger.info(
                            "Discarded %d chars the model wrote past its tool call: %r",
                            len(_discarded), _discarded[:120],
                        )
                except Exception:
                    # Dropping the tail is best-effort; never fail the turn over it.
                    logger.debug("Tool-call tail discard failed", exc_info=True)
            elif _tc_enabled and _tc_parser:
                for pe in _tc_parser.flush():
                    if pe.type == ParseEventType.TEXT and pe.text:
                        if _is_repetition_garbage(pe.text):
                            logger.warning(
                                "Suppressed degenerate repetition-loop garbage from flush "
                                "(%d tool-call markers in one buffer): %s",
                                pe.text.count(_TCO) + pe.text.count(_TCLO) + pe.text.count(_MTO),
                                pe.text[:150],
                            )
                            _apology = "I got stuck trying to write that — mind asking me again in a moment?"
                            response_pieces.append(_apology)
                            yield json.dumps({"type": "token", "value": _apology}) + "\n"
                        else:
                            response_pieces.append(pe.text)
                            yield json.dumps({"type": "token", "value": pe.text}) + "\n"

            # ── Execute pending tool calls and generate continuation ───
            if _pending_tool_calls:
                for tc in _pending_tool_calls:
                    _is_invalid_tool = id(tc) in _invalid_tool_ids
                    _tc_t0 = _time.time()
                    try:
                        # Execute via MCP JSON-RPC
                        from gaia_core.utils.mcp_client import call_jsonrpc
                        _META_VERBS = {"search", "do", "learn", "remember", "ask"}
                        _is_meta = tc.tool_name in _META_VERBS

                        if _is_invalid_tool:
                            # Flagged as hallucinated at detection time — skip the
                            # doomed MCP round-trip entirely (gaia-mcp would just
                            # return the same "not a valid, implemented tool" error).
                            rpc_result = {
                                "ok": False,
                                "error": f"Tool '{tc.tool_name}' is not a valid, implemented tool.",
                            }
                        elif tc.tool_name == "expand_context":
                            # CFR Phase 2b page-fault: resolve a BLURred turn's full
                            # text locally from this session's history (the backing
                            # store) — no MCP. Produce an rpc_result in call_jsonrpc's
                            # shape so the shared result-handling below covers it.
                            _rid = str((tc.tool_params or {}).get("id")
                                       or (tc.tool_params or {}).get("turn")
                                       or tc.tool_action or "").strip()
                            try:
                                _tl = _agent_core.session_manager.get_tool_ledger(session_id)
                            except Exception:
                                _tl = []
                            _rec = _resolve_cfr_recall(history, _rid, tool_ledger=_tl)
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
                            if tc.tool_name == "palace":
                                # She wrote palace(recalling) with nothing to look
                                # for, it was refused, and she said "I do recall that
                                # conversation" of a chat the palace held (2026-11-11).
                                from gaia_core.cognition.reply_scrub import palace_recall_params
                                tool_params = palace_recall_params(tool_params, user_input or "")
                            # Who's talking comes from the transport, whatever she wrote.
                            if tc.tool_name in ("knowledge", "memory", "palace"):
                                from gaia_core.cognition.told_facts import speaker
                                _who = speaker(source, session_id, metadata)
                                tool_params["_asker"] = _who["id"]
                                tool_params["_told_by"] = {**_who, "why": "she chose to keep it"}
                                tool_params.pop("_by", None)
                            rpc_result = await loop.run_in_executor(None, lambda: call_jsonrpc(
                                method=tc.tool_name,
                                params=tool_params,
                            ))
                        # The allowlist above only works when the packet carries
                        # one (Discord's don't); MCP saying the tool doesn't exist
                        # is the same finding, and her reply then goes out as she
                        # wrote it, not as "converse(close) failed" (2026-11-11).
                        import re as _re_nf
                        if not rpc_result.get("ok") and _re_nf.search(
                                r"not a valid, implemented tool|Tool '[^']*' not found",
                                str(rpc_result.get("error") or "")):
                            _is_invalid_tool = True
                        # Outcome ledger (gjuh.2): what the model's own call did.
                        from gaia_core.cognition import outcome_ledger as _ol
                        _ol.record(
                            # agent_core's id for this turn, not the inbound packet's
                            "tool_execution", session_id,
                            _ol.current_packet(session_id) or getattr(packet.header, "packet_id", ""),
                            path="inline", tool=tc.tool_name, action=tc.tool_action or "",
                            request=_ol.clip(user_input or ""),
                            param_keys=_ol.param_keys(tc.tool_params),
                            ok=bool(rpc_result.get("ok")), hallucinated=_is_invalid_tool,
                            error="" if rpc_result.get("ok") else _ol.clip(rpc_result.get("error")),
                            empty=bool(rpc_result.get("ok")) and not (rpc_result.get("response") or {}).get("result", rpc_result.get("response")),
                            latency_ms=round((_time.time() - _tc_t0) * 1000, 1))
                        # Parked behind the human gate (2026-11-10, Discord): the
                        # model used to get "requires approval. Challenge X" back,
                        # narrate it ("I just ran it — challenge DCKXL"), and the
                        # person's reversed code then arrived as chat, which she
                        # answered by inventing a command. Stop here instead: say
                        # what needs approval, and put the pending action on the
                        # packet so gaia-web's Discord bridge shows its approval
                        # card and takes the reversed code straight to MCP.
                        if rpc_result.get("pending_approval"):
                            _shown = {k: v for k, v in (tc.tool_params or {}).items() if not str(k).startswith("_")}
                            _label = tc.tool_name + (f"({tc.tool_action})" if tc.tool_action else "")
                            _appr_msg = (f"That needs your approval first: `{_label}` {json.dumps(_shown)}. "
                                         f"Challenge `{rpc_result.get('challenge')}`; reply with it reversed "
                                         f"to approve, or say cancel.")
                            response_pieces.append(_appr_msg)
                            yield json.dumps({"type": "token", "value": _appr_msg}) + "\n"
                            try:
                                from gaia_core.cognition import approval_reply
                                approval_reply.remember(_agent_core.session_manager, session_id,
                                                        rpc_result.get("action_id"), rpc_result.get("challenge"), _label)
                            except Exception:
                                logger.warning("Could not remember the pending approval", exc_info=True)
                            try:
                                from gaia_common.protocols.cognition_packet import (
                                    ToolExecutionResult as _TER, ToolRoutingState as _TRS)
                                if packet.tool_routing is None:
                                    packet.tool_routing = _TRS()
                                packet.tool_routing.execution_result = _TER(success=True, output={
                                    "pending_approval": True,
                                    "action_id": rpc_result.get("action_id"),
                                    "challenge": rpc_result.get("challenge"),
                                    "proposal": rpc_result.get("proposal", ""),
                                    "tool_name": _label,
                                    "message": _appr_msg,
                                })
                            except Exception:
                                logger.warning("Could not attach the pending approval to the packet", exc_info=True)
                            break
                        if rpc_result.get("ok"):
                            actual_result = rpc_result.get("response", {}).get("result", rpc_result.get("response", {}))
                            result_xml = format_tool_result(actual_result)
                            # 231 Phase 1: record tool-result provenance in the
                            # session ledger so a later "what's its name/the link?"
                            # can ground on it (skip the expand_context recall).
                            if tc.tool_name != "expand_context":
                                try:
                                    _title = _url = _src = _gist = ""
                                    _ar = actual_result
                                    if isinstance(_ar, dict):
                                        _rows = _ar.get("results")
                                        if isinstance(_rows, list) and _rows and isinstance(_rows[0], dict):
                                            _title = _rows[0].get("title") or _rows[0].get("name") or ""
                                            _url = _rows[0].get("url") or _rows[0].get("link") or ""
                                            _gist = " | ".join(
                                                (r.get("title") or r.get("snippet") or "")
                                                for r in _rows[:3] if isinstance(r, dict))
                                        else:
                                            _title = _ar.get("title") or _ar.get("name") or ""
                                            _url = _ar.get("url") or _ar.get("link") or ""
                                            _gist = str(_ar.get("content") or _ar.get("text")
                                                        or _ar.get("summary") or "")
                                    else:
                                        _gist = str(_ar)
                                    _agent_core.session_manager.record_tool_result(
                                        session_id, tool=tc.tool_name, action=tc.tool_action or "",
                                        title=_title, url=_url, source=_src, gist=_gist,
                                        body=_gist if len(_gist) > 300 else "")
                                except Exception:
                                    logger.debug("tool-ledger record failed (non-fatal)", exc_info=True)
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
                            if _is_invalid_tool:
                                # Sharper, targeted correction (nxxl follow-up):
                                # the generic "tool call failed, acknowledge and
                                # move on" hint wasn't enough to stop a second
                                # hallucinated attempt in production (2026-09-02
                                # live repro) — name the false belief directly
                                # instead of a vague failure acknowledgment.
                                _hint = (
                                    f"There is no tool named '{tc.tool_name}' — you do not have that "
                                    "capability and never did. Do not attempt any other tool call for "
                                    "this request. Write the content the user asked for yourself, "
                                    "directly, in plain text, right now."
                                )
                            else:
                                _hint = "Tool call failed. Acknowledge the failure briefly to the user and continue conversationally. Do NOT invent error-report URLs, support forms, or ticket-filing procedures. Do NOT retry with a different action unless the user asked you to."
                            result_xml = format_tool_result({
                                "ok": False,
                                "error": _norm_msg,
                                "hint": _hint,
                            })
                            # bz6f / 6839: what the call returned stays in front of
                            # her next turn, failures included. Only successes were
                            # kept; a failed call then had only her reply about it
                            # ("I filled it, it should now be recordable") to go on.
                            if not _is_invalid_tool and tc.tool_name != "expand_context":
                                try:
                                    _agent_core.session_manager.record_tool_result(
                                        session_id, tool=tc.tool_name, action=tc.tool_action or "",
                                        title=f"{tc.tool_name}({tc.tool_action or ''}) call failed",
                                        gist=str(_norm_msg))
                                except Exception:
                                    logger.debug("tool-ledger record of a failure failed (non-fatal)", exc_info=True)

                        # Show tool execution status to user. On failure, never show
                        # error detail here regardless of shape — _norm_msg is a
                        # best-effort strip of two known MCP error patterns for the
                        # MODEL's context (result_xml above), not a guarantee it's
                        # clean of internal HTTP status text/service URLs for every
                        # failure shape (9t27 fixed the known-pattern case; this
                        # closes the gap for anything _norm_msg's pattern matching
                        # doesn't recognize — e.g. a nonexistent tool domain).
                        if not _is_invalid_tool:
                            # No status for a tool that never existed (nxxl
                            # follow-up). The result itself is for the reply
                            # below, not for showing (8ail).
                            yield json.dumps({"type": "tool_status", "tool": tc.tool_name, "action": tc.tool_action,
                                              "state": "ok" if rpc_result.get("ok") else "failed"}) + "\n"
                        yield json.dumps({"type": "flush"}) + "\n"

                        # Affect appraisal (P0): competence drive ← tool outcome
                        # (a real MCP tool, not the expand_context recall). Failure
                        # raises the "get it right" tension; success eases it.
                        if tc.tool_name != "expand_context":
                            try:
                                from gaia_core.cognition.affect_appraiser import note_task_outcome
                                note_task_outcome(bool(rpc_result.get("ok")), label=tc.tool_name)
                            except Exception:
                                pass

                        # ── Continuation generation ──────────────────────
                        # Build messages: original prompt + model's first output
                        # (including tool_call) + tool_result, then generate again.
                        # ecnc: the continuation asks the question again with
                        # what the tool returned as plain text — the framing
                        # the clean-slate retry below already used, which
                        # grounded far better. Replaying her own <tool_call>
                        # turn plus an XML/JSON <tool_result> gave Core a tool
                        # protocol to imitate: it re-emitted the call on most
                        # runs (so the retry ran nearly every time) and
                        # summarised the page loosely ("a fable, folks").
                        if rpc_result.get("ok"):
                            _tool_text = tool_result_text(actual_result, question=user_input or "")
                        else:
                            import re as _re
                            _tool_text = _re.sub(r"</?tool_result>", "", result_xml).strip()
                        _tool_label = f"{tc.tool_name}({tc.tool_action})" if tc.tool_action else tc.tool_name
                        # 0kv2: with its name, so "the Discord API rejected
                        # that" is backed by the discord call it describes.
                        _tool_texts_for_observer.append(f"{_tool_label} ({tc.tool_name}): {_tool_text}")
                        # Continuation system prompt: the previous turn's
                        # full system prompt is NOT carried into this second
                        # generation. Without explicit guidance, the model
                        # confabulates fake error-report URLs, IT-helpdesk
                        # instructions, and ticket-filing flows when tools
                        # fail. This focused system message keeps the
                        # continuation grounded.
                        # The continuation's system + user turn, shared with
                        # prompt_builder's routed-tool answer (gjuh.7). The
                        # previous turn's full system prompt is NOT carried:
                        # without focused guidance Core confabulated
                        # IT-helpdesk flows on failures, and with the whole
                        # prompt it re-called the tool. Result first, then the
                        # request as already done (A/B 2026-10-28: named
                        # Moby-Dick 2/4, summarised the page correctly
                        # otherwise).
                        continuation_messages = tool_answer_messages(user_input, _tool_label, _tool_text)
                        # The clean-slate retry (only when the continuation
                        # called a tool again) quotes the request up front and
                        # ends on a plain reply cue — no request left to act
                        # on: 4/5 answered where the framing above re-called
                        # palace(recall) 3/4.
                        _retry_user_turn = (
                            f'The user asked: "{user_input}"\n\nYou already did that: '
                            f"{_tool_label} returned the following.\n\n{_tool_text}\n\n"
                            f"Your reply to them, from what it returned:"
                        )

                        # Generate continuation using the same model
                        try:
                            cont_gen = _agent_core.generate_continuation(
                                messages=continuation_messages,
                                session_id=session_id,
                                max_tokens=_CONT_MAX_TOKENS,
                                temperature=_CONT_TEMPERATURE,
                            )
                            if cont_gen and not rpc_result.get("ok") and not _is_invalid_tool:
                                # An invented tool (converse(close), 2026-11-11) ran nothing;
                                # her words go out as she wrote them, not as a failure notice.
                                from gaia_core.cognition.reply_scrub import honest_about_failure
                                cont_gen = honest_about_failure(
                                    cont_gen, _tool_label,
                                    memory=tc.tool_name in ("palace", "knowledge", "memory", "context"))
                            if cont_gen:
                                def _next_cont():
                                    try:
                                        set_turn(session_id, _turn_id)
                                        return next(cont_gen)
                                    except StopIteration:
                                        return None

                                # The continuation system prompt tells the
                                # model not to emit another <tool_call> unless
                                # asked — but nothing enforced that until now.
                                # Without this, a disobedient continuation's
                                # raw <tool_call> text streamed straight to
                                # the user unprotected (GAIA_Project-pfdw).
                                # Re-parse the continuation the same way the
                                # first pass is parsed; suppress (don't
                                # execute, don't leak as text) any tool call
                                # it emits instead.
                                _cont_tc_parser = ToolCallParser() if _tc_enabled else None
                                _cont_produced_text = False
                                _cont_chars = 0  # visible text, for the preamble check below
                                _cont_hallucinated_again = False

                                while True:
                                    cont_event = await loop.run_in_executor(None, _next_cont)
                                    if cont_event is None:
                                        break
                                    if isinstance(cont_event, dict) and cont_event.get("type") == "token":
                                        cont_val = cont_event.get("value", "")
                                        cont_val = _strip_think_token(cont_val)
                                        if not cont_val:
                                            continue
                                        if _cont_tc_parser is not None:
                                            for cpe in _cont_tc_parser.feed(cont_val):
                                                if cpe.type == ParseEventType.TEXT:
                                                    if cpe.text.strip():
                                                        _cont_produced_text = True
                                                        _cont_chars += len(cpe.text.strip())
                                                    response_pieces.append(cpe.text)
                                                    yield json.dumps({"type": "token", "value": cpe.text}) + "\n"
                                                elif cpe.type in (ParseEventType.TOOL_CALL_DETECTED, ParseEventType.TOOL_ERROR):
                                                    _cont_hallucinated_again = True
                                                    logger.warning(
                                                        "Continuation emitted an unexpected tool_call "
                                                        "(disobeyed no-retry instruction) — suppressed: %s",
                                                        cpe.tool_name or cpe.error,
                                                    )
                                        else:
                                            if cont_val.strip():
                                                _cont_produced_text = True
                                            response_pieces.append(cont_val)
                                            yield json.dumps({"type": "token", "value": cont_val}) + "\n"

                                if _cont_tc_parser is not None:
                                    for cpe in _cont_tc_parser.flush():
                                        if cpe.type == ParseEventType.TEXT and cpe.text:
                                            if _is_repetition_garbage(cpe.text):
                                                logger.warning(
                                                    "Suppressed degenerate repetition-loop garbage from "
                                                    "continuation flush: %s", cpe.text[:150],
                                                )
                                                _cont_hallucinated_again = True
                                            elif cpe.text.strip():
                                                _cont_produced_text = True
                                                _cont_chars += len(cpe.text.strip())
                                                response_pieces.append(cpe.text)
                                                yield json.dumps({"type": "token", "value": cpe.text}) + "\n"
                                            else:
                                                response_pieces.append(cpe.text)
                                                yield json.dumps({"type": "token", "value": cpe.text}) + "\n"

                                # nxxl follow-up: the continuation disobeyed the
                                # "no retry" instruction AND produced nothing
                                # else usable — the user would otherwise get a
                                # bare "[tool -> failed]" marker (now suppressed
                                # for invalid tools above) with no actual content
                                # at all (live repro, 2026-09-02). One bounded,
                                # clean-slate retry: a fresh completion with NO
                                # tool-call turn anywhere in its history (a prior
                                # tool_call in-context measurably increases the
                                # odds of imitating it again) and a blunt
                                # no-tools instruction. If even THIS hallucinates,
                                # give up gracefully with an honest one-line
                                # apology rather than silence or more plumbing.
                                # r4p6: the retry still carries what the tool
                                # returned, as plain text in the user turn —
                                # without it Core answered "what's on this
                                # page?" with an invented book.
                                # ecnc: a lead-in alone ("I'll search my palace.")
                                # before the repeated call counts as nothing —
                                # live, that was the whole reply.
                                if _cont_hallucinated_again and (
                                        not _cont_produced_text or _cont_chars < _CONT_PREAMBLE_CHARS):
                                    logger.info("Attempting one clean-slate corrective retry (no tool-call precedent in history)")
                                    # Same plain-text tool result as the
                                    # continuation (_tool_text, above).
                                    _retry_messages = [
                                        {"role": "system", "content": (
                                            "You are GAIA, a direct and capable writer. You do not have "
                                            "and cannot use any tools, functions, JSON, or XML tags for "
                                            "this — just answer the user's request directly, in plain "
                                            "natural-language text, right now."
                                        )},
                                        {"role": "user", "content": _retry_user_turn},
                                    ]
                                    try:
                                        _retry_gen = _agent_core.generate_continuation(
                                            messages=_retry_messages,
                                            session_id=session_id,
                                            max_tokens=_CONT_MAX_TOKENS,
                                            temperature=_CONT_TEMPERATURE,
                                        )
                                        _retry_parser = ToolCallParser() if _tc_enabled else None
                                        _retry_hallucinated = False
                                        if _retry_gen:
                                            def _next_retry():
                                                try:
                                                    set_turn(session_id, _turn_id)
                                                    return next(_retry_gen)
                                                except StopIteration:
                                                    return None
                                            while True:
                                                _retry_event = await loop.run_in_executor(None, _next_retry)
                                                if _retry_event is None:
                                                    break
                                                if isinstance(_retry_event, dict) and _retry_event.get("type") == "token":
                                                    _retry_val = _strip_think_token(_retry_event.get("value", ""))
                                                    if not _retry_val:
                                                        continue
                                                    if _retry_parser is not None:
                                                        for rpe in _retry_parser.feed(_retry_val):
                                                            if rpe.type == ParseEventType.TEXT:
                                                                if _is_repetition_garbage(rpe.text):
                                                                    logger.warning("Suppressed repetition-loop garbage from clean-slate retry stream")
                                                                    _retry_hallucinated = True
                                                                else:
                                                                    if rpe.text.strip():
                                                                        _cont_produced_text = True
                                                                    response_pieces.append(rpe.text)
                                                                    yield json.dumps({"type": "token", "value": rpe.text}) + "\n"
                                                            elif rpe.type in (ParseEventType.TOOL_CALL_DETECTED, ParseEventType.TOOL_ERROR):
                                                                _retry_hallucinated = True
                                                    else:
                                                        if _retry_val.strip():
                                                            _cont_produced_text = True
                                                        response_pieces.append(_retry_val)
                                                        yield json.dumps({"type": "token", "value": _retry_val}) + "\n"
                                            if _retry_parser is not None:
                                                for rpe in _retry_parser.flush():
                                                    if rpe.type == ParseEventType.TEXT and rpe.text:
                                                        if _is_repetition_garbage(rpe.text):
                                                            logger.warning("Suppressed repetition-loop garbage from clean-slate retry flush")
                                                            _retry_hallucinated = True
                                                        else:
                                                            if rpe.text.strip():
                                                                _cont_produced_text = True
                                                            response_pieces.append(rpe.text)
                                                            yield json.dumps({"type": "token", "value": rpe.text}) + "\n"
                                        if _retry_hallucinated or not _cont_produced_text:
                                            logger.warning("Clean-slate retry also failed to produce usable text — falling back to an honest apology")
                                            _apology = "I'm having trouble putting that into words right now — mind asking me again in a moment?"
                                            response_pieces.append(_apology)
                                            yield json.dumps({"type": "token", "value": _apology}) + "\n"
                                    except Exception as retry_err:
                                        logger.warning("Clean-slate corrective retry failed: %s", retry_err)
                                        _apology = "I'm having trouble putting that into words right now — mind asking me again in a moment?"
                                        response_pieces.append(_apology)
                                        yield json.dumps({"type": "token", "value": _apology}) + "\n"
                                    yield json.dumps({"type": "flush"}) + "\n"
                        except Exception as cont_err:
                            logger.warning("Continuation generation failed: %s", cont_err)
                            # She has the result but no words for it. The raw
                            # result is neither a reply nor worth keeping as one (8ail).
                            _apology = "I'm having trouble putting that into words right now — mind asking me again in a moment?"
                            response_pieces.append(_apology)
                            yield json.dumps({"type": "token", "value": _apology}) + "\n"

                    except Exception as e:
                        # Full detail server-side only — never surface raw
                        # exception text (internal URLs, HTTP status strings)
                        # to the user (9t27).
                        logger.warning("Tool call execution failed: %s", e)
                        yield json.dumps({"type": "tool_status", "tool": tc.tool_name, "action": tc.tool_action, "state": "failed"}) + "\n"
                        _apology = "That didn't work on my end, so I don't have an answer from it. Mind asking me again in a moment?"
                        response_pieces.append(_apology)
                        yield json.dumps({"type": "token", "value": _apology}) + "\n"

            # 7o99 (live incident 2026-10-13): a turn can legitimately end with
            # response_pieces totally empty — not a crash, not malformed-syntax
            # garbage (that's the flush() repetition-garbage apology above),
            # just a deliberation pass that emitted nothing but cleanly-parsed,
            # correctly-suppressed tool calls and zero surrounding prose. Every
            # downstream consumer (Discord included) treats silence + no final
            # packet as total failure with no way to tell the user anything
            # went wrong at all. Give them something rather than nothing.
            if not response_pieces and _suppressed_tool_count > 0:
                _apology = "Sorry — I got tangled up trying to answer that. Could you ask again?"
                response_pieces.append(_apology)
                yield json.dumps({"type": "token", "value": _apology}) + "\n"

            # Ensure a flush is always emitted so front-ends send accumulated text
            if response_pieces:
                yield json.dumps({"type": "flush"}) + "\n"

            # Finalize response processing
            full_response = "".join(response_pieces)
            from gaia_core.utils.output_router import _strip_think_tags_robust
            from gaia_core.cognition.reply_scrub import scrub_reply
            full_response = _strip_think_tags_robust(full_response)
            full_response = scrub_reply(full_response)

            # 1mg: the Observer (conscience) on the user path — /process_packet
            # bypassed it. Two aspects of gate-2, unified here:
            #   WORTH-VOICING remediation — strip leaked meta-commentary (the
            #     Observer's active tool; respects VOICE_GATE_ENABLED via filter_voiced).
            #   SAFETY conscience — observe_user_path(): curated cheap checks (error
            #     patterns, fabricated code paths, epistemic honesty), no LLM, no
            #     noisy identity-keyword heuristic. Fast (~10-30ms). Logs flags +
            #     updates health; acting on safety verdicts is a follow-up.
            # Fail-safe; flag OBSERVER_ON_STREAM.
            try:
                from gaia_core.cognition.voice_gate import filter_voiced
                _vg_out, _vg_dbg = filter_voiced(full_response)
                if _vg_dbg.get("dropped") and not _vg_dbg.get("measure_only"):
                    logger.info("Observer remediation (strip): %d meta sentence(s) | e.g. %r",
                                len(_vg_dbg["dropped"]), _vg_dbg["dropped"][0]["sent"][:80])
                    full_response = _vg_out
            except Exception:
                logger.debug("worth-voicing strip failed (non-fatal)", exc_info=True)
            try:
                if os.environ.get("OBSERVER_ON_STREAM", "1").lower() in ("1", "true", "yes", "on") and full_response:
                    from gaia_core.utils.stream_observer import StreamObserver
                    _t_obs = __import__("time").perf_counter()
                    _obs = StreamObserver(_agent_core.config, llm=None, name="StreamPath-Observer")
                    _verdict = _obs.observe_user_path(packet, full_response)
                    _dt = (__import__("time").perf_counter() - _t_obs) * 1000
                    if _verdict and _verdict.level != "OK":
                        logger.info("StreamObserver(user-path): %s — %s [%.0fms]",
                                    _verdict.level, (_verdict.reason or "")[:120], _dt)
                    else:
                        logger.debug("StreamObserver(user-path): OK [%.0fms]", _dt)
            except Exception:
                logger.debug("stream-path observer failed (non-fatal)", exc_info=True)

            # b9ts: the other mind reads this reply once the turn is over.
            if full_response.strip() and source in ("web", "discord") and not _pending_recital(full_response):
                _sources = "\n".join(x for x in (
                    _sources_for_observer(final_packet_dict, _tool_texts_for_observer),
                    _agent_core.take_turn_source(session_id)) if x)
                _after_turn["second_mind"] = (
                    _agent_core, user_input or "", full_response,
                    "\n".join(x for x in (_self_facts_for_observer(history, full_response), _sources) if x),
                    session_id, bool(_tool_texts_for_observer), bool(_sources))

            # 29yq / bz6f: a native tool call detected mid-stream BREAKS out of
            # the loop consuming run_turn()'s generator to run the tool and
            # answer via generate_continuation(). run_turn() is never advanced
            # again, so what it does after that token's yield — keeping the
            # reply in the session, the lookup feedback — never runs, and its
            # cleanup waited for garbage collection. This is the designed path
            # for every inline tool call, so this turn is finished here: the
            # generator is closed now, and the reply she actually gave (the
            # continuation, not the text with the call in it) is kept.
            # _pending_tool_calls nonempty = the break fired; a turn that
            # finished on its own (final_packet_dict) kept itself already.
            if _pending_tool_calls:
                try:
                    await loop.run_in_executor(None, gen.close)
                except Exception:
                    logger.debug("Closing the interrupted run_turn generator failed", exc_info=True)
            if _pending_tool_calls and not final_packet_dict and full_response.strip():
                logger.info("session=%s: reply after inline tool call %s kept by the stream path",
                            session_id, ", ".join(_t.tool_name + (f"({_t.tool_action})" if _t.tool_action else "")
                                                  for _t in _pending_tool_calls))
                try:
                    _agent_core.session_manager.add_message(session_id, "assistant", full_response)
                except Exception:
                    log_gaia_error(logger, "GAIA-CORE-081",
                                   f"session={session_id}: the reply after an inline tool call "
                                   "could not be kept in session history", exc_info=True)
                try:
                    _agent_core._lookup_feedback(full_response, packet)
                except Exception:
                    logger.debug("Lookup feedback after inline tool call failed", exc_info=True)
                from gaia_common.protocols.cognition_packet import PacketState
                packet.status.state = PacketState.COMPLETED
                packet.response.candidate = full_response
                final_packet_dict = packet.to_serializable_dict()

            # e6a: paths that answer and persist on their own (deliberation,
            # reflex, fallback/escalation) return without the terminal packet
            # event; gaia-web then had no completed packet ("Core stream
            # finished without a final packet"). Emit one from this turn's
            # packet — no persistence here, those paths already did it.
            if not final_packet_dict and full_response.strip():
                try:
                    from gaia_common.protocols.cognition_packet import PacketState
                    packet.status.state = PacketState.COMPLETED
                    packet.status.finalized = True
                    packet.response.candidate = full_response
                    final_packet_dict = packet.to_serializable_dict()
                except Exception:
                    logger.warning("Terminal packet synthesis failed", exc_info=True)

            # 9nxd: a reply that says she added, filed or saved something must
            # have a tool behind it that did. She answered "Noted: gi-000000"
            # with no call at all, then "The note was added successfully" after
            # a memory lookup. The correction is shown and kept in her history,
            # so the next turn doesn't build on a thing that never happened.
            if full_response.strip() and source in ("web", "discord"):
                try:
                    from gaia_core.cognition import action_claims as _ac
                    from gaia_core.cognition import outcome_ledger as _ol
                    _claim = _ac.check(full_response, _ol.session_tools(session_id))
                    if _claim:
                        _fix = _ac.correction(_claim)
                        logger.info("session=%s: reply claims %s with no write behind it; corrected",
                                    session_id, _claim["claims"][:3])
                        _ol.record("unbacked_claim", session_id,
                                   _ol.current_packet(session_id) or getattr(packet.header, "packet_id", ""),
                                   claims=_claim["claims"][:5], failed=_claim["failed"],
                                   request=_ol.clip(user_input or ""))
                        _fix_text = f"\n\n*[{_fix}]*"
                        yield json.dumps({"type": "token", "value": _fix_text}) + "\n"
                        yield json.dumps({"type": "flush"}) + "\n"
                        full_response += _fix_text
                        try:
                            _agent_core.session_manager.add_message(session_id, "assistant", _fix)
                        except Exception:
                            logger.debug("Claim correction not kept in history", exc_info=True)
                except Exception:
                    logger.debug("Action-claim check failed (non-fatal)", exc_info=True)

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

                yield json.dumps({"type": "packet", "value": final_packet_dict}, default=str) + "\n"

            # Reset idle timer after response completes -- an ambient turn
            # that produced no reply never marked active at all (above), so
            # it must not do so here either; one that genuinely interjected
            # (full_response non-empty) counts as active either way.
            if idle_monitor and (_addressed or full_response.strip()):
                idle_monitor.mark_active()

        except Exception as e:
            logger.exception(f"Error in streaming turn loop: {e}")
            yield json.dumps({"type": "error", "value": str(e)}) + "\n"

    from gaia_core.cognition.reply_scrub import scrub_stream, unfounded_apology
    _drop_apology = False
    try:
        # Read before the turn runs: her last reply is still the last one kept.
        _sid = ((packet_data or {}).get("header") or {}).get("session_id") or ""
        _asked = ((packet_data or {}).get("content") or {}).get("original_prompt") or ""
        _last = next((m.get("content") or "" for m in reversed(_agent_core.session_manager.get_history(_sid) or [])
                      if isinstance(m, dict) and m.get("role") == "assistant"), "") if _sid else ""
        _drop_apology = unfounded_apology(_asked, _last)
        # i4vn: who they are and their pronouns, kept as the message arrives so
        # this turn is told them. Saved by code, so "I've noted your pronouns"
        # is a write that happened, not one to correct.
        from gaia_common.utils.conversation_memory import recorded as _recorded
        if _sid and _recorded(_sid):
            from gaia_common.utils.people import PeopleRegistry as _People
            _people = _People()
            _people.note_claim(_sid, _asked)
            # kkij: in a channel, who wrote it, from the Discord account; and
            # "X is a friend of mine named Y" from anyone, kept as told.
            from gaia_common.utils import people as _pp
            from gaia_common.utils.conversation_memory import channel_of as _channel_of
            _teller = None
            if _pp.is_channel(_sid):
                _route = (((packet_data or {}).get("header") or {}).get("output_routing") or {}).get("primary") or {}
                _teller = _pp.note_channel_author(_sid, str(_route.get("user_id") or ""),
                                                  (_route.get("metadata") or {}).get("author_name") or "", _people)
            else:
                _teller = _people.for_session(_sid)
            _pp.note_introduction(_asked, _pp._main_name(_teller) if _teller else "", _channel_of(_sid), _people)
            if _people.note_pronouns(_sid, _asked, speaker=_teller if _pp.is_channel(_sid) else None):
                from gaia_core.cognition import outcome_ledger as _ol
                _ol.note_call(_sid, "people", "set_pronouns", True)
        if _drop_apology:
            logger.info("session=%s: corrected about what her last reply never said; an opening apology is cut", _sid)
    except Exception:
        logger.debug("Apology check unavailable (non-fatal)", exc_info=True)
    return StreamingResponse(scrub_stream(_run_loop(), drop_apology=_drop_apology),
                             media_type="application/x-ndjson")


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
        "nano": os.environ.get("NANO_ENDPOINT", "http://localhost:8092"),  # legacy alias → Core embedded engine
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
        "nano": os.environ.get("NANO_ENDPOINT", "http://localhost:8092"),  # legacy alias → Core embedded engine
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

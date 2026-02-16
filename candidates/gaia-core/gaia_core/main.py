"""
gaia-core FastAPI application entry point.

Provides the HTTP API for the cognitive loop service.
This is The Brain - Cognitive loop and reasoning.
"""

import os
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("GAIA.Core.API")

# Suppress health check access log spam
try:
    from gaia_common.utils import install_health_check_filter
    install_health_check_filter()
except ImportError:
    pass  # gaia_common not available

# Global references for the cognitive system
_agent_core = None
_ai_manager = None


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

        logger.info("GAIA cognitive system initialized successfully")
        return True

    except Exception as e:
        logger.exception(f"Failed to initialize cognitive system: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    success = initialize_cognitive_system()
    if not success:
        logger.error("Cognitive system failed to initialize - endpoints will return errors")

    # Start sleep cycle loop
    _sleep_loop = None
    try:
        from gaia_core.config import get_config
        from gaia_core.cognition.sleep_cycle_loop import SleepCycleLoop

        config = get_config()
        sleep_enabled = getattr(config, "SLEEP_ENABLED", True)
        if sleep_enabled:
            _sleep_loop = SleepCycleLoop(
                config,
                model_pool=_ai_manager.model_pool if _ai_manager else None,
                agent_core=_agent_core,
            )
            # Store on app.state for endpoint access
            app.state.sleep_wake_manager = _sleep_loop.sleep_wake_manager
            app.state.idle_monitor = _sleep_loop.idle_monitor
            app.state.sleep_cycle_loop = _sleep_loop
            _sleep_loop.start()
            logger.info("Sleep cycle loop started")
        else:
            logger.info("Sleep cycle disabled (SLEEP_ENABLED=False)")
    except Exception:
        logger.warning("Failed to start sleep cycle loop", exc_info=True)

    yield

    # Shutdown
    if _sleep_loop is not None:
        _sleep_loop.initiate_shutdown()
        logger.info("Sleep cycle loop stopped (OFFLINE)")
    logger.info("GAIA Core shutting down...")


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


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "gaia-core",
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


@app.post("/process_packet")
async def process_packet(packet_data: Dict[str, Any]):
    """
    Process a CognitionPacket through the cognitive loop.

    This is the main entry point for processing user requests.
    Accepts a serialized CognitionPacket and returns the completed packet
    with the response populated.
    """
    global _agent_core, _ai_manager

    # Mark system active for sleep cycle idle tracking
    idle_monitor = getattr(app.state, "idle_monitor", None)
    if idle_monitor:
        idle_monitor.mark_active()

    if _agent_core is None or _ai_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Cognitive system not initialized. Check logs for startup errors."
        )

    try:
        # Import the packet class for deserialization
        from gaia_common.protocols.cognition_packet import CognitionPacket

        # Deserialize the incoming packet
        packet = CognitionPacket.from_dict(packet_data)

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

        # Run the cognitive loop
        # AgentCore.run_turn is a generator that yields {"type": "token", "value": "..."}
        response_pieces = []

        for event in _agent_core.run_turn(
            user_input=user_input,
            session_id=session_id,
            destination=destination,
            source=source,
            metadata=metadata
        ):
            if isinstance(event, dict) and event.get("type") == "token":
                response_pieces.append(event.get("value", ""))

        # Combine all response pieces
        full_response = "".join(response_pieces)

        # Strip <think>/<thinking> tags — the model's reasoning blocks must
        # never reach the user.  The output_router handles this for the
        # non-packet path, but /process_packet assembles tokens directly.
        from gaia_core.utils.output_router import _strip_think_tags_robust
        full_response = _strip_think_tags_robust(full_response)

        # Update the packet with the response
        packet.response.candidate = full_response
        packet.response.confidence = 0.9

        # Mark packet as completed
        from gaia_common.protocols.cognition_packet import PacketState
        packet.status.finalized = True
        packet.status.state = PacketState.COMPLETED

        # Compute final hashes
        packet.compute_hashes()

        logger.info(f"Completed packet {packet.header.packet_id} with {len(full_response)} chars response")

        # Return the completed packet
        return JSONResponse(
            status_code=200,
            content=packet.to_serializable_dict()
        )

    except Exception as e:
        logger.exception(f"Error processing packet: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing packet: {str(e)}"
        )

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
    yield
    # Shutdown
    logger.info("GAIA Core shutting down...")


app = FastAPI(
    title="GAIA Core",
    description="The Brain - Cognitive loop and reasoning engine",
    version="0.1.0",
    lifespan=lifespan,
)


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
            "/gpu/status": "Get GPU status and loaded models",
            "/gpu/release": "Release GPU for candidate testing (POST)",
            "/gpu/reclaim": "Reclaim GPU after candidate testing (POST)",
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


# =============================================================================
# GPU Management Endpoints
# =============================================================================

@app.post("/gpu/release")
async def release_gpu():
    """
    Release GPU resources (vLLM models) to allow external processes to use the GPU.

    This endpoint shuts down GPU-backed models and frees VRAM. The service continues
    running but will not have GPU inference capability until /gpu/reclaim is called.

    Use case: Allow candidate testing to claim GPU resources without stopping the
    live service entirely.

    Returns:
        JSON with success status and list of released models
    """
    global _ai_manager

    if _ai_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Cognitive system not initialized"
        )

    model_pool = _ai_manager.model_pool
    if not hasattr(model_pool, 'release_gpu'):
        raise HTTPException(
            status_code=501,
            detail="GPU release not supported by this model pool"
        )

    try:
        result = model_pool.release_gpu()
        logger.info(f"GPU release result: {result}")
        return JSONResponse(
            status_code=200 if result["success"] else 500,
            content=result
        )
    except Exception as e:
        logger.exception(f"Error releasing GPU: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error releasing GPU: {str(e)}"
        )


@app.post("/gpu/reclaim")
async def reclaim_gpu():
    """
    Reclaim GPU resources by reloading vLLM models.

    Call this after candidate testing is complete to restore GPU inference capability.

    Returns:
        JSON with success status and list of loaded models
    """
    global _ai_manager

    if _ai_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Cognitive system not initialized"
        )

    model_pool = _ai_manager.model_pool
    if not hasattr(model_pool, 'reclaim_gpu'):
        raise HTTPException(
            status_code=501,
            detail="GPU reclaim not supported by this model pool"
        )

    try:
        result = model_pool.reclaim_gpu()
        logger.info(f"GPU reclaim result: {result}")
        return JSONResponse(
            status_code=200 if result["success"] else 500,
            content=result
        )
    except Exception as e:
        logger.exception(f"Error reclaiming GPU: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reclaiming GPU: {str(e)}"
        )


@app.get("/gpu/status")
async def get_gpu_status():
    """
    Get current GPU status including loaded models and memory usage.

    Returns:
        JSON with GPU state information including:
        - gpu_released: whether GPU has been released
        - gpu_models_loaded: list of GPU-backed models currently loaded
        - gpu_info: memory usage (free, total, used in GB)
    """
    global _ai_manager

    if _ai_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Cognitive system not initialized"
        )

    model_pool = _ai_manager.model_pool
    if not hasattr(model_pool, 'get_gpu_status'):
        raise HTTPException(
            status_code=501,
            detail="GPU status not supported by this model pool"
        )

    try:
        result = model_pool.get_gpu_status()
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        logger.exception(f"Error getting GPU status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting GPU status: {str(e)}"
        )

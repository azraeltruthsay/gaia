"""
gaia-web FastAPI application entry point.

Provides the HTTP API gateway for the GAIA system.
This is The Face - UI and API gateway.
"""

import os
import uuid
import logging
from datetime import datetime
from typing import Dict, Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from gaia_common.protocols.cognition_packet import (
    CognitionPacket, Header, Persona, Origin, OutputRouting, DestinationTarget, Content, DataField,
    OutputDestination, PersonaRole, Routing, Model, OperationalStatus, SystemTask, Intent, Context,
    SessionHistoryRef, Constraints, Response, Governance, Safety, Metrics, TokenUsage, Status,
    PacketState, ToolRoutingState, Reasoning, TargetEngine
)

logger = logging.getLogger("GAIA.Web.API")

# Suppress health check access log spam
try:
    from gaia_common.utils import install_health_check_filter
    install_health_check_filter()
except ImportError:
    pass  # gaia_common not available

# Configuration from environment
CORE_ENDPOINT = os.environ.get("CORE_ENDPOINT", "http://gaia-core-candidate:6415")
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
ENABLE_DISCORD = os.environ.get("ENABLE_DISCORD", "0") == "1"

app = FastAPI(
    title="GAIA Web",
    description="The Face - UI and API gateway",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "gaia-web",
        }
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "gaia-web",
        "description": "GAIA Web Gateway Service",
        "endpoints": {
            "/health": "Health check",
            "/": "This endpoint",
        }
    }


@app.post("/process_user_input")
async def process_user_input(user_input: str):
    """
    Process a user input string by converting it to a CognitionPacket
    and sending it to gaia-core for processing.
    """
    packet_id = str(uuid.uuid4())
    session_id = "web_ui_session" # Placeholder, can be dynamic
    current_time = datetime.now().isoformat()

    packet = CognitionPacket(
        version="0.2",
        header=Header(
            datetime=current_time,
            session_id=session_id,
            packet_id=packet_id,
            sub_id="0",
            persona=Persona(
                identity_id="default_user",
                persona_id="default_web_user",
                role=PersonaRole.DEFAULT,
                tone_hint="neutral"
            ),
            origin=Origin.USER,
            routing=Routing(
                target_engine=TargetEngine.PRIME,
                priority=5
            ),
            model=Model(
                name="default_model",
                provider="default_provider",
                context_window_tokens=8192
            ),
            output_routing=OutputRouting(
                primary=DestinationTarget(
                    destination=OutputDestination.WEB,
                    channel_id=session_id, # Placeholder for web session ID
                    user_id="web_user", # Placeholder
                    metadata={}
                ),
                source_destination=OutputDestination.WEB,
                addressed_to_gaia=True,
            ),
            operational_status=OperationalStatus(status="initialized")
        ),
        intent=Intent(user_intent="chat", system_task=SystemTask.GENERATE_DRAFT, confidence=0.0),
        context=Context(
            session_history_ref=SessionHistoryRef(type="web_session", value=session_id),
            cheatsheets=[],
            constraints=Constraints(max_tokens=2048, time_budget_ms=30000, safety_mode="strict"),
        ),
        content=Content(
            original_prompt=user_input,
            data_fields=[DataField(key="user_message", value=user_input, type="text")]
        ),
        reasoning=Reasoning(),
        response=Response(candidate="", confidence=0.0, stream_proposal=False),
        governance=Governance(
            safety=Safety(execution_allowed=False, dry_run=True)
        ),
        metrics=Metrics(
            token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            latency_ms=0
        ),
        status=Status(finalized=False, state=PacketState.INITIALIZED, next_steps=[]),
        tool_routing=ToolRoutingState()
    )
    packet.compute_hashes()

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{CORE_ENDPOINT}/process_packet",
                json=packet.to_serializable_dict(),
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

            completed_packet_dict = response.json()
            completed_packet = CognitionPacket.from_dict(completed_packet_dict)

            return JSONResponse(
                status_code=200,
                content={"response": completed_packet.response.candidate, "packet_id": completed_packet.header.packet_id}
            )

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="GAIA Core did not respond in time.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Error from GAIA Core: {e.response.text}")
    except Exception as e:
        logger.exception(f"Error processing user input: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/output_router")
async def output_router(packet: Dict[str, Any]):
    """
    Route output from gaia-core to the appropriate destination based on CognitionPacket.

    This endpoint is called by gaia-core when it has a response ready
    for delivery (either in response to a user message or autonomously).
    """
    logger.info(f"Output router received packet_id: {packet.header.packet_id}")

    # Determine the primary destination from the packet
    output_routing = packet.header.output_routing
    if not output_routing or not output_routing.primary:
        logger.warning(f"Packet {packet.header.packet_id} has no primary output routing. Logging only.")
        logger.info(f"Output (log only for packet {packet.header.packet_id}): {packet.response.candidate[:200]}...")
        return JSONResponse(content={"status": "success", "message": "Logged due to no routing info"}, status_code=200)

    primary_destination = output_routing.primary
    destination_type = primary_destination.destination
    content = packet.response.candidate # The LLM's response

    if destination_type == OutputDestination.DISCORD:
        try:
            from .discord_interface import send_to_channel, send_to_user, is_bot_ready

            if not is_bot_ready():
                logger.error("Discord bot not connected for routing.")
                return JSONResponse(content={"status": "error", "message": "Discord bot not connected"}, status_code=503)

            success = False
            # Check metadata for is_dm and use_id, as send_to_user is specific
            if primary_destination.user_id and primary_destination.metadata.get("is_dm"):
                success = await send_to_user(primary_destination.user_id, content)
            elif primary_destination.channel_id:
                success = await send_to_channel(primary_destination.channel_id, content, primary_destination.reply_to_message_id)
            else:
                logger.warning(f"Discord routing failed: No channel_id or user_id in packet {packet.header.packet_id}")
                return JSONResponse(content={"status": "error", "message": "No channel_id or user_id provided for Discord"}, status_code=400)

            if success:
                return JSONResponse(content={"status": "success", "message": "Message sent to Discord"}, status_code=200)
            else:
                return JSONResponse(content={"status": "error", "message": "Failed to send message to Discord"}, status_code=500)

        except ImportError:
            logger.error("Discord integration not available during routing.")
            return JSONResponse(content={"status": "error", "message": "Discord integration not available"}, status_code=500)
        except Exception as e:
            logger.exception(f"Error routing to Discord for packet {packet.header.packet_id}: {e}")
            return JSONResponse(content={"status": "error", "message": f"Error: {str(e)}"}, status_code=500)

    elif destination_type == OutputDestination.WEB:
        # Future: WebSocket push to web clients
        logger.info(f"Web output routing for packet {packet.header.packet_id} not yet implemented")
        return JSONResponse(content={"status": "error", "message": "Web output routing not yet implemented"}, status_code=501)

    elif destination_type == OutputDestination.LOG:
        # Just log the output
        logger.info(f"Output (log only for packet {packet.header.packet_id}): {content[:200]}...")
        return JSONResponse(content={"status": "success", "message": "Logged"}, status_code=200)

    else:
        logger.warning(f"Unknown destination type '{destination_type}' for packet {packet.header.packet_id}")
        return JSONResponse(content={"status": "error", "message": f"Unknown destination type: {destination_type}"}, status_code=400)


@app.on_event("startup")
async def startup_event():
    """Start Discord bot on application startup if enabled."""
    print(f"[STARTUP] ENABLE_DISCORD={ENABLE_DISCORD}, TOKEN_SET={bool(DISCORD_BOT_TOKEN)}")
    if ENABLE_DISCORD and DISCORD_BOT_TOKEN:
        print("[STARTUP] Starting Discord bot...")
        try:
            from .discord_interface import start_discord_bot
            result = start_discord_bot(DISCORD_BOT_TOKEN, CORE_ENDPOINT)
            print(f"[STARTUP] Discord bot startup result: {result}")
        except Exception as e:
            print(f"[STARTUP] Failed to start Discord bot: {e}")
            import traceback
            traceback.print_exc()
    elif ENABLE_DISCORD:
        print("[STARTUP] Discord enabled but no DISCORD_BOT_TOKEN provided")
    else:
        print("[STARTUP] Discord integration disabled (set ENABLE_DISCORD=1 to enable)")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop Discord bot on application shutdown."""
    if ENABLE_DISCORD:
        try:
            from .discord_interface import stop_discord_bot
            stop_discord_bot()
            logger.info("Discord bot stopped")
        except Exception as e:
            logger.error(f"Error stopping Discord bot: {e}")

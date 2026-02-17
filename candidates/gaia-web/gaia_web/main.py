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

from gaia_web.queue.message_queue import MessageQueue

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


@app.get("/queue/status")
async def queue_status():
    """Return current message queue status."""
    mq: MessageQueue | None = getattr(app.state, "message_queue", None)
    if mq is None:
        return JSONResponse(status_code=503, content={"error": "MessageQueue not initialized"})
    return await mq.get_queue_status()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "gaia-web",
        "description": "GAIA Web Gateway Service",
        "endpoints": {
            "/health": "Health check",
            "/queue/status": "Message queue status",
            "/": "This endpoint",
        }
    }


@app.post("/process_user_input")
async def process_user_input(user_input: str):
    """
    Process a user input string by converting it to a CognitionPacket
    and sending it to gaia-core for processing.
    """
    # Sleep-aware: if GAIA is asleep, enqueue + wake + wait
    mq: MessageQueue | None = getattr(app.state, "message_queue", None)
    if mq is not None:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                check = await client.get(f"{CORE_ENDPOINT}/sleep/distracted-check")
                if check.status_code == 200:
                    data = check.json()
                    if data.get("state") == "asleep":
                        from gaia_web.queue.message_queue import QueuedMessage
                        qm = QueuedMessage(
                            message_id=str(uuid.uuid4()),
                            content=user_input,
                            source="web",
                            session_id="web_ui_session",
                        )
                        await mq.enqueue(qm)
                        woke = await mq.wait_for_active(timeout=120.0)
                        await mq.dequeue()
                        if not woke:
                            return JSONResponse(
                                status_code=503,
                                content={"response": "I'm having trouble waking up right now. Please try again in a moment.", "packet_id": None},
                            )
        except Exception:
            logger.debug("Sleep-check failed in /process_user_input — proceeding", exc_info=True)

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


@app.post("/presence")
async def update_presence(body: Dict[str, Any]):
    """Update Discord bot presence (status dot + activity text).

    Called by gaia-core's SleepCycleLoop to show sleep/wake status.
    Operates directly on the bot instance owned by discord_interface.
    """
    try:
        from .discord_interface import _bot
    except ImportError:
        return JSONResponse(status_code=501, content={"ok": False, "error": "Discord module not available"})

    if not _bot or not _bot.is_ready():
        return JSONResponse(status_code=503, content={"ok": False, "error": "Bot not connected"})

    import discord

    activity_name = body.get("activity", "over the studio")
    status_str = body.get("status")  # "idle", "online", "dnd", or None
    status_map = {
        "idle": discord.Status.idle,
        "online": discord.Status.online,
        "dnd": discord.Status.dnd,
        "invisible": discord.Status.invisible,
    }
    effective_status = status_map.get(status_str, discord.Status.online)

    if effective_status == discord.Status.invisible:
        await _bot.change_presence(status=effective_status, activity=None)
    else:
        await _bot.change_presence(
            status=effective_status,
            activity=discord.Activity(type=discord.ActivityType.watching, name=activity_name)
        )
    return {"ok": True}


@app.post("/output_router")
async def output_router(packet: Dict[str, Any]):
    """
    Route output from gaia-core to the appropriate destination based on CognitionPacket.

    This endpoint is called by gaia-core when it has a response ready
    for delivery (either in response to a user message or autonomously).
    """
    header = packet.get("header", {})
    packet_id = header.get("packet_id", "unknown")
    logger.info(f"Output router received packet_id: {packet_id}")

    # Determine the primary destination from the packet
    output_routing = header.get("output_routing", {})
    primary_destination = output_routing.get("primary") if isinstance(output_routing, dict) else None
    if not primary_destination:
        logger.warning(f"Packet {packet_id} has no primary output routing. Logging only.")
        response = packet.get("response", {})
        candidate = response.get("candidate", "")
        logger.info(f"Output (log only for packet {packet_id}): {str(candidate)[:200]}...")
        return JSONResponse(content={"status": "success", "message": "Logged due to no routing info"}, status_code=200)

    destination_type = primary_destination.get("destination") if isinstance(primary_destination, dict) else None
    response = packet.get("response", {})
    content = response.get("candidate", "")  # The LLM's response

    if destination_type == "discord":
        try:
            from .discord_interface import send_to_channel, send_to_user, is_bot_ready

            if not is_bot_ready():
                logger.error("Discord bot not connected for routing.")
                return JSONResponse(content={"status": "error", "message": "Discord bot not connected"}, status_code=503)

            success = False
            user_id = primary_destination.get("user_id") if isinstance(primary_destination, dict) else None
            channel_id = primary_destination.get("channel_id") if isinstance(primary_destination, dict) else None
            reply_to = primary_destination.get("reply_to_message_id") if isinstance(primary_destination, dict) else None
            metadata = primary_destination.get("metadata", {}) if isinstance(primary_destination, dict) else {}

            if user_id and metadata.get("is_dm"):
                success = await send_to_user(user_id, content)
            elif channel_id:
                success = await send_to_channel(channel_id, content, reply_to)
            else:
                logger.warning(f"Discord routing failed: No channel_id or user_id in packet {packet_id}")
                return JSONResponse(content={"status": "error", "message": "No channel_id or user_id provided for Discord"}, status_code=400)

            if success:
                return JSONResponse(content={"status": "success", "message": "Message sent to Discord"}, status_code=200)
            else:
                return JSONResponse(content={"status": "error", "message": "Failed to send message to Discord"}, status_code=500)

        except ImportError:
            logger.error("Discord integration not available during routing.")
            return JSONResponse(content={"status": "error", "message": "Discord integration not available"}, status_code=500)
        except Exception as e:
            logger.exception(f"Error routing to Discord for packet {packet_id}: {e}")
            return JSONResponse(content={"status": "error", "message": "Internal routing error"}, status_code=500)

    elif destination_type == "web":
        logger.info(f"Web output routing for packet {packet_id} not yet implemented")
        return JSONResponse(content={"status": "error", "message": "Web output routing not yet implemented"}, status_code=501)

    elif destination_type == "log":
        logger.info(f"Output (log only for packet {packet_id}): {str(content)[:200]}...")
        return JSONResponse(content={"status": "success", "message": "Logged"}, status_code=200)

    else:
        logger.warning(f"Unknown destination type '{destination_type}' for packet {packet_id}")
        return JSONResponse(content={"status": "error", "message": f"Unknown destination type: {destination_type}"}, status_code=400)


@app.on_event("startup")
async def startup_event():
    """Start Discord bot on application startup if enabled."""
    # Initialize message queue for sleep/wake cycle
    app.state.message_queue = MessageQueue(core_url=CORE_ENDPOINT)

    print(f"[STARTUP] ENABLE_DISCORD={ENABLE_DISCORD}, TOKEN_SET={bool(DISCORD_BOT_TOKEN)}")
    if ENABLE_DISCORD and DISCORD_BOT_TOKEN:
        print("[STARTUP] Starting Discord bot...")
        try:
            from .discord_interface import start_discord_bot
            result = start_discord_bot(DISCORD_BOT_TOKEN, CORE_ENDPOINT, message_queue=app.state.message_queue)
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

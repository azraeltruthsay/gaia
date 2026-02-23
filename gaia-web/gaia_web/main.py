"""
gaia-web FastAPI application entry point.

Provides the HTTP API gateway for the GAIA system.
This is The Face - UI and API gateway.
"""

import os
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from gaia_web.queue.message_queue import MessageQueue
from gaia_web.routes.blueprints import router as blueprints_router
from gaia_web.routes.files import router as files_router
from gaia_web.routes.hooks import router as hooks_router
from gaia_web.routes.terminal import router as terminal_router
from gaia_web.routes.voice import router as voice_router
from gaia_web.routes.wiki import router as wiki_router
from gaia_web.routes.generation import router as generation_router
from gaia_web.routes.logs import router as logs_router

from gaia_common.protocols.cognition_packet import (
    CognitionPacket, Header, Persona, Origin, OutputRouting, DestinationTarget, Content, DataField,
    OutputDestination, PersonaRole, Routing, Model, OperationalStatus, SystemTask, Intent, Context,
    SessionHistoryRef, Constraints, Response, Governance, Safety, Metrics, TokenUsage, Status,
    PacketState, ToolRoutingState, Reasoning, TargetEngine
)

# Persistent file logging — writes to /logs/gaia-web.log (mounted volume)
try:
    from gaia_common.utils import setup_logging, install_health_check_filter
    _log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    setup_logging(log_dir="/logs", level=_log_level, service_name="gaia-web")
    install_health_check_filter()
except ImportError:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("GAIA.Web.API")

# Configuration from environment
CORE_ENDPOINT = os.environ.get("CORE_ENDPOINT", "http://gaia-core-candidate:6415")
CORE_FALLBACK_ENDPOINT = os.environ.get("CORE_FALLBACK_ENDPOINT", "")
ORCHESTRATOR_ENDPOINT = os.environ.get("ORCHESTRATOR_ENDPOINT", "http://gaia-orchestrator:6410")
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
ENABLE_DISCORD = os.environ.get("ENABLE_DISCORD", "0") == "1"
AUDIO_ENDPOINT = os.environ.get("AUDIO_ENDPOINT", "http://gaia-audio:8080")
GAIA_CONSTANTS_PATH = os.environ.get("GAIA_CONSTANTS_PATH", "/app/gaia_common/constants/gaia_constants.json")


def _load_constants() -> dict:
    """Load gaia_constants.json (cached after first read)."""
    if not hasattr(_load_constants, "_cache"):
        import json
        try:
            with open(GAIA_CONSTANTS_PATH, encoding="utf-8") as f:
                _load_constants._cache = json.load(f)
        except Exception:
            _load_constants._cache = {}
    return _load_constants._cache

app = FastAPI(
    title="GAIA Web",
    description="The Face - UI and API gateway",
    version="0.1.0",
)

# API routers (must be before static mount)
app.include_router(blueprints_router)
app.include_router(files_router)
app.include_router(hooks_router)
app.include_router(terminal_router)
app.include_router(voice_router)
app.include_router(wiki_router)
app.include_router(generation_router)
app.include_router(logs_router)

# Static file serving for dashboard UI
_static_dir = Path(__file__).parent.parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


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
            "/dashboard": "Mission Control dashboard",
            "/api/blueprints": "Blueprint API",
            "/api/blueprints/graph": "Blueprint graph topology",
            "/api/system/status": "System status (proxy)",
            "/api/system/sleep": "Sleep status (proxy)",
            "/": "This endpoint",
        }
    }


@app.get("/dashboard")
async def dashboard_redirect():
    """Redirect to the Mission Control dashboard."""
    return RedirectResponse(url="/static/index.html")


@app.get("/api/system/status")
async def system_status_proxy():
    """Proxy to orchestrator /status endpoint (avoids CORS from browser)."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{ORCHESTRATOR_ENDPOINT}/status")
            return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.ConnectError:
        return JSONResponse(status_code=503, content={"error": "orchestrator unreachable"})
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


# Service registry: name → (internal URL, is_candidate)
_SERVICE_REGISTRY = {
    "gaia-core": (os.environ.get("CORE_ENDPOINT", "http://gaia-core:6415"), False),
    "gaia-web": ("http://localhost:6414", False),  # self
    "gaia-orchestrator": (os.environ.get("ORCHESTRATOR_ENDPOINT", "http://gaia-orchestrator:6410"), False),
    "gaia-prime": (os.environ.get("PRIME_ENDPOINT", "http://gaia-prime:7777"), False),
    "gaia-mcp": (os.environ.get("MCP_HEALTH_ENDPOINT", "http://gaia-mcp:8765"), False),
    "gaia-study": (os.environ.get("STUDY_ENDPOINT", "http://gaia-study:8766"), False),
}

# Add candidate services if they exist
_CANDIDATE_CORE = os.environ.get("CANDIDATE_CORE_ENDPOINT", "http://gaia-core-candidate:6415")
if _CANDIDATE_CORE:
    _SERVICE_REGISTRY["gaia-core-candidate"] = (_CANDIDATE_CORE, True)


@app.get("/api/system/services")
async def system_services():
    """Health check all known services. Returns status for dashboard."""
    results = []

    async with httpx.AsyncClient(timeout=3.0) as client:
        for name, (url, is_candidate) in _SERVICE_REGISTRY.items():
            entry = {
                "id": name,
                "candidate": is_candidate,
                "url": url,
                "status": "unknown",
                "latency_ms": None,
            }
            try:
                import time as _time
                t0 = _time.monotonic()
                resp = await client.get(f"{url}/health")
                latency = (_time.monotonic() - t0) * 1000
                entry["latency_ms"] = round(latency, 1)
                if resp.status_code == 200:
                    entry["status"] = "online"
                else:
                    entry["status"] = f"error ({resp.status_code})"
            except httpx.ConnectError:
                entry["status"] = "offline"
            except httpx.TimeoutException:
                entry["status"] = "timeout"
            except Exception:
                entry["status"] = "error"
            results.append(entry)

    # Add Discord status (not an HTTP service)
    try:
        from .discord_interface import get_discord_status
        discord_status = get_discord_status()
        results.append({
            "id": "discord",
            "candidate": False,
            "url": None,
            "status": "online" if discord_status["connected"] else discord_status["status"],
            "latency_ms": discord_status.get("latency_ms"),
            "discord": discord_status,
        })
    except ImportError:
        results.append({
            "id": "discord",
            "candidate": False,
            "url": None,
            "status": "not_available",
            "latency_ms": None,
        })

    return results


@app.get("/api/system/sleep")
async def system_sleep_proxy():
    """Proxy to gaia-core /sleep/status endpoint (avoids CORS from browser)."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{CORE_ENDPOINT}/sleep/status")
            return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.ConnectError:
        return JSONResponse(status_code=503, content={"error": "gaia-core unreachable"})
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


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
                    if data.get("state") in ("asleep", "drowsy"):
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
        version="0.3",
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
        from gaia_web.utils.retry import post_with_retry

        fallback = f"{CORE_FALLBACK_ENDPOINT}/process_packet" if CORE_FALLBACK_ENDPOINT else None
        response = await post_with_retry(
            f"{CORE_ENDPOINT}/process_packet",
            json=packet.to_serializable_dict(),
            fallback_url=fallback,
        )

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


@app.post("/process_audio_input")
async def process_audio_input(body: Dict[str, Any]):
    """Accept transcribed text from gaia-audio, route through gaia-core.

    Like /process_user_input but sets origin=audio and destination=AUDIO
    so the response is routed back to gaia-audio for synthesis.
    """
    user_input = body.get("text", "")
    session_id = body.get("session_id", "voice_session")
    if not user_input:
        raise HTTPException(status_code=400, detail="'text' field required")

    packet_id = f"pkt-audio-{uuid.uuid4().hex[:12]}"
    now = datetime.now().isoformat()

    packet = CognitionPacket(
        version="0.3",
        header=Header(
            datetime=now,
            session_id=session_id,
            packet_id=packet_id,
            sub_id="audio-web-gateway",
            persona=Persona(identity_id="Prime", persona_id="Default", role=PersonaRole.DEFAULT),
            origin=Origin(source="audio", source_destination=OutputDestination.AUDIO),
            routing=Routing(target_engine=TargetEngine.PRIME),
            model=Model(name="auto", provider="auto", context_window_tokens=8192),
            output_routing=OutputRouting(
                primary=DestinationTarget(destination=OutputDestination.AUDIO.value),
            ),
        ),
        intent=Intent(
            user_intent=user_input[:200],
            system_task=SystemTask.CHAT,
            confidence=0.9,
        ),
        context=Context(
            session_history_ref=SessionHistoryRef(type="ref", value=session_id),
            cheatsheets=[],
            constraints=Constraints(max_tokens=2048, time_budget_ms=30000, safety_mode="standard"),
        ),
        content=Content(original_prompt=user_input),
        reasoning=Reasoning(),
        response=Response(candidate="", confidence=0.0, stream_proposal=False),
        governance=Governance(safety=Safety(execution_allowed=False, dry_run=False)),
        metrics=Metrics(token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0), latency_ms=0),
        status=Status(finalized=False, state=PacketState.PROCESSING),
    )

    try:
        from gaia_web.utils.retry import post_with_retry

        fallback = f"{CORE_FALLBACK_ENDPOINT}/process_packet" if CORE_FALLBACK_ENDPOINT else None
        response = await post_with_retry(
            f"{CORE_ENDPOINT}/process_packet",
            json=packet.to_serializable_dict(),
            fallback_url=fallback,
        )
        completed_packet_dict = response.json()
        completed_packet = CognitionPacket.from_dict(completed_packet_dict)
        return JSONResponse(
            status_code=200,
            content={
                "response": completed_packet.response.candidate,
                "packet_id": completed_packet.header.packet_id,
                "destination": "audio",
            },
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="GAIA Core did not respond in time.")
    except Exception as e:
        logger.exception(f"Error processing audio input: {e}")
        raise HTTPException(status_code=500, detail=f"Audio processing error: {e}")


@app.post("/presence")
async def update_presence(body: Dict[str, Any]):
    """Update Discord bot presence (status dot + activity text).

    Called by gaia-core's SleepCycleLoop to show sleep/wake status.
    Uses run_coroutine_threadsafe to safely schedule on the bot's event loop.
    """
    try:
        from .discord_interface import change_presence_from_external, is_bot_ready
    except ImportError:
        return JSONResponse(status_code=501, content={"ok": False, "error": "Discord module not available"})

    if not is_bot_ready():
        return JSONResponse(status_code=503, content={"ok": False, "error": "Bot not connected"})

    activity_name = body.get("activity", "over the studio")
    status_str = body.get("status")  # "idle", "online", "dnd", or None

    try:
        change_presence_from_external(activity_name, status_str)
        return {"ok": True}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


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

    elif destination_type == "audio":
        try:
            constants = _load_constants()
            audio_cfg = constants.get("INTEGRATIONS", {}).get("audio", {})
            audio_endpoint = audio_cfg.get("endpoint", "http://gaia-audio:8080")
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{audio_endpoint}/synthesize",
                    json={"text": content, "voice": None},
                )
                resp.raise_for_status()
            logger.info(f"Audio synthesis dispatched for packet {packet_id}")
            return JSONResponse(content={"status": "success", "message": "Dispatched to gaia-audio"}, status_code=200)
        except Exception as e:
            logger.error(f"Audio routing failed for packet {packet_id}: {e}")
            return JSONResponse(content={"status": "error", "message": f"Audio dispatch failed: {e}"}, status_code=500)

    else:
        logger.warning(f"Unknown destination type '{destination_type}' for packet {packet_id}")
        return JSONResponse(content={"status": "error", "message": f"Unknown destination type: {destination_type}"}, status_code=400)


@app.on_event("startup")
async def startup_event():
    """Start Discord bot on application startup if enabled."""
    # Initialize message queue for sleep/wake cycle
    app.state.message_queue = MessageQueue(core_url=CORE_ENDPOINT)

    # Initialize voice manager (whitelist persists to /app/data/)
    voice_manager = None
    try:
        from gaia_web.voice_manager import VoiceManager, VoiceWhitelist
        constants = _load_constants()
        voice_cfg = constants.get("INTEGRATIONS", {}).get("discord", {}).get("voice", {})
        whitelist = VoiceWhitelist(data_dir=os.environ.get("VOICE_DATA_DIR", "/app/data"))
        voice_manager = VoiceManager(
            core_endpoint=CORE_ENDPOINT,
            audio_endpoint=AUDIO_ENDPOINT,
            whitelist=whitelist,
            voice_config=voice_cfg,
        )
        app.state.voice_manager = voice_manager
        print("[STARTUP] Voice manager initialized")
    except Exception as e:
        print(f"[STARTUP] Voice manager init failed (non-fatal): {e}")
        app.state.voice_manager = None

    print(f"[STARTUP] ENABLE_DISCORD={ENABLE_DISCORD}, TOKEN_SET={bool(DISCORD_BOT_TOKEN)}")
    if ENABLE_DISCORD and DISCORD_BOT_TOKEN:
        print("[STARTUP] Starting Discord bot...")
        try:
            from .discord_interface import start_discord_bot
            result = start_discord_bot(
                DISCORD_BOT_TOKEN, CORE_ENDPOINT,
                message_queue=app.state.message_queue,
                voice_manager=voice_manager,
                core_fallback_endpoint=CORE_FALLBACK_ENDPOINT,
            )
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
    # Disconnect voice if active
    vm = getattr(app.state, "voice_manager", None)
    if vm is not None:
        try:
            await vm.disconnect()
        except Exception:
            pass

    if ENABLE_DISCORD:
        try:
            from .discord_interface import stop_discord_bot
            stop_discord_bot()
            logger.info("Discord bot stopped")
        except Exception as e:
            logger.error(f"Error stopping Discord bot: {e}")

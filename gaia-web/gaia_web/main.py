"""
gaia-web FastAPI application entry point.

Provides the HTTP API gateway for the GAIA system.
This is The Face - UI and API gateway.
"""

import os
import uuid
import logging
import json
import httpx
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List

from gaia_web.routes.blueprints import router as blueprints_router
from gaia_web.routes.files import router as files_router
from gaia_web.routes.hooks import router as hooks_router
from gaia_web.routes.terminal import router as terminal_router
from gaia_web.routes.voice import router as voice_router
from gaia_web.routes.wiki import router as wiki_router
from gaia_web.routes.generation import router as generation_router
from gaia_web.routes.logs import router as logs_router
from gaia_web.routes.audio import router as audio_router
from gaia_web.routes.discord import router as discord_router
from gaia_web.routes.system import router as system_router
from gaia_web.routes.chaos import router as chaos_router
from gaia_web.routes.changelog import router as changelog_router
from gaia_web.routes.codemind import router as codemind_router
from gaia_web.routes.conversations import router as conversations_router
from gaia_web.routes.activations import router as activations_router
from gaia_web.routes.autonomous import router as autonomous_router
from gaia_web.routes.curriculum import router as curriculum_router

# Setup logging
try:
    from gaia_common.utils import setup_logging, install_health_check_filter
    setup_logging(log_dir="/logs", level=logging.INFO, service_name="gaia-web")
    install_health_check_filter()
except ImportError:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("GAIA.Web.API")

try:
    from gaia_common.utils.error_logging import log_gaia_error
except ImportError:
    def log_gaia_error(lgr, code, detail="", **kw):
        lgr.error("[%s] %s", code, detail)

# Module-level security middleware singleton
from gaia_web.security.middleware import SecurityScanMiddleware
_security_middleware = SecurityScanMiddleware()

# Global singleton for the Discord interface (if enabled)
discord_bot = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the web gateway."""
    logger.info("Initializing GAIA Web Gateway...")
    
    # Initialize Discord bot in background if enabled
    if os.getenv("ENABLE_DISCORD", "1") == "1":
        from gaia_web.discord_interface import DiscordInterface
        global discord_bot
        
        # Prefer Docker secret file, fall back to env var
        bot_token = None
        secret_path = "/run/secrets/discord_bot_token"
        if os.path.exists(secret_path):
            bot_token = open(secret_path).read().strip()
        if not bot_token:
            bot_token = os.getenv("DISCORD_BOT_TOKEN")
        core_url = os.getenv("CORE_ENDPOINT", "http://gaia-core:6415")
        core_fallback = os.getenv("CORE_FALLBACK_ENDPOINT", "")
        
        if not bot_token:
            log_gaia_error(logger, "GAIA-WEB-015", "Discord bot enabled but DISCORD_BOT_TOKEN not set")
        else:
            from gaia_web.queue.message_queue import MessageQueue
            mq = MessageQueue(core_url=core_url)
            discord_bot = DiscordInterface(
                bot_token=bot_token,
                core_endpoint=core_url,
                message_queue=mq,
                core_fallback_endpoint=core_fallback
            )
            asyncio.create_task(discord_bot.start())
            logger.info("Discord bot initialization task started")
    
    yield
    
    if discord_bot:
        from gaia_web.discord_interface import stop_discord_bot
        stop_discord_bot()
        logger.info("Discord bot closed")
    logger.info("GAIA Web Gateway shutting down...")

app = FastAPI(lifespan=lifespan, title="GAIA Web Gateway")

# Inter-service HMAC authentication
# NOTE: gaia-web is special — it serves users directly, so we need to
# allow unauthenticated access to user-facing routes (/, /static, /dashboard)
# while requiring auth on API routes called by other services.
try:
    from gaia_common.utils.service_auth import AuthMiddleware, _PUBLIC_PATHS
    if AuthMiddleware:
        # Add user-facing paths to the public list
        _PUBLIC_PATHS.update({"/", "/dashboard", "/static", "/api/activations"})
        app.add_middleware(AuthMiddleware)
except ImportError:
    pass

# Static files directory
static_dir = Path(__file__).parent.parent / "static"

# Root route — serve dashboard at /
@app.get("/")
async def root():
    """Serve the Mission Control dashboard."""
    return FileResponse(str(static_dir / "index.html"))


# /dashboard redirect → static dashboard
@app.get("/dashboard")
async def dashboard_redirect():
    """Redirect /dashboard to the static Mission Control page."""
    from starlette.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html", status_code=307)

# Mount static assets at /static
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include routers
# NOTE: Most routers define their own /api/<name> prefix internally,
# so we mount them at root to avoid double-prefixing.
app.include_router(blueprints_router, tags=["blueprints"])
app.include_router(files_router, tags=["files"])
app.include_router(hooks_router, tags=["hooks"])
app.include_router(terminal_router, tags=["terminal"])
app.include_router(voice_router, tags=["voice"])
app.include_router(wiki_router, tags=["wiki"])
app.include_router(generation_router, tags=["generation"])
app.include_router(logs_router, tags=["logs"])
app.include_router(audio_router, tags=["audio"])
app.include_router(discord_router, tags=["discord"])
# system_router uses relative paths, so it needs the prefix
app.include_router(system_router, prefix="/api/system", tags=["system"])
app.include_router(chaos_router, prefix="/api/chaos", tags=["chaos"])
app.include_router(changelog_router, prefix="/api/changelog", tags=["changelog"])
app.include_router(codemind_router, prefix="/api/codemind", tags=["codemind"])
app.include_router(conversations_router, prefix="/api/conversations", tags=["conversations"])
app.include_router(activations_router, tags=["activations"])
app.include_router(autonomous_router, tags=["autonomous"])
app.include_router(curriculum_router, prefix="/api/curriculum", tags=["curriculum"])

@app.post("/process_user_input")
async def process_user_input(user_input: str, request: Request):
    """
    Standard entry point for user text input.
    Routes to the Core service and returns an NDJSON stream.
    """
    session_id = request.headers.get("X-Session-ID")
    if not session_id:
        session_id = f"web_{uuid.uuid4().hex[:8]}"
        logger.warning("No X-Session-ID header — ephemeral session %s (follow-ups will lack context)", session_id)
    logger.info("Processing user input for session %s", session_id)
    context_pool = request.headers.get("X-Context-Pool", "").lower() in ("true", "1", "yes")
    core_url = os.getenv("CORE_ENDPOINT", "http://gaia-core:6415")
    packet_id = f"web_{uuid.uuid4().hex[:8]}"

    logger.info("Processing user input for session %s", session_id)

    # Inbound security scan before forwarding to gaia-core
    redacted_input, scan, should_block = _security_middleware.scan_text(user_input, packet_id, session_id)
    if should_block:
        async def _blocked():
            yield json.dumps({"type": "error", "value": "Request blocked by security scan.", "error_code": "GAIA-WEB-020", "hint": "The security scanner blocked this request. Rephrase and try again."}) + "\n"
        return StreamingResponse(_blocked(), media_type="application/x-ndjson")

    # Pre-flight: if planning request detected, request FOCUSING mode so Prime
    # loads while core processes the Nano triage and cascade routing.
    _planning_kw = ["implementation plan", "detailed plan", "create a plan",
                     "design a system", "plan for adding", "architecture plan",
                     "step by step implementation"]
    if any(kw in user_input.lower() for kw in _planning_kw):
        try:
            orch_url = os.getenv("ORCHESTRATOR_ENDPOINT", "http://gaia-orchestrator:6410")
            async with httpx.AsyncClient(timeout=180.0) as orch_client:
                state_resp = await orch_client.get(f"{orch_url}/lifecycle/state", timeout=5.0)
                if state_resp.status_code == 200:
                    state = state_resp.json().get("state", "")
                    if state != "focusing":
                        logger.info("Planning request detected — requesting FOCUSING mode (waiting for GPU Prime load)")
                        focus_resp = await orch_client.post(f"{orch_url}/consciousness/focusing", timeout=120.0)
                        if focus_resp.status_code == 200:
                            logger.info("FOCUSING mode active — Prime on GPU")
                        else:
                            logger.warning("FOCUSING request returned %d", focus_resp.status_code)
                    else:
                        logger.info("Already in FOCUSING mode — Prime should be on GPU")
        except Exception as _focus_err:
            logger.warning("Pre-flight FOCUSING failed (non-blocking): %s", _focus_err)

    async def _stream_response():
        # Readiness check — mirrors the gate already in the Discord interface.
        # If core is asleep/drowsy: notify the user, send a wake signal, wait.
        # If core returns a canned response (DREAMING/DISTRACTED): surface it and stop.
        try:
            async with httpx.AsyncClient(timeout=5.0) as check_client:
                check = await check_client.get(f"{core_url}/sleep/distracted-check")
                if check.status_code == 200:
                    data = check.json()
                    core_state = data.get("state", "active")
                    canned = data.get("canned_response")
                    if canned:
                        yield json.dumps({"type": "token", "value": canned}) + "\n"
                        return
                    if core_state in ("asleep", "drowsy"):
                        # Double-check: if the engine IS loaded, the sleep manager
                        # is just stale. Force wake and proceed immediately.
                        try:
                            async with httpx.AsyncClient(timeout=3.0) as _eng_check:
                                _eng = await _eng_check.get(f"{core_url}/model/status")
                                if _eng.status_code == 200 and _eng.json().get("model_loaded"):
                                    logger.info("Sleep state stale — engine loaded, forcing wake")
                                    await _eng_check.post(f"{core_url}/sleep/wake")
                                    core_state = "active"  # skip the wake-wait loop
                        except Exception:
                            pass
                    if core_state in ("asleep", "drowsy"):
                        yield json.dumps({"type": "status", "value": "GAIA is waking up — your message is queued, please hold..."}) + "\n"
                        # Send wake signal. The web connection stays open so we
                        # don't need persistent queue storage — just signal + poll.
                        try:
                            async with httpx.AsyncClient(timeout=5.0) as wake_client:
                                await wake_client.post(f"{core_url}/sleep/wake")
                                # Also tell orchestrator to shift to AWAKE (loads GPU model)
                                orch_url = os.getenv("ORCHESTRATOR_ENDPOINT", "http://gaia-orchestrator:6410")
                                await wake_client.post(f"{orch_url}/consciousness/awake")
                        except Exception:
                            logger.debug("Wake signal send failed (non-blocking)")
                        import time as _time
                        _deadline = _time.monotonic() + 120.0
                        _woke = False
                        while _time.monotonic() < _deadline:
                            await asyncio.sleep(1.5)
                            try:
                                async with httpx.AsyncClient(timeout=5.0) as poll_client:
                                    s = await poll_client.get(f"{core_url}/sleep/status")
                                    if s.status_code == 200 and s.json().get("state") == "active":
                                        _woke = True
                                        break
                            except Exception:
                                pass
                        if not _woke:
                            yield json.dumps({"type": "error", "value": "GAIA is having trouble waking up. Please try again in a moment.", "error_code": "GAIA-WEB-050"}) + "\n"
                            return
        except Exception:
            logger.debug("Readiness check failed — proceeding normally", exc_info=True)

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                # Forward to core as a v0.4 CognitionPacket
                payload = {
                    "version": "v0.4",
                    "header": {
                        "session_id": session_id,
                        "packet_id": packet_id,
                        "persona": {"persona_id": "gaia", "role": "assistant"}
                    },
                    "content": {"original_prompt": redacted_input},
                    "governance": {
                        "security_scan": {
                            "ran": scan.ran,
                            "passed": scan.passed,
                            "injection_score": scan.injection_score,
                        },
                        "context_pool": context_pool,
                    },
                }

                async with client.stream("POST", f"{core_url}/process_packet", json=payload) as resp:
                    if resp.status_code != 200:
                        yield json.dumps({"type": "error", "value": f"Core returned {resp.status_code}", "error_code": "GAIA-WEB-001", "hint": "gaia-core is not responding correctly. Check that gaia-core is running."}) + "\n"
                        return

                    async for line in resp.aiter_lines():
                        if line:
                            yield line + "\n"

        except Exception as e:
            log_gaia_error(logger, "GAIA-WEB-030", str(e), exc_info=True)
            yield json.dumps({"type": "error", "value": str(e), "error_code": "GAIA-WEB-030"}) + "\n"

    return StreamingResponse(_stream_response(), media_type="application/x-ndjson")


_WEB_ATTACHMENT_DIR = os.environ.get("WEB_ATTACHMENT_DIR", "/shared/web_attachments")
_ALLOWED_IMAGE_MIMES = {"image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp"}
_ALLOWED_AUDIO_MIMES = {"audio/wav", "audio/x-wav", "audio/wave",
                        "audio/mp3", "audio/mpeg",
                        "audio/ogg", "audio/webm", "audio/flac"}
_MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB per file
_MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25MB per audio clip (~3min @ 128kbps)


@app.post("/process_user_input_multipart")
async def process_user_input_multipart(
    request: Request,
    text: str = Form(""),
    files: List[UploadFile] = File(default=[]),
):
    """Multipart entry point for user text + image/audio attachments.

    Images are saved to /shared/web_attachments/{session_id}/ (volume mounted
    on both gaia-web and gaia-core) and the packet's `attachments` field is
    populated with the file path; gaia-core's multimodal vision path picks
    them up via prompt_builder + agent_core.

    Audio files are read into memory, base64-encoded, and packed into the
    packet's `audio_payloads` list (v0.5 schema). gaia-core's NeuralRouter
    sees audio_payloads via metadata pass-through and routes to Core's
    multimodal audio path. Audio quality is bound by 7rq (audio side
    training); plumbing here is correct regardless.
    """
    import hashlib
    import base64 as _b64

    session_id = request.headers.get("X-Session-ID")
    if not session_id:
        session_id = f"web_{uuid.uuid4().hex[:8]}"
        logger.warning("No X-Session-ID header — ephemeral session %s (follow-ups will lack context)", session_id)
    context_pool = request.headers.get("X-Context-Pool", "").lower() in ("true", "1", "yes")
    core_url = os.getenv("CORE_ENDPOINT", "http://gaia-core:6415")
    packet_id = f"web_{uuid.uuid4().hex[:8]}"

    logger.info("Multipart input for session %s: text_len=%d files=%d",
                session_id, len(text or ""), len(files) if files else 0)

    redacted_input, scan, should_block = _security_middleware.scan_text(text or "", packet_id, session_id)
    if should_block:
        async def _blocked():
            yield json.dumps({"type": "error", "value": "Request blocked by security scan.",
                              "error_code": "GAIA-WEB-020"}) + "\n"
        return StreamingResponse(_blocked(), media_type="application/x-ndjson")

    session_dir = Path(_WEB_ATTACHMENT_DIR) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    _EXT_MIME = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
        ".wav": "audio/wav", ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg", ".webm": "audio/webm",
        ".flac": "audio/flac", ".m4a": "audio/mp4",
    }

    attachments_meta = []
    audio_payloads_meta = []
    for f in (files or []):
        mime = (f.content_type or "").lower()
        # Browsers (and curl) sometimes send octet-stream / blank for known
        # extensions. Fall back to extension-based detection.
        if mime in ("", "application/octet-stream", "binary/octet-stream"):
            ext = Path(f.filename or "").suffix.lower()
            mime = _EXT_MIME.get(ext, mime)
        is_image = mime in _ALLOWED_IMAGE_MIMES
        is_audio = mime in _ALLOWED_AUDIO_MIMES or mime.startswith("audio/")
        if not (is_image or is_audio):
            async def _bad_mime():
                yield json.dumps({
                    "type": "error",
                    "value": f"Unsupported file type: {mime or 'unknown'} "
                             f"(images: PNG/JPEG/GIF/WEBP/BMP; audio: WAV/MP3/OGG/WEBM/FLAC)",
                    "error_code": "GAIA-WEB-MM-001",
                }) + "\n"
            return StreamingResponse(_bad_mime(), media_type="application/x-ndjson")

        data = await f.read()
        max_bytes = _MAX_IMAGE_BYTES if is_image else _MAX_AUDIO_BYTES
        if len(data) > max_bytes:
            async def _too_big():
                yield json.dumps({
                    "type": "error",
                    "value": f"'{f.filename}' exceeds {max_bytes // (1024 * 1024)}MB",
                    "error_code": "GAIA-WEB-MM-002",
                }) + "\n"
            return StreamingResponse(_too_big(), media_type="application/x-ndjson")

        chash = hashlib.sha256(data).hexdigest()[:16]
        safe_name = Path(f.filename or ("image" if is_image else "audio")).name.replace("/", "_")

        if is_image:
            local_path = str(session_dir / f"{chash}_{safe_name}")
            with open(local_path, "wb") as out:
                out.write(data)
            attachments_meta.append({
                "name": safe_name,
                "mime": mime,
                "content_hash": chash,
                "bytes": len(data),
                "location": local_path,
            })
            logger.info("Saved image upload %s (%d bytes) → %s", safe_name, len(data), local_path)
        else:
            # Audio: pack as AudioPayload (base64). Don't persist to disk —
            # core-side payload propagation is via packet content (v0.5
            # schema). gaia-audio fallback path can read /shared/* if needed
            # so we also write a copy.
            local_path = str(session_dir / f"{chash}_{safe_name}")
            with open(local_path, "wb") as out:
                out.write(data)
            audio_payloads_meta.append({
                "data_base64": _b64.b64encode(data).decode("ascii"),
                "mime_type": mime or "audio/wav",
                "filename": safe_name,
                "content_hash": chash,
                "encoding": "base64",
            })
            logger.info("Captured audio %s (%d bytes) → packet.audio_payloads + %s",
                        safe_name, len(data), local_path)

    payload = {
        "version": "v0.4",
        "header": {
            "session_id": session_id,
            "packet_id": packet_id,
            "persona": {"persona_id": "gaia", "role": "assistant"},
        },
        "content": {
            "original_prompt": redacted_input,
            "attachments": attachments_meta,
            "audio_payloads": audio_payloads_meta,
        },
        "governance": {
            "security_scan": {
                "ran": scan.ran,
                "passed": scan.passed,
                "injection_score": scan.injection_score,
            },
            "context_pool": context_pool,
        },
    }

    async def _stream():
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("POST", f"{core_url}/process_packet", json=payload) as resp:
                    if resp.status_code != 200:
                        yield json.dumps({"type": "error",
                                          "value": f"Core returned {resp.status_code}",
                                          "error_code": "GAIA-WEB-001"}) + "\n"
                        return
                    async for line in resp.aiter_lines():
                        if line:
                            yield line + "\n"
        except Exception as e:
            log_gaia_error(logger, "GAIA-WEB-031", str(e), exc_info=True)
            yield json.dumps({"type": "error", "value": str(e), "error_code": "GAIA-WEB-031"}) + "\n"

    return StreamingResponse(_stream(), media_type="application/x-ndjson")


@app.post("/presence")
async def update_presence(request: Request):
    """Update Discord bot presence from gaia-core sleep cycle.

    Called by gaia-core's _update_presence() in SOA mode (no direct Discord connector).
    Payload: {"activity": "sleeping...", "status": "idle"|"invisible"|"dnd"|"online"}
    """
    try:
        body = await request.json()
        activity = body.get("activity", "over the studio")
        status = body.get("status")

        from gaia_web.discord_interface import _bot, _bot_loop
        if _bot is None or not _bot.is_ready():
            return {"ok": False, "error": "Bot not connected"}
        if _bot_loop is None or _bot_loop.is_closed():
            return {"ok": False, "error": "Bot event loop not available"}

        import discord as _discord
        status_map = {"idle": _discord.Status.idle, "online": _discord.Status.online,
                      "dnd": _discord.Status.dnd, "invisible": _discord.Status.invisible}
        effective_status = status_map.get(status, _discord.Status.online)

        async def _change():
            if effective_status == _discord.Status.invisible:
                await _bot.change_presence(status=effective_status, activity=None)
            else:
                await _bot.change_presence(
                    status=effective_status,
                    activity=_discord.Activity(type=_discord.ActivityType.watching, name=activity),
                )

        # Fire-and-forget — don't block on the bot's congested event loop
        asyncio.run_coroutine_threadsafe(_change(), _bot_loop)
        return {"ok": True, "activity": activity, "status": status or "online"}
    except Exception as e:
        logger.warning("Presence update failed: %s: %s", type(e).__name__, e)
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@app.get("/health")
async def health_check():
    """System health check."""
    return {"status": "healthy", "service": "gaia-web", "timestamp": datetime.now(timezone.utc).isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("gaia_web.main:app", host="0.0.0.0", port=6414)

"""gaia-audio — GAIA sensory service for STT/TTS.

Half-duplex audio processing with GPU management, real-time status
streaming, and sleep-state governance integration.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from gaia_audio.config import AudioConfig
from gaia_audio.gpu_manager import GPUManager
from gaia_audio.models import (
    AudioEventResponse,
    AudioStatusResponse,
    HealthResponse,
    SynthesizeRequest,
    SynthesizeResponse,
    TranscribeRequest,
    TranscribeResponse,
    RefineRequest,
    RefineResponse,
    AnalyzeAudioRequest,
    AnalyzeAudioResponse,
    VoiceInfo,
)
from gaia_audio.status import status_tracker
from gaia_audio.stt_engine import STTEngine, audio_bytes_to_array
from gaia_audio.tts_engine import TTSEngine
from gaia_audio.refiner_engine import RefinerEngine
from gaia_audio.music_engine import MusicEngine

logger = logging.getLogger("GAIA.Audio")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ── Globals (initialized at startup) ─────────────────────────────────

config: AudioConfig | None = None
gpu_manager: GPUManager | None = None
stt_engine: STTEngine | None = None
tts_engine: TTSEngine | None = None
refiner_engine: RefinerEngine | None = None
music_engine: MusicEngine | None = None


# ── Lifecycle ─────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    global config, gpu_manager, stt_engine, tts_engine

    config = AudioConfig.from_constants()
    logger.info("Audio config loaded: stt=%s, tts=%s, vram_budget=%dMB",
                config.stt_model, config.tts_engine, config.vram_budget_mb)

    stt_engine = STTEngine(
        model_size=config.stt_model,
        device="cuda",
        compute_type="int8",
    )
    tts_engine = TTSEngine(
        engine_type=config.tts_engine,
        voice=config.tts_voice,
    )
    gpu_manager = GPUManager(
        stt_engine=stt_engine,
        tts_engine=tts_engine,
        vram_budget_mb=config.vram_budget_mb,
        half_duplex=config.half_duplex,
    )

    # Initialize Nano-Refiner (CPU model)
    nano_model_path = os.getenv("NANO_MODEL_PATH", "/models/nano_refiner.gguf")
    refiner_engine = RefinerEngine(model_path=nano_model_path)
    try:
        refiner_engine.load()
    except Exception as e:
        logger.error(f"Failed to load RefinerEngine: {e}")

    # Initialize MusicEngine (CPU models + librosa)
    music_engine = MusicEngine()
    try:
        music_engine.load()
    except Exception as e:
        logger.error(f"Failed to load MusicEngine: {e}")

    duplex_mode = "half-duplex" if config.half_duplex else "full-duplex"
    await status_tracker.emit("startup", f"gaia-audio ready (STT={config.stt_model}, TTS={config.tts_engine}, {duplex_mode})")

    # Register with orchestrator (best-effort)
    await _register_with_orchestrator()

    yield

    # Shutdown: release GPU resources
    logger.info("Shutting down gaia-audio — releasing GPU resources")
    if gpu_manager:
        await gpu_manager.release()
    await status_tracker.emit("shutdown", "gaia-audio stopped")


app = FastAPI(
    title="gaia-audio",
    description="GAIA sensory service — half-duplex STT/TTS with GPU management",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Health & Status ───────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness probe."""
    return HealthResponse()


@app.get("/status", response_model=AudioStatusResponse)
async def status():
    """Full status snapshot for dashboard polling."""
    snap = status_tracker.snapshot()
    return AudioStatusResponse(
        state=snap["state"],
        gpu_mode=snap["gpu_mode"],
        stt_model=snap.get("stt_model"),
        tts_engine=snap.get("tts_engine"),
        vram_used_mb=snap["vram_used_mb"],
        muted=snap["muted"],
        last_transcription=snap.get("last_transcription"),
        last_synthesis_text=snap.get("last_synthesis_text"),
        queue_depth=snap["queue_depth"],
        events=[
            AudioEventResponse(
                timestamp=e["timestamp"],
                event_type=e["event_type"],
                detail=e["detail"],
                latency_ms=e["latency_ms"],
            )
            for e in snap["events"][-20:]  # Last 20 events
        ],
    )


@app.websocket("/status/ws")
async def status_ws(websocket: WebSocket):
    """Real-time status stream via WebSocket for dashboard widget."""
    await websocket.accept()
    queue = status_tracker.subscribe()

    try:
        # Send initial snapshot
        await websocket.send_json(status_tracker.snapshot())

        # Stream events as they arrive
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json(event)
            except asyncio.TimeoutError:
                # Send keepalive ping
                await websocket.send_json({"event_type": "keepalive", "timestamp": ""})
    except WebSocketDisconnect:
        logger.debug("Dashboard WebSocket disconnected")
    except Exception:
        logger.debug("WebSocket error", exc_info=True)
    finally:
        status_tracker.unsubscribe(queue)


# ── Transcription (STT) ──────────────────────────────────────────────


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: TranscribeRequest):
    """Transcribe audio to text.

    Accepts base64-encoded audio (WAV, MP3, FLAC, OGG, or raw PCM).
    """
    if not gpu_manager or not stt_engine:
        raise HTTPException(503, "Audio service not initialized")

    if status_tracker.muted:
        raise HTTPException(423, "Audio service is muted (GAIA is sleeping)")

    if not request.audio_base64:
        raise HTTPException(400, "audio_base64 is required")

    t0 = time.monotonic()
    status_tracker.state = "transcribing"
    await status_tracker.emit("stt_start", f"Transcribing audio ({len(request.audio_base64)} chars base64)")

    try:
        # Decode audio
        audio_bytes = base64.b64decode(request.audio_base64)
        audio_array = audio_bytes_to_array(audio_bytes, request.sample_rate)

        # Run transcription with GPU management
        result = await gpu_manager.run_stt(
            stt_engine.transcribe_sync,
            audio_array=audio_array,
            sample_rate=request.sample_rate,
            language=request.language,
        )

        latency_ms = (time.monotonic() - t0) * 1000
        status_tracker.state = "idle"
        status_tracker.last_transcription = result["text"][:200]  # Truncate for status
        await status_tracker.emit("stt_complete", result["text"][:80], latency_ms)

        # Signal wake to gaia-core (voice activity detected)
        asyncio.create_task(_signal_wake())

        return TranscribeResponse(
            text=result["text"],
            language=result.get("language"),
            confidence=result.get("confidence", 0.0),
            duration_seconds=result.get("duration_seconds", 0.0),
            latency_ms=latency_ms,
            segments=result.get("segments", []),
            context_markers=result.get("context_markers", []),
        )

    except Exception as e:
        status_tracker.state = "idle"
        await status_tracker.emit("error", f"STT failed: {e}")
        raise HTTPException(500, f"Transcription failed: {e}") from e


# ── Audio Analysis (Music/Environment) ─────────────────────────────


@app.post("/analyze", response_model=AnalyzeAudioResponse)
async def analyze_audio(request: AnalyzeAudioRequest):
    """Perform deep musical and environmental analysis of audio."""
    if not music_engine:
        raise HTTPException(503, "Music engine not initialized")

    t0 = time.monotonic()
    status_tracker.state = "analyzing"
    await status_tracker.emit("analyze_start", f"Analyzing audio sample ({request.sample_rate}Hz)")

    try:
        # Convert base64 to numpy array
        audio_bytes = base64.b64decode(request.audio_base64)
        audio_array = audio_bytes_to_array(audio_bytes, request.sample_rate)

        # Run analysis in executor (CPU intensive)
        analysis_result = await asyncio.get_event_loop().run_in_executor(
            None,
            music_engine.analyze,
            audio_array,
            request.sample_rate
        )

        latency_ms = (time.monotonic() - t0) * 1000
        status_tracker.state = "idle"
        await status_tracker.emit("analyze_complete", f"BPM: {analysis_result['bpm']}, Key: {analysis_result['key']}", latency_ms)

        return AnalyzeAudioResponse(
            **analysis_result
        )

    except Exception as e:
        status_tracker.state = "idle"
        await status_tracker.emit("error", f"Analysis failed: {e}")
        logger.error("Analysis failed", exc_info=True)
        raise HTTPException(500, f"Analysis failed: {e}") from e


# ── Refinement (Nano LLM) ───────────────────────────────────────────


@app.post("/refine", response_model=RefineResponse)
async def refine(request: RefineRequest):
    """Clean up and format a transcript using the nano-refiner (CPU)."""
    global refiner_engine
    logger.info(f"Refine request received. Engine initialized: {refiner_engine is not None}")
    if not refiner_engine:
        # Emergency initialization if lifespan missed it or global state was lost
        nano_model_path = os.getenv("NANO_MODEL_PATH", "/models/nano_refiner.gguf")
        logger.warning(f"Refiner engine was None! Initializing on-demand from {nano_model_path}")
        refiner_engine = RefinerEngine(model_path=nano_model_path)
        refiner_engine.load()

    t0 = time.monotonic()
    status_tracker.state = "refining"
    await status_tracker.emit("refine_start", f"Refining text ({len(request.text)} chars)")

    try:
        # The refiner runs on CPU, so it doesn't need gpu_manager synchronization
        refined_text = await asyncio.get_event_loop().run_in_executor(
            None, 
            refiner_engine.refine, 
            request.text, 
            request.max_tokens
        )

        latency_ms = (time.monotonic() - t0) * 1000
        status_tracker.state = "idle"
        await status_tracker.emit("refine_complete", f"Refined {len(refined_text)} chars", latency_ms)

        return RefineResponse(
            refined_text=refined_text,
            latency_ms=latency_ms
        )

    except Exception as e:
        status_tracker.state = "idle"
        await status_tracker.emit("error", f"Refinement failed: {e}")
        raise HTTPException(500, f"Refinement failed: {e}") from e


# ── Synthesis (TTS) ───────────────────────────────────────────────────


@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    """Synthesize text to speech.

    Returns audio as base64 JSON or raw bytes (based on Accept header).
    """
    if not gpu_manager or not tts_engine:
        raise HTTPException(503, "Audio service not initialized")

    if not tts_engine.loaded:
        logger.info("TTS engine not loaded; loading on-demand")
        await gpu_manager.run_tts(tts_engine.load)

    t0 = time.monotonic()
    status_tracker.state = "synthesizing"
    await status_tracker.emit("tts_start", request.text[:80])

    try:
        # Use requested engine override or default
        engine = tts_engine
        if request.engine and request.engine != tts_engine.engine_type:
            # TODO: support dynamic engine switching
            logger.warning("Engine override requested (%s) but not yet supported; using %s",
                           request.engine, tts_engine.engine_type)

        result = await gpu_manager.run_tts(
            engine.synthesize_sync,
            text=request.text,
            voice=request.voice,
        )

        latency_ms = (time.monotonic() - t0) * 1000
        status_tracker.state = "idle"
        status_tracker.last_synthesis_text = request.text[:200]
        await status_tracker.emit("tts_complete", request.text[:80], latency_ms)

        audio_b64 = base64.b64encode(result["audio_bytes"]).decode("ascii")

        return SynthesizeResponse(
            audio_base64=audio_b64,
            sample_rate=result.get("sample_rate", 22050),
            duration_seconds=result.get("duration_seconds", 0.0),
            latency_ms=latency_ms,
            engine_used=result.get("engine_used", tts_engine.engine_type),
        )

    except Exception as e:
        status_tracker.state = "idle"
        await status_tracker.emit("error", f"TTS failed: {e}")
        raise HTTPException(500, f"Synthesis failed: {e}") from e


# ── Voices ────────────────────────────────────────────────────────────


@app.get("/voices", response_model=list[VoiceInfo])
async def list_voices():
    """List available TTS voices."""
    if not tts_engine:
        return []
    raw = tts_engine.list_voices()
    return [VoiceInfo(**v) for v in raw]


# ── Config ────────────────────────────────────────────────────────────


@app.get("/config")
async def get_config():
    """Return current audio config (read-only)."""
    if not config:
        return {}
    return {
        "stt_model": config.stt_model,
        "tts_engine": config.tts_engine,
        "tts_voice": config.tts_voice,
        "sample_rate": config.sample_rate,
        "vram_budget_mb": config.vram_budget_mb,
        "half_duplex": config.half_duplex,
        "mute_on_sleep": config.mute_on_sleep,
    }


# ── Mute / Unmute (Sleep Governance) ──────────────────────────────────


@app.post("/mute")
async def mute():
    """Mute the audio service (called by gaia-core on sleep)."""
    status_tracker.muted = True
    status_tracker.state = "muted"
    await status_tracker.emit("mute", "Audio service muted (GAIA sleeping)")
    return {"status": "muted"}


@app.post("/unmute")
async def unmute():
    """Unmute the audio service (called by gaia-core on wake)."""
    status_tracker.muted = False
    status_tracker.state = "idle"
    await status_tracker.emit("unmute", "Audio service unmuted (GAIA awake)")
    return {"status": "unmuted"}


# ── Deep Sleep / Wake (GPU model unload/reload) ─────────────────────


@app.post("/sleep")
async def sleep_mode():
    """Deep sleep: mute + unload all GPU models to free VRAM.

    Called by gaia-core when entering ASLEEP/DREAMING states.
    Idempotent — safe to call multiple times.
    """
    status_tracker.muted = True
    status_tracker.state = "sleeping"
    await status_tracker.emit("sleep_start", "Entering deep sleep — unloading GPU models")

    if gpu_manager:
        await gpu_manager.release()

    await status_tracker.emit("sleep_complete", "GPU models unloaded — VRAM freed")
    logger.info("Audio deep sleep: models unloaded, VRAM freed")
    return {"status": "sleeping", "vram_freed": True}


@app.post("/wake")
async def wake_mode():
    """Wake from deep sleep: unmute + eagerly reload GPU models in background.

    Returns immediately — model reload runs as a background task (~11s)
    which is hidden behind Prime's ~37s cold-start.
    Idempotent — safe to call multiple times.
    """
    status_tracker.muted = False
    status_tracker.state = "waking"
    await status_tracker.emit("wake_start", "Waking — starting background model reload")
    logger.info("Audio wake: starting background model reload")

    if gpu_manager:
        asyncio.create_task(_background_reload())

    return {"status": "waking", "reload_started": True}


async def _background_reload():
    """Reload GPU models in background after wake.

    Non-fatal on failure — lazy loading will catch up on next request.
    """
    try:
        t0 = time.monotonic()
        if gpu_manager:
            await gpu_manager.acquire_for_stt()  # full-duplex: loads both
        elapsed = (time.monotonic() - t0) * 1000
        status_tracker.state = "idle"
        await status_tracker.emit("wake_complete", f"Models reloaded ({elapsed:.0f}ms)", elapsed)
        logger.info("Background model reload complete in %.0fms", elapsed)
    except Exception:
        logger.error("Background model reload failed (non-fatal)", exc_info=True)
        status_tracker.state = "idle"
        await status_tracker.emit("wake_error", "Background reload failed — will lazy-load on demand")


# ── Internal helpers ──────────────────────────────────────────────────


async def _signal_wake():
    """Signal gaia-core that voice activity was detected."""
    if not config:
        return
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(f"{config.core_endpoint}/sleep/wake")
    except Exception:
        logger.debug("Could not signal wake to gaia-core", exc_info=True)


async def _register_with_orchestrator():
    """Register gaia-audio with the orchestrator (best-effort)."""
    if not config:
        return
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"{config.orchestrator_endpoint}/register",
                json={
                    "service_id": "gaia-audio",
                    "endpoint": config.endpoint,
                    "health_endpoint": f"{config.endpoint}/health",
                    "capabilities": ["stt", "tts"],
                },
            )
        logger.info("Registered with orchestrator")
    except Exception:
        logger.debug("Could not register with orchestrator (non-fatal)", exc_info=True)

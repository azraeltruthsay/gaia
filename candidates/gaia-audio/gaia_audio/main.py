"""gaia-audio — GAIA sensory service for STT/TTS.

Three-tier audio architecture:
  - Listener:      Qwen3-ASR 0.6B (GPU, always-on, coexists with Prime LLM)
  - Nano Speaker:  Qwen3-TTS 0.6B (CPU, always-on, instant short phrases)
  - Prime Speaker: Qwen3-TTS 1.7B (GPU, on-demand, high-quality long-form)
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
from gaia_audio.tts_engine import NanoSpeaker, PrimeSpeaker, EspeakFallback
from gaia_audio.refiner_engine import RefinerEngine
from gaia_audio.music_engine import MusicEngine

logger = logging.getLogger("GAIA.Audio")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ── Globals (initialized at startup) ─────────────────────────────────

config: AudioConfig | None = None
gpu_manager: GPUManager | None = None
stt_engine: STTEngine | None = None
audio_queue = None
audio_watcher = None
nano_speaker: NanoSpeaker | None = None
prime_speaker: PrimeSpeaker | None = None
espeak_fallback: EspeakFallback | None = None
refiner_engine: RefinerEngine | None = None
music_engine: MusicEngine | None = None


# ── Lifecycle ─────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    global config, gpu_manager, stt_engine, nano_speaker, prime_speaker
    global espeak_fallback, refiner_engine, music_engine

    config = AudioConfig.from_constants()
    logger.info("Audio config loaded: listener=%s, nano_speaker=%s, prime_speaker=%s",
                config.listener_model_path, config.nano_speaker_model_path,
                config.prime_speaker_model_path)

    # Initialize three-tier engines
    stt_engine = STTEngine(
        model_path=config.listener_model_path,
        device=config.listener_device,
    )
    nano_speaker = NanoSpeaker(
        model_path=config.nano_speaker_model_path,
        voice_ref_audio=config.voice_ref_audio,
        voice_ref_text=config.voice_ref_text,
    )
    prime_speaker = PrimeSpeaker(
        model_path=config.prime_speaker_model_path,
        voice_ref_audio=config.voice_ref_audio,
        voice_ref_text=config.voice_ref_text,
    )
    espeak_fallback = EspeakFallback()

    gpu_manager = GPUManager(
        stt_engine=stt_engine,
        nano_speaker=nano_speaker,
        prime_speaker=prime_speaker,
        espeak_fallback=espeak_fallback,
        prime_speaker_timeout=config.prime_speaker_timeout,
    )

    # Boot: load Listener (GPU) + Nano Speaker (CPU)
    await gpu_manager.startup()

    # Initialize Nano-Refiner (via gaia-nano HTTP endpoint)
    nano_endpoint = os.getenv("NANO_ENDPOINT", "http://gaia-nano:8080")
    refiner_engine = RefinerEngine(endpoint=nano_endpoint)
    try:
        refiner_engine.load()
    except Exception as e:
        logger.error(f"Failed to connect to Nano-Refiner: {e}")

    # Initialize MusicEngine (CPU models + librosa)
    music_engine = MusicEngine()
    try:
        music_engine.load()
    except Exception as e:
        logger.error(f"Failed to load MusicEngine: {e}")

    await status_tracker.emit("startup", "gaia-audio ready (three-tier Qwen3 architecture)")

    # Initialize audio processing queue with readiness gate
    global audio_queue, audio_watcher
    from gaia_audio.audio_queue import AudioProcessingQueue, AudioFileWatcher

    audio_queue = AudioProcessingQueue()
    audio_queue.set_ready_checker(lambda: stt_engine is not None and stt_engine.loaded)
    if stt_engine and stt_engine.loaded:
        audio_queue.signal_model_ready()

    # Watch /shared/audio_queue/incoming/ for new audio files
    audio_watcher = AudioFileWatcher(
        watch_dir="/shared/audio_queue/incoming",
        queue=audio_queue,
        source="listener",
        priority=5,
        poll_interval=2.0,
    )
    await audio_watcher.start()
    await audio_queue.start_processing()
    logger.info("Audio queue + file watcher started")

    # Register with orchestrator (best-effort)
    await _register_with_orchestrator()

    yield

    # Shutdown: release all resources
    logger.info("Shutting down gaia-audio — releasing all resources")
    if audio_queue:
        await audio_queue.stop_processing()
    if audio_watcher:
        await audio_watcher.stop()
    if gpu_manager:
        await gpu_manager.release()
    await status_tracker.emit("shutdown", "gaia-audio stopped")


app = FastAPI(
    title="gaia-audio",
    description="GAIA sensory service — three-tier Qwen3 STT/TTS",
    version="0.2.0",
    lifespan=lifespan,
)

# Inter-service HMAC authentication
try:
    from gaia_common.utils.service_auth import AuthMiddleware
    if AuthMiddleware:
        app.add_middleware(AuthMiddleware)
except ImportError:
    pass


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
            for e in snap["events"][-20:]
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
    """Transcribe audio to text using Qwen3-ASR (Listener).

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
        audio_bytes = base64.b64decode(request.audio_base64)
        audio_array = audio_bytes_to_array(audio_bytes, request.sample_rate)

        result = await gpu_manager.run_stt(
            stt_engine.transcribe_sync,
            audio_array=audio_array,
            sample_rate=request.sample_rate,
            language=request.language,
        )

        latency_ms = (time.monotonic() - t0) * 1000
        status_tracker.state = "idle"
        status_tracker.last_transcription = result["text"][:200]
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
        audio_bytes = base64.b64decode(request.audio_base64)
        audio_array = audio_bytes_to_array(audio_bytes, request.sample_rate)

        analysis_result = await asyncio.get_event_loop().run_in_executor(
            None,
            music_engine.analyze,
            audio_array,
            request.sample_rate
        )

        latency_ms = (time.monotonic() - t0) * 1000
        status_tracker.state = "idle"
        await status_tracker.emit("analyze_complete", f"BPM: {analysis_result['bpm']}, Key: {analysis_result['key']}", latency_ms)

        return AnalyzeAudioResponse(**analysis_result)

    except Exception as e:
        status_tracker.state = "idle"
        await status_tracker.emit("error", f"Analysis failed: {e}")
        logger.error("Analysis failed", exc_info=True)
        raise HTTPException(500, f"Analysis failed: {e}") from e


# ── Refinement (Nano LLM) ───────────────────────────────────────────


@app.post("/refine", response_model=RefineResponse)
async def refine(request: RefineRequest):
    """Clean up and format a transcript using the nano-refiner."""
    global refiner_engine
    logger.info(f"Refine request received. Engine initialized: {refiner_engine is not None}")
    if not refiner_engine:
        nano_endpoint = os.getenv("NANO_ENDPOINT", "http://gaia-nano:8080")
        logger.warning(f"Refiner engine was None! Initializing on-demand from {nano_endpoint}")
        refiner_engine = RefinerEngine(endpoint=nano_endpoint)
        refiner_engine.load()

    t0 = time.monotonic()
    status_tracker.state = "refining"
    await status_tracker.emit("refine_start", f"Refining text ({len(request.text)} chars)")

    try:
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


# ── Synthesis (TTS) — Three-Tier Routing ─────────────────────────────


@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    """Synthesize text to speech with three-tier routing.

    Tier routing:
      - auto:  len(text) ≤ threshold → Nano, else → Prime (Nano fallback)
      - nano:  force CPU (Qwen3-TTS 0.6B)
      - prime: force GPU (Qwen3-TTS 1.7B), 503 if unavailable after timeout
    """
    if not gpu_manager:
        raise HTTPException(503, "Audio service not initialized")

    t0 = time.monotonic()
    status_tracker.state = "synthesizing"
    await status_tracker.emit("tts_start", request.text[:80])

    try:
        tier = request.tier
        threshold = config.tts_auto_threshold if config else 200
        result = None
        tier_used = "nano"

        if tier == "prime" or (tier == "auto" and len(request.text) > threshold):
            # Try Prime Speaker (GPU)
            result = await gpu_manager.run_tts_prime(
                prime_speaker.synthesize_sync,
                text=request.text,
                voice=request.voice,
            )
            if result is not None:
                tier_used = "prime"

        if result is None:
            # Use Nano Speaker (CPU) — either by request, auto-routing, or Prime fallback
            if nano_speaker and nano_speaker.loaded:
                result = await gpu_manager.run_tts_nano(
                    nano_speaker.synthesize_sync,
                    text=request.text,
                    voice=request.voice,
                )
                tier_used = "nano"
            elif espeak_fallback and espeak_fallback.loaded:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    espeak_fallback.synthesize_sync,
                    request.text,
                    request.voice,
                )
                tier_used = "espeak"
            else:
                raise HTTPException(503, "No TTS engine available")

        latency_ms = (time.monotonic() - t0) * 1000
        status_tracker.state = "idle"
        status_tracker.last_synthesis_text = request.text[:200]
        await status_tracker.emit("tts_complete", f"{tier_used}: {request.text[:60]}", latency_ms)

        audio_b64 = base64.b64encode(result["audio_bytes"]).decode("ascii")

        return SynthesizeResponse(
            audio_base64=audio_b64,
            sample_rate=result.get("sample_rate", 24000),
            duration_seconds=result.get("duration_seconds", 0.0),
            latency_ms=latency_ms,
            engine_used=result.get("engine_used", "nano_speaker"),
            tier_used=tier_used,
        )

    except HTTPException:
        raise
    except Exception as e:
        status_tracker.state = "idle"
        await status_tracker.emit("error", f"TTS failed: {e}")
        raise HTTPException(500, f"Synthesis failed: {e}") from e


# ── Voices ────────────────────────────────────────────────────────────


@app.get("/voices", response_model=list[VoiceInfo])
async def list_voices():
    """List available TTS voices across all tiers."""
    voices = []
    if nano_speaker:
        voices.extend([VoiceInfo(**v) for v in nano_speaker.list_voices()])
    if prime_speaker:
        voices.extend([VoiceInfo(**v) for v in prime_speaker.list_voices()])
    if espeak_fallback and espeak_fallback.loaded:
        voices.extend([VoiceInfo(**v) for v in espeak_fallback.list_voices()])
    return voices


# ── Config ────────────────────────────────────────────────────────────


@app.get("/config")
async def get_config():
    """Return current audio config (read-only)."""
    if not config:
        return {}
    return {
        "listener_model": config.listener_model_path,
        "listener_device": config.listener_device,
        "nano_speaker_model": config.nano_speaker_model_path,
        "prime_speaker_model": config.prime_speaker_model_path,
        "voice_ref_audio": config.voice_ref_audio,
        "tts_auto_threshold": config.tts_auto_threshold,
        "prime_speaker_timeout": config.prime_speaker_timeout,
        "sample_rate": config.sample_rate,
        "vram_budget_mb": config.vram_budget_mb,
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
    """Deep sleep: mute + unload Listener from GPU. Nano Speaker stays on CPU.

    Called by gaia-core when entering ASLEEP/DREAMING states.
    Idempotent — safe to call multiple times.
    """
    status_tracker.muted = True
    status_tracker.state = "sleeping"
    await status_tracker.emit("sleep_start", "Entering deep sleep — unloading GPU models")

    if gpu_manager:
        await gpu_manager.sleep()

    await status_tracker.emit("sleep_complete", "GPU models unloaded — Nano Speaker still on CPU")
    logger.info("Audio deep sleep: Listener unloaded, Nano Speaker available on CPU")
    return {"status": "sleeping", "vram_freed": True, "nano_available": True}


@app.post("/wake")
async def wake_mode():
    """Wake from deep sleep: unmute + reload Listener on GPU in background.

    Returns immediately — model reload runs as a background task.
    Idempotent — safe to call multiple times.
    """
    status_tracker.muted = False
    status_tracker.state = "waking"
    await status_tracker.emit("wake_start", "Waking — starting background Listener reload")
    logger.info("Audio wake: starting background Listener reload")

    if gpu_manager:
        asyncio.create_task(_background_wake())

    return {"status": "waking", "reload_started": True}


async def _background_wake():
    """Reload Listener on GPU in background after wake."""
    try:
        t0 = time.monotonic()
        if gpu_manager:
            await gpu_manager.wake()
        elapsed = (time.monotonic() - t0) * 1000
        status_tracker.state = "idle"
        await status_tracker.emit("wake_complete", f"Listener reloaded ({elapsed:.0f}ms)", elapsed)
        logger.info("Background Listener reload complete in %.0fms", elapsed)
    except Exception:
        logger.error("Background Listener reload failed (non-fatal)", exc_info=True)
        status_tracker.state = "idle"
        await status_tracker.emit("wake_error", "Listener reload failed — will lazy-load on demand")


# ── GPU release/reclaim (for external GPU handoff) ───────────────────

@app.post("/gpu/release")
async def gpu_release():
    """Release GPU VRAM by unloading all GPU models.

    Unlike /sleep, this does NOT mute the service — STT will lazy-reload
    on the next /transcribe request when GPU is available again.
    Use this when another service needs the GPU temporarily.

    Pair with /gpu/reclaim to reload models afterward.
    """
    if not gpu_manager:
        return {"status": "no_gpu_manager", "vram_freed_mb": 0}

    vram_before = status_tracker.vram_used_mb
    await gpu_manager.release()
    vram_freed = vram_before - status_tracker.vram_used_mb
    logger.info("GPU released: freed %.0f MB VRAM", vram_freed)
    return {
        "status": "released",
        "vram_freed_mb": round(vram_freed),
        "models_unloaded": True,
        "muted": status_tracker.muted,  # mute state unchanged
    }


@app.post("/gpu/reclaim")
async def gpu_reclaim():
    """Reclaim GPU by reloading STT model in background.

    Returns immediately — model reload runs as a background task.
    Use after /gpu/release when the GPU is available again.
    """
    if not gpu_manager:
        return {"status": "no_gpu_manager"}

    asyncio.create_task(_background_gpu_reclaim())
    return {"status": "reclaiming", "reload_started": True}


async def _background_gpu_reclaim():
    """Reload STT model on GPU in background after release."""
    try:
        t0 = time.monotonic()
        if gpu_manager:
            await gpu_manager.wake()  # Reuses wake logic: loads Listener on GPU
        elapsed = (time.monotonic() - t0) * 1000
        await status_tracker.emit(
            "gpu_reclaim", f"STT model reloaded on GPU ({elapsed:.0f}ms)", elapsed
        )
        logger.info("GPU reclaimed: STT model reloaded in %.0fms", elapsed)
    except Exception:
        logger.error("GPU reclaim failed — STT will lazy-load on demand", exc_info=True)
        await status_tracker.emit("gpu_reclaim_error", "STT reload failed — will lazy-load")


@app.get("/gpu/status")
async def gpu_status():
    """Current GPU usage by the audio service."""
    return {
        "vram_used_mb": round(status_tracker.vram_used_mb),
        "stt_loaded": gpu_manager.stt.loaded if gpu_manager else False,
        "tts_nano_loaded": gpu_manager.nano.loaded if gpu_manager else False,
        "tts_prime_loaded": gpu_manager.prime.loaded if gpu_manager else False,
        "gpu_mode": status_tracker.gpu_mode,
        "muted": status_tracker.muted,
    }


# ── Audio Queue Endpoints ─────────────────────────────────────────────


@app.get("/queue/status")
async def queue_status():
    """Audio processing queue status."""
    if not audio_queue:
        return {"enabled": False}
    return {
        "enabled": True,
        "queue_size": audio_queue.queue_size,
        "is_processing": audio_queue.is_processing,
        "model_ready": stt_engine.loaded if stt_engine else False,
    }


@app.post("/queue/enqueue")
async def queue_enqueue(file_path: str, source: str = "upload", priority: int = 5):
    """Manually enqueue an audio file for processing."""
    if not audio_queue:
        raise HTTPException(503, "Audio queue not initialized")
    ok = audio_queue.enqueue_file(file_path, source=source, priority=priority)
    if not ok:
        raise HTTPException(429, "Queue full or file not found")
    return {"ok": True, "queue_size": audio_queue.queue_size}


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
                    "capabilities": ["stt", "tts", "tts_nano", "tts_prime"],
                },
            )
        logger.info("Registered with orchestrator")
    except Exception:
        logger.debug("Could not register with orchestrator (non-fatal)", exc_info=True)

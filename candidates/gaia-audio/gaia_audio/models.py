"""Pydantic request/response schemas for gaia-audio endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ── Request schemas ──────────────────────────────────────────────────


class TranscribeRequest(BaseModel):
    """Audio bytes for transcription (sent as base64 or multipart)."""

    audio_base64: str | None = Field(None, description="Base64-encoded audio data")
    sample_rate: int = Field(16000, description="Sample rate of the audio")
    language: str | None = Field(None, description="Language hint (e.g. 'en')")


class SynthesizeRequest(BaseModel):
    """Text to synthesize into speech."""

    text: str = Field(..., description="Text to convert to speech")
    voice: str | None = Field(None, description="Voice ID or name")
    engine: str | None = Field(None, description="Override TTS engine (system/coqui/elevenlabs)")
    sample_rate: int = Field(22050, description="Desired output sample rate")


# ── Response schemas ─────────────────────────────────────────────────


class TranscribeResponse(BaseModel):
    """Transcription result with optional context metadata."""

    text: str
    language: str | None = None
    confidence: float = 0.0
    duration_seconds: float = 0.0
    latency_ms: float = 0.0
    segments: list[dict] = Field(default_factory=list, description="Per-segment metadata (timing, confidence, words)")
    context_markers: list[str] = Field(default_factory=list, description="Derived context signals (pace, pauses, noise)")


class SynthesizeResponse(BaseModel):
    """Synthesis result (audio returned separately as streaming bytes)."""

    audio_base64: str
    sample_rate: int = 22050
    duration_seconds: float = 0.0
    latency_ms: float = 0.0
    engine_used: str = "system"


class VoiceInfo(BaseModel):
    """Available voice metadata."""

    voice_id: str
    name: str
    engine: str
    language: str = "en"
    description: str = ""


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    service: str = "gaia-audio"
    version: str = "0.1.0"


class AudioStatusResponse(BaseModel):
    """Full status for dashboard."""

    state: str = "idle"
    gpu_mode: str = "idle"
    stt_model: str | None = None
    tts_engine: str | None = None
    vram_used_mb: float = 0.0
    muted: bool = False
    last_transcription: str | None = None
    last_synthesis_text: str | None = None
    queue_depth: int = 0
    events: list[AudioEventResponse] = []


class AudioEventResponse(BaseModel):
    """Single audio processing event for dashboard stream."""

    timestamp: str
    event_type: str
    detail: str = ""
    latency_ms: float = 0.0

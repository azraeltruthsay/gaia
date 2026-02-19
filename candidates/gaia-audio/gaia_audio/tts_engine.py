"""Text-to-Speech engine wrapping RealtimeTTS backends.

Supports multiple engines with automatic fallback:
  - system:     espeak-ng via pyttsx3/RealtimeTTS SystemEngine (zero VRAM)
  - coqui:      XTTS v2 via RealtimeTTS CoquiEngine (~2-4GB VRAM)
  - elevenlabs: ElevenLabs API via RealtimeTTS (zero VRAM, requires API key)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np

from gaia_audio.status import status_tracker

logger = logging.getLogger("GAIA.Audio.TTS")

# Approximate VRAM usage by engine
ENGINE_VRAM_MB: dict[str, int] = {
    "system": 0,
    "coqui": 3000,
    "elevenlabs": 0,
}


class TTSEngine:
    """Text-to-speech with lazy loading and engine swapping."""

    def __init__(
        self,
        engine_type: str = "system",
        voice: str | None = None,
        cloud_api_key_env: str = "ELEVENLABS_API_KEY",
    ) -> None:
        self.engine_type = engine_type
        self.voice = voice
        self.cloud_api_key_env = cloud_api_key_env
        self._engine: Any = None
        self._stream: Any = None

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    @property
    def vram_mb(self) -> int:
        return ENGINE_VRAM_MB.get(self.engine_type, 0)

    def load(self) -> None:
        """Load the TTS engine. Called by GPUManager."""
        if self._engine is not None:
            return

        logger.info("Loading TTS engine: %s", self.engine_type)
        t0 = time.monotonic()

        try:
            if self.engine_type == "system":
                self._load_system()
            elif self.engine_type == "coqui":
                self._load_coqui()
            elif self.engine_type == "elevenlabs":
                self._load_elevenlabs()
            else:
                raise ValueError(f"Unknown TTS engine type: {self.engine_type}")

            elapsed = (time.monotonic() - t0) * 1000
            logger.info("TTS engine %s loaded in %.0fms", self.engine_type, elapsed)
            status_tracker.tts_engine = self.engine_type
        except Exception:
            logger.error("Failed to load TTS engine %s", self.engine_type, exc_info=True)
            self._engine = None
            raise

    def _load_system(self) -> None:
        """Load espeak-ng system engine (zero VRAM)."""
        from RealtimeTTS import SystemEngine, TextToAudioStream

        self._engine = SystemEngine()
        self._stream = TextToAudioStream(self._engine)

    def _load_coqui(self) -> None:
        """Load Coqui XTTS engine (GPU, ~2-4GB VRAM)."""
        from RealtimeTTS import CoquiEngine, TextToAudioStream

        self._engine = CoquiEngine()
        self._stream = TextToAudioStream(self._engine)

    def _load_elevenlabs(self) -> None:
        """Load ElevenLabs cloud TTS engine."""
        api_key = os.environ.get(self.cloud_api_key_env)
        if not api_key:
            raise RuntimeError(f"ElevenLabs API key not found in env var {self.cloud_api_key_env}")
        from RealtimeTTS import ElevenlabsEngine, TextToAudioStream

        self._engine = ElevenlabsEngine(api_key=api_key)
        self._stream = TextToAudioStream(self._engine)

    def unload(self) -> None:
        """Free resources. Called by GPUManager before STT load."""
        if self._engine is None:
            return
        logger.info("Unloading TTS engine: %s", self.engine_type)
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream = None
            self._engine = None
        except Exception:
            logger.debug("TTS unload cleanup error", exc_info=True)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        status_tracker.tts_engine = None

    def synthesize_sync(self, text: str, voice: str | None = None) -> dict:
        """Synchronous synthesis — run in executor for async context.

        Args:
            text: The text to synthesize.
            voice: Optional voice override.

        Returns:
            Dict with keys: audio_bytes, sample_rate, duration_seconds, engine_used.
        """
        if self._engine is None:
            raise RuntimeError("TTS engine not loaded — call load() first")

        # Use the TextToAudioStream to generate audio to a buffer
        audio_chunks: list[bytes] = []

        def on_audio_chunk(chunk: bytes) -> None:
            audio_chunks.append(chunk)

        try:
            # RealtimeTTS generates audio chunks
            if self._stream is not None:
                self._stream.feed(text)
                self._stream.play(
                    on_audio_chunk=on_audio_chunk,
                    muted=True,  # Don't play locally, just capture chunks
                )

            if audio_chunks:
                audio_bytes = b"".join(audio_chunks)
            else:
                # Fallback: generate silence placeholder
                logger.warning("TTS produced no audio; generating silence")
                sample_rate = 22050
                duration = max(0.5, len(text) * 0.06)  # Rough estimate
                samples = int(sample_rate * duration)
                silence = np.zeros(samples, dtype=np.int16)
                audio_bytes = silence.tobytes()

            # Estimate duration from audio bytes (16-bit PCM at 22050 Hz)
            sample_rate = 22050
            duration_seconds = len(audio_bytes) / (sample_rate * 2)  # 2 bytes per sample

            return {
                "audio_bytes": audio_bytes,
                "sample_rate": sample_rate,
                "duration_seconds": duration_seconds,
                "engine_used": self.engine_type,
            }
        except Exception:
            logger.error("TTS synthesis failed", exc_info=True)
            raise

    def list_voices(self) -> list[dict]:
        """List available voices for the current engine."""
        voices = []
        if self.engine_type == "system":
            voices.append({
                "voice_id": "default",
                "name": "System Default",
                "engine": "system",
                "language": "en",
                "description": "espeak-ng default voice",
            })
        elif self.engine_type == "coqui":
            voices.append({
                "voice_id": "default",
                "name": "XTTS Default",
                "engine": "coqui",
                "language": "en",
                "description": "Coqui XTTS v2 default voice",
            })
        elif self.engine_type == "elevenlabs":
            voices.append({
                "voice_id": "default",
                "name": "ElevenLabs Default",
                "engine": "elevenlabs",
                "language": "en",
                "description": "ElevenLabs default voice (API)",
            })
        return voices

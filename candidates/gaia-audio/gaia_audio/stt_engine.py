"""Speech-to-Text engine wrapping faster-whisper.

Uses the same backend as RealtimeSTT but provides a clean async interface
for HTTP-based audio transcription (microservice mode).
"""

from __future__ import annotations

import io
import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from gaia_audio.status import status_tracker

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger("GAIA.Audio.STT")

# VRAM estimates by model size (approximate, int8 quantized)
MODEL_VRAM_MB: dict[str, int] = {
    "tiny": 75,
    "tiny.en": 75,
    "base": 150,
    "base.en": 150,
    "small": 500,
    "small.en": 500,
    "medium": 1500,
    "medium.en": 1500,
    "large-v3": 3000,
}


class STTEngine:
    """Whisper-based speech-to-text with lazy GPU loading."""

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cuda",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: WhisperModel | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def vram_mb(self) -> int:
        """Estimated VRAM usage when loaded."""
        return MODEL_VRAM_MB.get(self.model_size, 200)

    def load(self) -> None:
        """Load Whisper model onto device. Called by GPUManager."""
        if self._model is not None:
            return
        logger.info("Loading Whisper model %s on %s (%s)", self.model_size, self.device, self.compute_type)
        t0 = time.monotonic()
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            elapsed = (time.monotonic() - t0) * 1000
            logger.info("Whisper model loaded in %.0fms", elapsed)
            status_tracker.stt_model = self.model_size
        except Exception:
            logger.error("Failed to load Whisper model %s", self.model_size, exc_info=True)
            self._model = None
            raise

    def unload(self) -> None:
        """Free GPU memory. Called by GPUManager before TTS load."""
        if self._model is None:
            return
        logger.info("Unloading Whisper model %s", self.model_size)
        del self._model
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        status_tracker.stt_model = None

    def transcribe_sync(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> dict:
        """Synchronous transcription — run in executor for async context.

        Args:
            audio_array: Float32 numpy array of audio samples, mono.
            sample_rate: Sample rate of the input audio.
            language: Optional language hint.

        Returns:
            Dict with keys: text, language, confidence, duration_seconds.
        """
        if self._model is None:
            raise RuntimeError("STT model not loaded — call load() first")

        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            try:
                from scipy.signal import resample

                num_samples = int(len(audio_array) * 16000 / sample_rate)
                audio_array = resample(audio_array, num_samples).astype(np.float32)
            except ImportError:
                logger.warning("scipy not available for resampling; passing raw audio")

        duration_seconds = len(audio_array) / 16000.0

        segments, info = self._model.transcribe(
            audio_array,
            language=language,
            beam_size=5,
            vad_filter=True,
        )

        # Collect all segment text
        texts = []
        total_confidence = 0.0
        n_segments = 0
        for segment in segments:
            texts.append(segment.text.strip())
            total_confidence += segment.avg_logprob
            n_segments += 1

        text = " ".join(texts)
        avg_confidence = (total_confidence / n_segments) if n_segments > 0 else 0.0

        return {
            "text": text,
            "language": info.language,
            "confidence": avg_confidence,
            "duration_seconds": duration_seconds,
        }


def audio_bytes_to_array(audio_bytes: bytes, sample_rate: int = 16000) -> np.ndarray:
    """Convert raw audio bytes to float32 numpy array.

    Supports WAV, MP3, FLAC, OGG via soundfile.
    Falls back to raw int16 PCM interpretation.
    """
    try:
        import soundfile as sf

        audio_array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)  # Stereo to mono
        return audio_array
    except Exception:
        # Fallback: assume raw int16 PCM
        logger.debug("soundfile failed; treating as raw int16 PCM")
        raw = np.frombuffer(audio_bytes, dtype=np.int16)
        return raw.astype(np.float32) / 32768.0

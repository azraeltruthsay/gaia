"""Speech-to-Text engine wrapping Qwen3-ASR.

Uses the qwen_asr package for high-quality multilingual transcription
with streaming/offline unified inference.
"""

from __future__ import annotations

import io
import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from gaia_audio.status import status_tracker

logger = logging.getLogger("GAIA.Audio.STT")


class STTEngine:
    """Qwen3-ASR speech-to-text with lazy loading."""

    def __init__(
        self,
        model_path: str = "/models/Qwen3-ASR-0.6B",
        device: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.device = device
        self._model = None
        self._on_gpu: bool = False

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def vram_mb(self) -> int:
        """Estimated VRAM usage when loaded on GPU."""
        return 1800 if self._on_gpu else 0

    def load(self) -> None:
        """Load Qwen3-ASR model. Called by GPUManager."""
        if self._model is not None:
            return
        logger.info("Loading Qwen3-ASR from %s (device=%s)", self.model_path, self.device)
        t0 = time.monotonic()
        try:
            import torch
            from qwen_asr import Qwen3ASRModel

            # Determine device
            if self.device == "auto":
                device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
            else:
                device_map = self.device

            self._on_gpu = "cuda" in str(device_map)
            dtype = torch.bfloat16 if self._on_gpu else torch.float32

            self._model = Qwen3ASRModel.from_pretrained(
                self.model_path,
                dtype=dtype,
                device_map=device_map,
            )
            elapsed = (time.monotonic() - t0) * 1000
            logger.info("Qwen3-ASR loaded in %.0fms (device=%s)", elapsed, device_map)
            status_tracker.stt_model = "Qwen3-ASR-0.6B"
        except Exception:
            logger.error("Failed to load Qwen3-ASR from %s", self.model_path, exc_info=True)
            self._model = None
            raise

    def unload(self) -> None:
        """Free GPU memory. Called by GPUManager."""
        if self._model is None:
            return
        logger.info("Unloading Qwen3-ASR")
        del self._model
        self._model = None
        self._on_gpu = False
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
            Dict with keys: text, language, confidence, duration_seconds,
            segments, context_markers.
        """
        if self._model is None:
            raise RuntimeError("STT model not loaded — call load() first")

        duration_seconds = len(audio_array) / sample_rate

        # Qwen3-ASR accepts (np.ndarray, sample_rate) tuple or file path
        results = self._model.transcribe(
            audio=(audio_array, sample_rate),
            language=language,
        )

        # Build response from Qwen3-ASR results
        texts = []
        segment_details = []
        total_words = 0
        total_speech_duration = 0.0

        for result in results:
            text = result.text.strip()
            texts.append(text)

            # Word count estimate
            word_count = len(text.split()) if text else 0
            total_words += word_count

            seg_info = {
                "text": text,
                "language": getattr(result, "language", None),
            }

            # Extract timestamps if available
            if hasattr(result, "time_stamps") and result.time_stamps:
                stamps = result.time_stamps
                if stamps:
                    seg_start = stamps[0].get("start", 0.0) if isinstance(stamps[0], dict) else 0.0
                    seg_end = stamps[-1].get("end", duration_seconds) if isinstance(stamps[-1], dict) else duration_seconds
                    seg_info["start"] = round(seg_start, 2)
                    seg_info["end"] = round(seg_end, 2)
                    seg_duration = seg_end - seg_start
                    total_speech_duration += seg_duration
                    if seg_duration > 0:
                        seg_info["speaking_rate_wps"] = round(word_count / seg_duration, 1)

            segment_details.append(seg_info)

        full_text = " ".join(texts)
        detected_language = results[0].language if results else language

        # Derive context markers
        context_markers = _extract_context_markers(segment_details, total_words, total_speech_duration)

        return {
            "text": full_text,
            "language": detected_language,
            "confidence": 0.0,  # Qwen3-ASR doesn't expose logprob; 0.0 as neutral
            "duration_seconds": duration_seconds,
            "segments": segment_details,
            "context_markers": context_markers,
        }


def _extract_context_markers(
    segments: list[dict],
    total_words: int,
    total_speech_duration: float,
) -> list[str]:
    """Derive human-readable context markers from segment metadata."""
    markers = []

    if not segments:
        return markers

    # Overall speaking rate
    if total_speech_duration > 0:
        overall_wps = total_words / total_speech_duration
        if overall_wps > 4.0:
            markers.append("fast_speech")
        elif overall_wps < 1.5 and total_words > 0:
            markers.append("slow_speech")

    return markers


def _ffmpeg_to_wav(audio_bytes: bytes) -> bytes | None:
    """Transcode arbitrary audio to WAV via ffmpeg."""
    import subprocess

    try:
        result = subprocess.run(
            ["ffmpeg", "-i", "pipe:0", "-f", "wav", "-ar", "16000", "-ac", "1", "pipe:1"],
            input=audio_bytes,
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0 and len(result.stdout) > 44:
            return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.debug("ffmpeg transcode failed", exc_info=True)
    return None


def audio_bytes_to_array(audio_bytes: bytes, sample_rate: int = 16000) -> np.ndarray:
    """Convert raw audio bytes to float32 numpy array.

    Supports WAV, MP3, FLAC, OGG via soundfile.
    For other formats (m4a, AAC, opus, webm, etc.), transcodes via ffmpeg.
    Falls back to raw int16 PCM interpretation.
    """
    import soundfile as sf

    # Try soundfile first (handles WAV, MP3, FLAC, OGG natively)
    try:
        audio_array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        return audio_array
    except Exception:
        pass

    # Transcode via ffmpeg
    wav_bytes = _ffmpeg_to_wav(audio_bytes)
    if wav_bytes is not None:
        try:
            audio_array, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)
            logger.debug("Audio transcoded via ffmpeg (%d bytes → %d samples)", len(audio_bytes), len(audio_array))
            return audio_array
        except Exception:
            logger.debug("soundfile failed on ffmpeg output", exc_info=True)

    # Last resort: assume raw int16 PCM
    logger.debug("All decoders failed; treating as raw int16 PCM")
    raw = np.frombuffer(audio_bytes, dtype=np.int16)
    return raw.astype(np.float32) / 32768.0

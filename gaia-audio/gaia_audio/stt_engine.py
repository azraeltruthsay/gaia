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
            word_timestamps=True,
            suppress_tokens=[],  # Preserve non-speech markers ([Music], [Laughter], etc.)
        )

        # Collect segment text + rich metadata
        texts = []
        total_confidence = 0.0
        n_segments = 0
        segment_details = []
        prev_end = 0.0
        total_words = 0
        total_speech_duration = 0.0

        for segment in segments:
            texts.append(segment.text.strip())
            total_confidence += segment.avg_logprob
            n_segments += 1

            pause = segment.start - prev_end if prev_end > 0 else 0.0
            seg_duration = segment.end - segment.start

            # Word-level stats
            word_count = 0
            words_data = []
            if segment.words:
                word_count = len(segment.words)
                for w in segment.words:
                    words_data.append({
                        "word": w.word,
                        "start": round(w.start, 2),
                        "end": round(w.end, 2),
                        "probability": round(w.probability, 3),
                    })

            speaking_rate = (word_count / seg_duration) if seg_duration > 0 else 0.0
            total_words += word_count
            total_speech_duration += seg_duration

            seg_info = {
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip(),
                "no_speech_prob": round(segment.no_speech_prob, 3),
                "avg_logprob": round(segment.avg_logprob, 3),
                "compression_ratio": round(segment.compression_ratio, 3),
                "speaking_rate_wps": round(speaking_rate, 1),
            }
            if pause > 0.5:
                seg_info["pause_before"] = round(pause, 2)
            if words_data:
                seg_info["words"] = words_data
            segment_details.append(seg_info)

            prev_end = segment.end

        text = " ".join(texts)
        avg_confidence = (total_confidence / n_segments) if n_segments > 0 else 0.0

        # Derive context markers from segment metadata
        context_markers = _extract_context_markers(segment_details, total_words, total_speech_duration)

        return {
            "text": text,
            "language": info.language,
            "confidence": avg_confidence,
            "duration_seconds": duration_seconds,
            "segments": segment_details,
            "context_markers": context_markers,
        }


def _extract_context_markers(
    segments: list[dict],
    total_words: int,
    total_speech_duration: float,
) -> list[str]:
    """Derive human-readable context markers from segment metadata.

    Extracts signals like speaking pace, pauses, low-confidence speech,
    non-speech events, and noise without requiring additional models.
    """
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

    # Scan segments for notable patterns
    long_pauses = 0
    noise_segments = 0
    low_confidence_segments = 0

    for seg in segments:
        # Long pauses (> 2s)
        if seg.get("pause_before", 0) > 2.0:
            long_pauses += 1

        # High no_speech_prob → likely music, noise, or ambient sound
        if seg.get("no_speech_prob", 0) > 0.6:
            noise_segments += 1

        # Low avg_logprob → mumbling, whispering, or unclear speech
        if seg.get("avg_logprob", 0) < -1.0:
            low_confidence_segments += 1

        # High compression ratio → repetitive content (stuttering, loops)
        if seg.get("compression_ratio", 0) > 2.5:
            markers.append("repetitive_segment")

    if long_pauses > 0:
        markers.append(f"long_pauses:{long_pauses}")
    if noise_segments > 0:
        markers.append(f"non_speech_segments:{noise_segments}")
    if low_confidence_segments > 0:
        markers.append(f"unclear_speech:{low_confidence_segments}")

    return markers


def _ffmpeg_to_wav(audio_bytes: bytes) -> bytes | None:
    """Transcode arbitrary audio to WAV via ffmpeg.

    Handles m4a/AAC, opus, webm, and any other format ffmpeg supports.
    Returns WAV bytes on success, None on failure.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["ffmpeg", "-i", "pipe:0", "-f", "wav", "-ar", "16000", "-ac", "1", "pipe:1"],
            input=audio_bytes,
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0 and len(result.stdout) > 44:  # WAV header = 44 bytes
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
            audio_array = audio_array.mean(axis=1)  # Stereo to mono
        return audio_array
    except Exception:
        pass

    # Transcode via ffmpeg (handles m4a, AAC, opus, webm, etc.)
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

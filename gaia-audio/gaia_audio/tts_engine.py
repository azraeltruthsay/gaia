"""Text-to-Speech engines using Qwen3-TTS.

Three-tier architecture:
  - NanoSpeaker:    Qwen3-TTS 0.6B on CPU — always-on, instant short phrases
  - PrimeSpeaker:   Qwen3-TTS 1.7B on GPU — on-demand, high-quality long-form
  - EspeakFallback: espeak-ng subprocess — emergency fallback if both fail
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import time

import numpy as np

logger = logging.getLogger("GAIA.Audio.TTS")


def _sanitize_for_speech(text: str) -> str:
    """Strip markdown and structural artifacts that shouldn't be spoken."""
    # Remove ALL hashes
    text = text.replace("#", "")

    # Remove bold/italic markers
    text = text.replace("**", "").replace("__", "").replace("*", "").replace("_", "")

    # Convert numbered list items into pauses
    text = re.sub(r'^[ \t]*\d+\.\s+', '... ', text, flags=re.MULTILINE)

    # Clean up list markers
    text = re.sub(r'^[ \t]*[-*+]\s+', ', ', text, flags=re.MULTILINE)

    # Remove blockquote markers
    text = re.sub(r'^[ \t]*>\s*', '', text, flags=re.MULTILINE)

    # Remove model tags
    text = text.replace("[Prime]", "").replace("[Lite]", "").replace("[Observer]", "")

    # Ensure sentences end with punctuation
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if line and line[-1] not in ".!?,:;":
            line += ","
        lines.append(line)
    text = " ".join(lines)

    # Normalize whitespace
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\.\s*\.', '.', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


class NanoSpeaker:
    """Qwen3-TTS 0.6B on CPU — always-on, instant for short phrases."""

    def __init__(
        self,
        model_path: str | None = None,
        voice_ref_audio: str | None = None,
        voice_ref_text: str | None = None,
    ) -> None:
        if model_path is None:
            try:
                from gaia_audio.config import get_config
                model_path = get_config().nano_speaker_model_path
            except Exception:
                model_path = "/models/Qwen3-TTS-12Hz-0.6B-Base"
        self.model_path = model_path
        self.voice_ref_audio = voice_ref_audio
        self.voice_ref_text = voice_ref_text
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def vram_mb(self) -> int:
        return 0  # CPU only

    def load(self) -> None:
        """Load Qwen3-TTS 0.6B on CPU."""
        if self._model is not None:
            return
        logger.info("Loading NanoSpeaker (Qwen3-TTS 0.6B) on CPU from %s", self.model_path)
        t0 = time.monotonic()
        try:
            import torch
            from qwen_tts import Qwen3TTSModel

            self._model = Qwen3TTSModel.from_pretrained(
                self.model_path,
                device_map="cpu",
                dtype=torch.float32,
            )
            elapsed = (time.monotonic() - t0) * 1000
            logger.info("NanoSpeaker loaded in %.0fms", elapsed)
        except Exception:
            logger.error("Failed to load NanoSpeaker", exc_info=True)
            self._model = None
            raise

    def unload(self) -> None:
        """Free memory."""
        if self._model is None:
            return
        logger.info("Unloading NanoSpeaker")
        del self._model
        self._model = None

    def synthesize_sync(self, text: str, voice: str | None = None) -> dict:
        """Synthesize text on CPU. Run in executor for async context."""
        if self._model is None:
            raise RuntimeError("NanoSpeaker not loaded — call load() first")

        text = _sanitize_for_speech(text)
        if not text:
            raise ValueError("Text is empty after sanitization")

        ref_audio = voice or self.voice_ref_audio
        ref_text = self.voice_ref_text

        t0 = time.monotonic()

        # Resolve reference audio — check configured path, then shared fallback
        if not ref_audio or not os.path.isfile(ref_audio):
            ref_audio = os.environ.get("GAIA_VOICE_REF", "/shared/voice/gaia_identity.wav")
        if not ref_text:
            ref_text = "I am GAIA, a sovereign artificial intelligence. I was created by Azrael to be curious, truthful, and helpful."

        if ref_audio and os.path.isfile(ref_audio):
            wavs, sr = self._model.generate_voice_clone(
                text=text,
                language="English",
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
        else:
            raise RuntimeError(
                f"No voice reference audio found at {ref_audio}. "
                "Qwen3-TTS 0.6B requires a reference WAV for voice cloning. "
                "Generate one with: espeak-ng -v en+f3 'I am GAIA' -w /shared/voice/gaia_identity.wav"
            )

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug("NanoSpeaker synthesized %d chars in %.0fms", len(text), elapsed_ms)

        wav_array = wavs[0] if isinstance(wavs[0], np.ndarray) else np.array(wavs[0], dtype=np.float32)
        # Convert to 16-bit PCM
        pcm = (wav_array * 32767).clip(-32768, 32767).astype(np.int16)
        audio_bytes = pcm.tobytes()
        duration_seconds = len(pcm) / sr

        return {
            "audio_bytes": audio_bytes,
            "sample_rate": sr,
            "duration_seconds": duration_seconds,
            "engine_used": "nano_speaker",
        }

    def list_voices(self) -> list[dict]:
        """List available voices."""
        voices = [{"voice_id": "default", "name": "Qwen3-TTS Nano", "engine": "nano_speaker",
                    "language": "multilingual", "description": "Qwen3-TTS 0.6B (CPU)"}]
        if self.voice_ref_audio:
            voices.append({"voice_id": "clone", "name": "Voice Clone", "engine": "nano_speaker",
                           "language": "multilingual", "description": "Voice cloning via reference audio"})
        return voices


class PrimeSpeaker:
    """Qwen3-TTS 1.7B on GPU — on-demand, high-quality long-form."""

    def __init__(
        self,
        model_path: str | None = None,
        voice_ref_audio: str | None = None,
        voice_ref_text: str | None = None,
    ) -> None:
        if model_path is None:
            try:
                from gaia_audio.config import get_config
                model_path = get_config().prime_speaker_model_path
            except Exception:
                model_path = "/models/Qwen3-TTS-12Hz-1.7B-Base"
        self.model_path = model_path
        self.voice_ref_audio = voice_ref_audio
        self.voice_ref_text = voice_ref_text
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def vram_mb(self) -> int:
        return 4300 if self._model is not None else 0

    def load(self) -> None:
        """Load Qwen3-TTS 1.7B on GPU."""
        if self._model is not None:
            return
        logger.info("Loading PrimeSpeaker (Qwen3-TTS 1.7B) on GPU from %s", self.model_path)
        t0 = time.monotonic()
        try:
            import torch
            from qwen_tts import Qwen3TTSModel

            self._model = Qwen3TTSModel.from_pretrained(
                self.model_path,
                device_map="cuda:0",
                dtype=torch.bfloat16,
            )
            elapsed = (time.monotonic() - t0) * 1000
            logger.info("PrimeSpeaker loaded in %.0fms", elapsed)
        except Exception:
            logger.error("Failed to load PrimeSpeaker", exc_info=True)
            self._model = None
            raise

    def unload(self) -> None:
        """Free GPU memory."""
        if self._model is None:
            return
        logger.info("Unloading PrimeSpeaker")
        del self._model
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def synthesize_sync(self, text: str, voice: str | None = None) -> dict:
        """Synthesize text on GPU. Run in executor for async context."""
        if self._model is None:
            raise RuntimeError("PrimeSpeaker not loaded — call load() first")

        text = _sanitize_for_speech(text)
        if not text:
            raise ValueError("Text is empty after sanitization")

        ref_audio = voice or self.voice_ref_audio
        ref_text = self.voice_ref_text

        t0 = time.monotonic()

        if ref_audio and os.path.isfile(ref_audio):
            wavs, sr = self._model.generate_voice_clone(
                text=text,
                language="English",
                ref_audio=ref_audio,
                ref_text=ref_text or "",
            )
        else:
            wavs, sr = self._model.generate_voice_clone(
                text=text,
                language="English",
                ref_audio=ref_audio or "",
                ref_text=ref_text or "",
            )

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug("PrimeSpeaker synthesized %d chars in %.0fms", len(text), elapsed_ms)

        wav_array = wavs[0] if isinstance(wavs[0], np.ndarray) else np.array(wavs[0], dtype=np.float32)
        pcm = (wav_array * 32767).clip(-32768, 32767).astype(np.int16)
        audio_bytes = pcm.tobytes()
        duration_seconds = len(pcm) / sr

        return {
            "audio_bytes": audio_bytes,
            "sample_rate": sr,
            "duration_seconds": duration_seconds,
            "engine_used": "prime_speaker",
        }

    def list_voices(self) -> list[dict]:
        """List available voices."""
        voices = [{"voice_id": "default", "name": "Qwen3-TTS Prime", "engine": "prime_speaker",
                    "language": "multilingual", "description": "Qwen3-TTS 1.7B (GPU)"}]
        if self.voice_ref_audio:
            voices.append({"voice_id": "clone", "name": "Voice Clone", "engine": "prime_speaker",
                           "language": "multilingual", "description": "Voice cloning via reference audio"})
        return voices


class EspeakFallback:
    """espeak-ng subprocess — emergency fallback if Qwen3-TTS fails."""

    def __init__(self) -> None:
        self._ready: bool = False

    @property
    def loaded(self) -> bool:
        return self._ready

    @property
    def vram_mb(self) -> int:
        return 0

    def load(self) -> None:
        """Verify espeak-ng is available."""
        if self._ready:
            return
        try:
            subprocess.run(["espeak-ng", "--version"], capture_output=True, check=True)
            self._ready = True
            logger.info("EspeakFallback ready")
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError("espeak-ng not found in container") from exc

    def unload(self) -> None:
        """No-op."""
        self._ready = False

    def synthesize_sync(self, text: str, voice: str | None = None) -> dict:
        """Generate audio via espeak-ng subprocess."""
        if not self._ready:
            raise RuntimeError("EspeakFallback not ready — call load() first")

        text = _sanitize_for_speech(text)
        if not text:
            raise ValueError("Text is empty after sanitization")

        sample_rate = 22050
        espeak_voice = voice or "en"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            subprocess.run(
                ["espeak-ng", "-v", espeak_voice, "-w", tmp_path, "--", text],
                capture_output=True, check=True, timeout=30,
            )
            import soundfile as sf
            data, file_sr = sf.read(tmp_path, dtype="int16")
            audio_bytes = data.tobytes()
            duration_seconds = len(data) / file_sr

            if file_sr != sample_rate:
                from scipy.signal import resample
                num_samples = int(len(data) * sample_rate / file_sr)
                data = resample(data, num_samples).astype(np.int16)
                audio_bytes = data.tobytes()
                duration_seconds = len(data) / sample_rate

            return {
                "audio_bytes": audio_bytes,
                "sample_rate": sample_rate,
                "duration_seconds": duration_seconds,
                "engine_used": "espeak_fallback",
            }
        finally:
            os.unlink(tmp_path)

    def list_voices(self) -> list[dict]:
        """List available voices."""
        return [{"voice_id": "default", "name": "System Default", "engine": "espeak_fallback",
                 "language": "en", "description": "espeak-ng emergency fallback"}]

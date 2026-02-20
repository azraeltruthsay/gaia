"""Text-to-Speech engine wrapping RealtimeTTS backends.

Supports multiple engines with automatic fallback:
  - system:     espeak-ng via subprocess (zero VRAM, headless-safe)
  - coqui:      XTTS v2 via Coqui TTS library (~2-4GB VRAM, headless-safe)
  - elevenlabs: ElevenLabs API via RealtimeTTS (zero VRAM, requires API key)
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
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
        """Load espeak-ng system engine (zero VRAM, headless-safe).

        Uses espeak-ng directly via subprocess — no PortAudio device needed.
        """
        # Verify espeak-ng is available
        try:
            subprocess.run(["espeak-ng", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError("espeak-ng not found in container") from exc
        self._engine = "espeak-ng"  # Sentinel — synthesis handled in synthesize_sync

    def _load_coqui(self) -> None:
        """Load Coqui XTTS v2 engine (GPU, ~2-4GB VRAM, headless-safe).

        Uses the TTS library directly — no PortAudio or playback device needed.
        The model downloads from HuggingFace on first load (~1.8GB).
        """
        import torch

        # PyTorch >=2.6 defaults weights_only=True which breaks TTS model loading.
        # Temporarily allow unsafe load for the trusted Coqui XTTS checkpoint.
        _orig_load = torch.load
        torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})

        try:
            from TTS.api import TTS

            use_gpu = torch.cuda.is_available()
            device = "cuda" if use_gpu else "cpu"
            logger.info("Loading XTTS v2 (device=%s)...", device)

            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            self._engine = tts
            self._coqui_sample_rate = 24000  # XTTS v2 native output rate
        finally:
            torch.load = _orig_load

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
            # Coqui TTS: move model off GPU before dropping reference
            if self.engine_type == "coqui" and hasattr(self._engine, "synthesizer"):
                try:
                    self._engine.synthesizer.tts_model.cpu()
                except Exception:
                    pass
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

        try:
            if self._engine == "espeak-ng":
                return self._synthesize_espeak(text, voice)
            if self.engine_type == "coqui":
                return self._synthesize_coqui(text, voice)
            return self._synthesize_realtimetts(text, voice)
        except Exception:
            logger.error("TTS synthesis failed", exc_info=True)
            raise

    def _synthesize_espeak(self, text: str, voice: str | None = None) -> dict:
        """Generate audio via espeak-ng subprocess (headless-safe)."""
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
            # Resample if needed
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
                "engine_used": "system",
            }
        finally:
            os.unlink(tmp_path)

    def _synthesize_coqui(self, text: str, voice: str | None = None) -> dict:
        """Generate audio via Coqui XTTS v2 (headless, GPU-accelerated).

        Args:
            voice: Path to a reference WAV file for voice cloning,
                   or None for the default XTTS voice.
        """
        from TTS.api import TTS

        tts: TTS = self._engine
        sample_rate = self._coqui_sample_rate  # 24000 Hz

        # XTTS v2 supports voice cloning from a reference audio file,
        # or one of the built-in speakers (e.g. "Claribel Dervla").
        speaker_wav = voice or self.voice  # path to reference WAV or speaker name
        kwargs: dict = {"text": text, "language": "en"}
        if speaker_wav and os.path.isfile(speaker_wav):
            kwargs["speaker_wav"] = speaker_wav
            logger.debug("Using voice reference: %s", speaker_wav)
        elif speaker_wav and hasattr(tts, "speakers") and speaker_wav in (tts.speakers or []):
            kwargs["speaker"] = speaker_wav
            logger.debug("Using built-in speaker: %s", speaker_wav)
        else:
            # Default to a built-in speaker when no voice reference is given
            kwargs["speaker"] = "Claribel Dervla"

        wav_list = tts.tts(**kwargs)

        # tts.tts() returns a list of floats in [-1, 1]
        wav_array = np.array(wav_list, dtype=np.float32)
        # Convert to 16-bit PCM
        pcm = (wav_array * 32767).clip(-32768, 32767).astype(np.int16)
        audio_bytes = pcm.tobytes()
        duration_seconds = len(pcm) / sample_rate

        return {
            "audio_bytes": audio_bytes,
            "sample_rate": sample_rate,
            "duration_seconds": duration_seconds,
            "engine_used": "coqui",
        }

    def _synthesize_realtimetts(self, text: str, voice: str | None = None) -> dict:
        """Generate audio via RealtimeTTS (requires audio device or muted mode)."""
        audio_chunks: list[bytes] = []

        def on_audio_chunk(chunk: bytes) -> None:
            audio_chunks.append(chunk)

        if self._stream is not None:
            self._stream.feed(text)
            self._stream.play(
                on_audio_chunk=on_audio_chunk,
                muted=True,
            )

        if audio_chunks:
            audio_bytes = b"".join(audio_chunks)
        else:
            logger.warning("TTS produced no audio; generating silence")
            sample_rate = 22050
            duration = max(0.5, len(text) * 0.06)
            samples = int(sample_rate * duration)
            silence = np.zeros(samples, dtype=np.int16)
            audio_bytes = silence.tobytes()

        sample_rate = 22050
        duration_seconds = len(audio_bytes) / (sample_rate * 2)

        return {
            "audio_bytes": audio_bytes,
            "sample_rate": sample_rate,
            "duration_seconds": duration_seconds,
            "engine_used": self.engine_type,
        }

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
            # List built-in XTTS speakers if model is loaded
            if self._engine is not None and hasattr(self._engine, "speakers") and self._engine.speakers:
                for speaker in self._engine.speakers:
                    voices.append({
                        "voice_id": speaker,
                        "name": speaker,
                        "engine": "coqui",
                        "language": "multilingual",
                        "description": f"XTTS v2 built-in speaker: {speaker}",
                    })
            else:
                voices.append({
                    "voice_id": "default",
                    "name": "Claribel Dervla",
                    "engine": "coqui",
                    "language": "multilingual",
                    "description": "Coqui XTTS v2 default speaker",
                })
            # Voice cloning option
            voices.append({
                "voice_id": "clone",
                "name": "Voice Clone",
                "engine": "coqui",
                "language": "multilingual",
                "description": "XTTS v2 voice cloning (provide speaker_wav path)",
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

"""Half-duplex GPU manager — ensures only one audio model at a time.

The RTX 5080 has ~5.6GB VRAM remaining after gaia-prime (vLLM). This manager
guarantees that Whisper (STT) and XTTS (TTS) never coexist in VRAM.

State transitions:
    idle  → stt   (load Whisper for transcription)
    idle  → tts   (load TTS engine for synthesis)
    stt   → tts   (unload Whisper, load TTS)
    tts   → stt   (unload TTS, load Whisper)
    any   → idle  (unload everything)
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from typing import Literal

from gaia_audio.status import status_tracker
from gaia_audio.stt_engine import STTEngine
from gaia_audio.tts_engine import TTSEngine

logger = logging.getLogger("GAIA.Audio.GPU")


class GPUManager:
    """Manages half-duplex GPU allocation between STT and TTS."""

    def __init__(
        self,
        stt_engine: STTEngine,
        tts_engine: TTSEngine,
        vram_budget_mb: int = 5600,
    ) -> None:
        self.stt = stt_engine
        self.tts = tts_engine
        self.vram_budget_mb = vram_budget_mb
        self.current_mode: Literal["idle", "stt", "tts"] = "idle"
        self._lock = asyncio.Lock()

    async def acquire_for_stt(self) -> None:
        """Ensure STT model is loaded, unloading TTS if necessary."""
        async with self._lock:
            if self.current_mode == "stt" and self.stt.loaded:
                return  # Already in STT mode

            t0 = time.monotonic()

            # Unload TTS if loaded
            if self.current_mode == "tts" and self.tts.loaded:
                logger.info("GPU swap: TTS → STT")
                await asyncio.get_event_loop().run_in_executor(None, self.tts.unload)
                await status_tracker.emit("gpu_swap", "TTS unloaded for STT")

            # Load STT
            if not self.stt.loaded:
                await asyncio.get_event_loop().run_in_executor(None, self.stt.load)

            self.current_mode = "stt"
            status_tracker.gpu_mode = "stt"
            status_tracker.vram_used_mb = float(self.stt.vram_mb)

            elapsed = (time.monotonic() - t0) * 1000
            await status_tracker.emit("gpu_acquire", f"STT mode active ({self.stt.model_size})", elapsed)

    async def acquire_for_tts(self) -> None:
        """Ensure TTS model is loaded, unloading STT if necessary."""
        async with self._lock:
            if self.current_mode == "tts" and self.tts.loaded:
                return  # Already in TTS mode

            t0 = time.monotonic()

            # Unload STT if loaded
            if self.current_mode == "stt" and self.stt.loaded:
                logger.info("GPU swap: STT → TTS")
                await asyncio.get_event_loop().run_in_executor(None, self.stt.unload)
                await status_tracker.emit("gpu_swap", "STT unloaded for TTS")

            # Load TTS (only if it uses VRAM — system engine doesn't need GPU swap)
            if not self.tts.loaded:
                await asyncio.get_event_loop().run_in_executor(None, self.tts.load)

            self.current_mode = "tts"
            status_tracker.gpu_mode = "tts"
            status_tracker.vram_used_mb = float(self.tts.vram_mb)

            elapsed = (time.monotonic() - t0) * 1000
            await status_tracker.emit("gpu_acquire", f"TTS mode active ({self.tts.engine_type})", elapsed)

    async def release(self) -> None:
        """Unload all models and free VRAM."""
        async with self._lock:
            if self.stt.loaded:
                await asyncio.get_event_loop().run_in_executor(None, self.stt.unload)
            if self.tts.loaded:
                await asyncio.get_event_loop().run_in_executor(None, self.tts.unload)
            self.current_mode = "idle"
            status_tracker.gpu_mode = "idle"
            status_tracker.vram_used_mb = 0.0
            await status_tracker.emit("gpu_release", "All audio models unloaded")

    async def run_stt(self, func, *args, **kwargs):
        """Acquire STT, run a function, return result."""
        await self.acquire_for_stt()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

    async def run_tts(self, func, *args, **kwargs):
        """Acquire TTS, run a function, return result."""
        await self.acquire_for_tts()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

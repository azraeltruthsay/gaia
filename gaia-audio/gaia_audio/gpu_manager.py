"""GPU manager for STT/TTS audio models.

Supports two modes controlled by the ``half_duplex`` flag:

  full-duplex (half_duplex=False, default):
      Both Whisper STT and XTTS TTS stay loaded on the GPU simultaneously.
      RTX 5080 VRAM budget: Prime ~11.5 GB + Whisper ~0.15 GB + XTTS ~1.8 GB
      = ~13.5 GB of 16 GB — comfortable.

  half-duplex (half_duplex=True):
      Only one audio model at a time.  Used when VRAM is tight (e.g. larger
      Whisper model or lower-VRAM GPU).

State transitions (half-duplex):
    idle  → stt   (load Whisper)
    idle  → tts   (load TTS)
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
    """Manages GPU allocation between STT and TTS."""

    def __init__(
        self,
        stt_engine: STTEngine,
        tts_engine: TTSEngine,
        vram_budget_mb: int = 5600,
        half_duplex: bool = False,
    ) -> None:
        self.stt = stt_engine
        self.tts = tts_engine
        self.vram_budget_mb = vram_budget_mb
        self.half_duplex = half_duplex
        self.current_mode: Literal["idle", "stt", "tts", "full"] = "idle"
        self._lock = asyncio.Lock()

    async def _ensure_loaded(self, engine, label: str) -> float:
        """Load an engine if not already loaded.  Returns load time in ms."""
        if engine.loaded:
            return 0.0
        t0 = time.monotonic()
        await asyncio.get_event_loop().run_in_executor(None, engine.load)
        return (time.monotonic() - t0) * 1000

    # ── Full-duplex helpers ───────────────────────────────────────────

    async def _ensure_both_loaded(self) -> None:
        """Load both STT and TTS if not already loaded (full-duplex).

        TTS load failures are non-fatal — STT will still work.
        """
        async with self._lock:
            if self.current_mode == "full" and self.stt.loaded and self.tts.loaded:
                return

            t0 = time.monotonic()
            stt_ms = await self._ensure_loaded(self.stt, "STT")

            tts_ms = 0.0
            try:
                tts_ms = await self._ensure_loaded(self.tts, "TTS")
            except Exception:
                logger.warning(
                    "TTS failed to load — STT will work but TTS unavailable",
                    exc_info=True,
                )

            total_ms = (time.monotonic() - t0) * 1000

            self.current_mode = "full"
            status_tracker.gpu_mode = "full-duplex"
            vram = float(self.stt.vram_mb)
            if self.tts.loaded:
                vram += float(self.tts.vram_mb)
            status_tracker.vram_used_mb = vram

            if stt_ms > 0 or tts_ms > 0:
                tts_label = self.tts.engine_type if self.tts.loaded else "FAILED"
                logger.info(
                    "Full-duplex: STT(%s) + TTS(%s) loaded (%.0fms)",
                    self.stt.model_size, tts_label, total_ms,
                )
                await status_tracker.emit(
                    "gpu_acquire",
                    f"Full-duplex: STT({self.stt.model_size}) + TTS({tts_label})",
                    total_ms,
                )

    # ── Half-duplex helpers ───────────────────────────────────────────

    async def _acquire_half_duplex_stt(self) -> None:
        """Half-duplex: ensure STT loaded, unloading TTS if needed."""
        async with self._lock:
            if self.current_mode == "stt" and self.stt.loaded:
                return

            t0 = time.monotonic()

            if self.tts.loaded:
                logger.info("GPU swap: TTS → STT")
                await asyncio.get_event_loop().run_in_executor(None, self.tts.unload)
                await status_tracker.emit("gpu_swap", "TTS unloaded for STT")

            await self._ensure_loaded(self.stt, "STT")

            self.current_mode = "stt"
            status_tracker.gpu_mode = "stt"
            status_tracker.vram_used_mb = float(self.stt.vram_mb)

            elapsed = (time.monotonic() - t0) * 1000
            await status_tracker.emit("gpu_acquire", f"STT mode ({self.stt.model_size})", elapsed)

    async def _acquire_half_duplex_tts(self) -> None:
        """Half-duplex: ensure TTS loaded, unloading STT if needed."""
        async with self._lock:
            if self.current_mode == "tts" and self.tts.loaded:
                return

            t0 = time.monotonic()

            if self.stt.loaded:
                logger.info("GPU swap: STT → TTS")
                await asyncio.get_event_loop().run_in_executor(None, self.stt.unload)
                await status_tracker.emit("gpu_swap", "STT unloaded for TTS")

            await self._ensure_loaded(self.tts, "TTS")

            self.current_mode = "tts"
            status_tracker.gpu_mode = "tts"
            status_tracker.vram_used_mb = float(self.tts.vram_mb)

            elapsed = (time.monotonic() - t0) * 1000
            await status_tracker.emit("gpu_acquire", f"TTS mode ({self.tts.engine_type})", elapsed)

    # ── Public API ────────────────────────────────────────────────────

    async def acquire_for_stt(self) -> None:
        """Ensure STT model is loaded and ready."""
        if self.half_duplex:
            await self._acquire_half_duplex_stt()
        else:
            await self._ensure_both_loaded()

    async def acquire_for_tts(self) -> None:
        """Ensure TTS model is loaded and ready."""
        if self.half_duplex:
            await self._acquire_half_duplex_tts()
        else:
            await self._ensure_both_loaded()

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

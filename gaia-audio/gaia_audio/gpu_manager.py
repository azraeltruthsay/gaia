"""Three-tier GPU manager for gaia-audio.

Architecture:
  - Listener (Qwen3-ASR 0.6B) — always on GPU, coexists with Prime LLM (~1.8GB)
  - Nano Speaker (Qwen3-TTS 0.6B) — always on CPU, no GPU contention
  - Prime Speaker (Qwen3-TTS 1.7B) — on-demand GPU, requires Prime LLM to yield

State transitions:
    startup()  → load Listener (GPU) + Nano Speaker (CPU)
    run_stt()  → Listener always loaded, just run
    run_tts_nano() → Nano Speaker always loaded, just run
    run_tts_prime() → acquire GPU → load Prime Speaker → run → unload → release
    sleep()    → unload Listener from GPU, keep Nano Speaker
    wake()     → reload Listener on GPU
    release()  → unload all
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import time
import httpx

from gaia_audio.status import status_tracker
from gaia_audio.stt_engine import STTEngine
from gaia_audio.tts_engine import NanoSpeaker, PrimeSpeaker, EspeakFallback

logger = logging.getLogger("GAIA.Audio.GPU")


class GPUManager:
    """Three-tier GPU management for STT + TTS."""

    def __init__(
        self,
        stt_engine: STTEngine,
        nano_speaker: NanoSpeaker,
        prime_speaker: PrimeSpeaker,
        espeak_fallback: EspeakFallback,
        orchestrator_endpoint: str = "http://gaia-orchestrator:6410",
        prime_speaker_timeout: int = 30,
    ) -> None:
        self.stt = stt_engine
        self.nano = nano_speaker
        self.prime = prime_speaker
        self.espeak = espeak_fallback
        self.orchestrator_endpoint = orchestrator_endpoint
        self.prime_speaker_timeout = prime_speaker_timeout
        self._lock = asyncio.Lock()
        self._gpu_lease_id: str | None = None

    async def startup(self) -> None:
        """Boot-time initialization: Listener (GPU) + Nano Speaker (CPU).

        Prime Speaker stays unloaded until needed.
        """
        async with self._lock:
            loop = asyncio.get_event_loop()

            # Load Listener (GPU) — coexists with Prime LLM
            t0 = time.monotonic()
            try:
                await loop.run_in_executor(None, self.stt.load)
                logger.info("Listener (Qwen3-ASR) loaded on GPU")
            except Exception:
                logger.error("Failed to load Listener — STT unavailable", exc_info=True)

            # Load Nano Speaker (CPU) — always available
            try:
                await loop.run_in_executor(None, self.nano.load)
                logger.info("Nano Speaker loaded on CPU")
            except Exception:
                logger.error("Failed to load Nano Speaker — falling back to espeak", exc_info=True)
                try:
                    await loop.run_in_executor(None, self.espeak.load)
                except Exception:
                    logger.error("EspeakFallback also failed", exc_info=True)

            elapsed = (time.monotonic() - t0) * 1000
            status_tracker.gpu_mode = "three-tier"
            status_tracker.vram_used_mb = float(self.stt.vram_mb)
            status_tracker.stt_model = "Qwen3-ASR-0.6B" if self.stt.loaded else None
            status_tracker.tts_engine = "nano_speaker" if self.nano.loaded else "espeak_fallback"
            await status_tracker.emit("startup", f"Three-tier audio ready ({elapsed:.0f}ms)", elapsed)

            # Start idle timer — auto-sleep after 5 min of no activity
            idle_seconds = int(os.environ.get("AUDIO_IDLE_TIMEOUT", "300"))
            await self.start_idle_timer(idle_seconds)

    async def run_stt(self, func, *args, **kwargs):
        """Run STT function. Auto-wakes if sleeping, resets idle timer."""
        await self.ensure_awake()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))
        self.touch_activity()
        return result

    async def run_tts_nano(self, func, *args, **kwargs):
        """Run Nano TTS function. Always on CPU, no contention."""
        self.touch_activity()
        if not self.nano.loaded:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.nano.load)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))
        self.touch_activity()
        return result

    async def run_tts_prime(self, func, *args, **kwargs):
        """Run Prime TTS function. Acquires GPU from orchestrator, loads model,
        runs, unloads, releases GPU."""
        async with self._lock:
            loop = asyncio.get_event_loop()

            # Acquire GPU lease from orchestrator
            lease_acquired = await self._acquire_gpu_lease()
            if not lease_acquired:
                logger.warning("Could not acquire GPU for Prime Speaker — falling back to Nano")
                return None  # Caller handles fallback

            try:
                t0 = time.monotonic()
                await loop.run_in_executor(None, self.prime.load)
                status_tracker.vram_used_mb = float(self.stt.vram_mb + self.prime.vram_mb)

                result = await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

                elapsed = (time.monotonic() - t0) * 1000
                await status_tracker.emit("tts_prime", f"Prime Speaker synthesized ({elapsed:.0f}ms)", elapsed)
                return result
            finally:
                # Always unload and release
                await loop.run_in_executor(None, self.prime.unload)
                status_tracker.vram_used_mb = float(self.stt.vram_mb)
                await self._release_gpu_lease()

    # ── Idle Timer: auto-sleep after inactivity ─────────────────────

    async def start_idle_timer(self, idle_seconds: int = 300) -> None:
        """Start the idle watchdog. Auto-sleeps after idle_seconds of no activity."""
        self._idle_timeout = idle_seconds
        self._last_activity = asyncio.get_event_loop().time()
        self._idle_task = asyncio.create_task(self._idle_watchdog())
        logger.info("Idle timer started: %ds timeout", idle_seconds)

    def touch_activity(self) -> None:
        """Reset the idle timer — called on any STT/TTS activity."""
        self._last_activity = asyncio.get_event_loop().time()

    async def _idle_watchdog(self) -> None:
        """Background task: sleep audio if idle for too long."""
        while True:
            await asyncio.sleep(30)  # Check every 30s
            if not hasattr(self, '_idle_timeout'):
                continue
            elapsed = asyncio.get_event_loop().time() - self._last_activity
            if elapsed >= self._idle_timeout and self.stt.loaded:
                logger.info("Idle timeout (%.0fs) — auto-sleeping audio to free VRAM", elapsed)
                await self.sleep()
                await status_tracker.emit("idle_sleep", f"Auto-sleep after {int(elapsed)}s idle")

    async def ensure_awake(self) -> None:
        """Wake if sleeping — called before any STT/TTS operation."""
        self.touch_activity()
        if not self.stt.loaded:
            logger.info("Audio waking on demand...")
            await self.wake()

    async def sleep(self) -> None:
        """Deep sleep: unload Listener from GPU. Keep Nano Speaker on CPU."""
        async with self._lock:
            loop = asyncio.get_event_loop()

            if self.stt.loaded:
                await loop.run_in_executor(None, self.stt.unload)
                logger.info("Listener unloaded for sleep")

            if self.prime.loaded:
                await loop.run_in_executor(None, self.prime.unload)
                logger.info("Prime Speaker unloaded for sleep")

            status_tracker.vram_used_mb = 0.0
            status_tracker.gpu_mode = "sleeping"
            await status_tracker.emit("sleep", "GPU models unloaded — Nano Speaker still on CPU")

    async def wake(self) -> None:
        """Wake from sleep: reload Listener on GPU."""
        async with self._lock:
            loop = asyncio.get_event_loop()

            try:
                await loop.run_in_executor(None, self.stt.load)
                logger.info("Listener reloaded after wake")
            except Exception:
                logger.error("Failed to reload Listener on wake", exc_info=True)

            status_tracker.vram_used_mb = float(self.stt.vram_mb)
            status_tracker.gpu_mode = "three-tier"
            await status_tracker.emit("wake", "Listener reloaded on GPU")

    async def release(self) -> None:
        """Unload all models and free everything."""
        async with self._lock:
            loop = asyncio.get_event_loop()

            if self.stt.loaded:
                await loop.run_in_executor(None, self.stt.unload)
            if self.nano.loaded:
                await loop.run_in_executor(None, self.nano.unload)
            if self.prime.loaded:
                await loop.run_in_executor(None, self.prime.unload)

            status_tracker.gpu_mode = "idle"
            status_tracker.vram_used_mb = 0.0
            await status_tracker.emit("gpu_release", "All audio models unloaded")

    # ── Orchestrator GPU lease ───────────────────────────────────────

    async def _acquire_gpu_lease(self) -> bool:
        """Request GPU lease from orchestrator for Prime Speaker."""
        try:
            async with httpx.AsyncClient(timeout=self.prime_speaker_timeout) as client:
                resp = await client.post(
                    f"{self.orchestrator_endpoint}/gpu/acquire",
                    json={
                        "requester": "gaia-audio",
                        "reason": "Prime Speaker TTS synthesis",
                        "timeout_seconds": self.prime_speaker_timeout,
                        "priority": 0,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("success"):
                        self._gpu_lease_id = data.get("lease_id")
                        logger.info("GPU lease acquired: %s", self._gpu_lease_id)
                        return True
                logger.warning("GPU lease denied: %s", resp.text)
                return False
        except Exception as e:
            logger.warning("Could not contact orchestrator for GPU lease: %s", e)
            # If orchestrator unreachable, try loading anyway (best effort)
            return True

    async def _release_gpu_lease(self) -> None:
        """Release GPU lease back to orchestrator."""
        if not self._gpu_lease_id:
            return
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{self.orchestrator_endpoint}/gpu/release",
                    json={"lease_id": self._gpu_lease_id},
                )
                logger.info("GPU lease released: %s", self._gpu_lease_id)
        except Exception as e:
            logger.warning("Failed to release GPU lease: %s", e)
        finally:
            self._gpu_lease_id = None

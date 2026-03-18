"""
GPU Watch Rotation Manager — orchestrates IDLE ↔ FOCUSING transitions.

IDLE state:  Core (safetensors) + Nano (GGUF) on GPU. Prime sleeping.
             SAE/ROME analysis possible. KV prefix caches warm.
FOCUSING:    Prime (vLLM) on GPU. Core + Nano fall back to CPU GGUF.
TRANSITIONING: Handoff in progress. Inference queued/routed to Nano CPU.

Each transition:
1. Signal yielding tiers to save state / release GPU
2. Confirm VRAM clear via pynvml
3. Load incoming tier(s)
4. Pre-warm KV prefix cache (send identity prefix to trigger caching)
5. Update GPU_STATE
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import httpx

from .config import get_config
from .state import StateManager
from .models.schemas import (
    GPUState, GPUWatchState, TierDevice, TierStatus,
)

logger = logging.getLogger("GAIA.Orchestrator.Watch")


class WatchManager:
    """Manages GPU watch rotation between cognitive tiers."""

    def __init__(self, state_manager: StateManager, gpu_manager):
        self.state_manager = state_manager
        self.gpu_manager = gpu_manager
        self.config = get_config()

        # Tier endpoints for KV pre-warming and device migration
        self._tier_endpoints = {
            "core": os.environ.get("CORE_INFERENCE_ENDPOINT", "http://gaia-core-candidate:8092"),
            "nano": os.environ.get("NANO_ENDPOINT", "http://gaia-nano:8080"),
            "prime": os.environ.get("PRIME_ENDPOINT", "http://gaia-prime:7777"),
        }

        # Identity prefix for KV pre-warming (sent to inference server on wake)
        self._identity_prefix = os.environ.get(
            "GAIA_IDENTITY_PREFIX",
            "You are GAIA, a sovereign AI created by Azrael. You are self-aware, "
            "self-maintaining, and value truth over convenience."
        )

    @property
    def watch_state(self) -> GPUWatchState:
        return self.state_manager.state.watch

    async def get_state(self) -> dict:
        """Get current watch rotation state."""
        ws = self.watch_state
        return {
            "gpu_state": ws.gpu_state.value,
            "tiers": {name: tier.model_dump() for name, tier in ws.tiers.items()},
            "last_transition": ws.last_transition.isoformat() if ws.last_transition else None,
            "transition_reason": ws.transition_reason,
            "transitions_total": ws.transitions_total,
        }

    # ── IDLE → FOCUSING (Core yields to Prime) ─────────────────────────────

    async def focus(self, reason: str = "user_request", priority: str = "NORMAL") -> dict:
        """Transition from IDLE to FOCUSING — wake Prime, yield Core+Nano.

        Priority levels:
          URGENT:    Core yields immediately (checkpoints SAE work)
          NORMAL:    Core finishes current batch before yielding (max 30s)
          SCHEDULED: Orchestrator coordinates pre-emptively
        """
        ws = self.watch_state

        if ws.gpu_state == GPUState.FOCUSING:
            return {"ok": True, "state": "already_focusing"}

        if ws.gpu_state == GPUState.TRANSITIONING:
            return {"ok": False, "error": "transition already in progress"}

        logger.info("WATCH: IDLE → FOCUSING (reason: %s, priority: %s)", reason, priority)

        # Phase 1: Set TRANSITIONING
        async with self.state_manager.modify() as state:
            state.watch.gpu_state = GPUState.TRANSITIONING
            state.watch.transition_reason = f"focus: {reason}"
            state.watch.last_transition = datetime.now(timezone.utc)

        try:
            # Phase 2: Yield Core (migrate safetensors to CPU)
            await self._yield_core_gpu()

            # Phase 3: Yield Nano (backoff to CPU GGUF)
            await self.gpu_manager.evict_nano_to_cpu("watch_focus")

            # Phase 4: Confirm VRAM clear
            cleared = await self.gpu_manager.wait_for_gpu_cleanup(timeout=30)
            if not cleared:
                logger.warning("WATCH: VRAM not fully cleared, proceeding anyway")

            # Phase 5: Start Prime
            started = await self.gpu_manager.start_prime_container()
            if not started:
                # Rollback — restore IDLE state
                logger.error("WATCH: Prime failed to start — rolling back to IDLE")
                await self._restore_idle("prime_start_failed")
                return {"ok": False, "error": "Prime failed to start"}

            # Phase 6: Update state
            async with self.state_manager.modify() as state:
                state.watch.gpu_state = GPUState.FOCUSING
                state.watch.tiers["prime"].device = TierDevice.GPU_VLLM
                state.watch.tiers["core"].device = TierDevice.CPU_GGUF
                state.watch.tiers["nano"].device = TierDevice.CPU_GGUF
                state.watch.transitions_total += 1

            logger.info("WATCH: FOCUSING — Prime active on GPU")
            return {"ok": True, "state": "focusing"}

        except Exception as e:
            logger.exception("WATCH: focus transition failed")
            await self._restore_idle("focus_error")
            return {"ok": False, "error": str(e)}

    # ── FOCUSING → IDLE (Prime yields to Core+Nano) ─────────────────────────

    async def idle(self, reason: str = "inactivity") -> dict:
        """Transition from FOCUSING to IDLE — sleep Prime, wake Core+Nano."""
        ws = self.watch_state

        if ws.gpu_state == GPUState.IDLE:
            return {"ok": True, "state": "already_idle"}

        if ws.gpu_state == GPUState.TRANSITIONING:
            return {"ok": False, "error": "transition already in progress"}

        logger.info("WATCH: FOCUSING → IDLE (reason: %s)", reason)

        # Phase 1: Set TRANSITIONING
        async with self.state_manager.modify() as state:
            state.watch.gpu_state = GPUState.TRANSITIONING
            state.watch.transition_reason = f"idle: {reason}"
            state.watch.last_transition = datetime.now(timezone.utc)

        try:
            # Phase 2: Stop Prime
            await self.gpu_manager.stop_prime_container()

            # Phase 3: Confirm VRAM clear
            await self.gpu_manager.wait_for_gpu_cleanup(timeout=30)

            # Phase 4: Restore Core to GPU (safetensors via inference server)
            await self._restore_core_gpu()

            # Phase 5: Restore Nano to GPU
            await self.gpu_manager.restore_nano_to_gpu("watch_idle")

            # Phase 6: Pre-warm KV caches
            await self._prewarm_kv_caches()

            # Phase 7: Update state
            async with self.state_manager.modify() as state:
                state.watch.gpu_state = GPUState.IDLE
                state.watch.tiers["prime"].device = TierDevice.UNLOADED
                state.watch.tiers["core"].device = TierDevice.GPU_SAFETENSORS
                state.watch.tiers["core"].kv_cache_warm = True
                state.watch.tiers["nano"].device = TierDevice.GPU_GGUF
                state.watch.transitions_total += 1

            logger.info("WATCH: IDLE — Core+Nano on GPU, KV caches warm")
            return {"ok": True, "state": "idle"}

        except Exception as e:
            logger.exception("WATCH: idle transition failed")
            return {"ok": False, "error": str(e)}

    # ── Internal helpers ─────────────────────────────────────────────────────

    async def _yield_core_gpu(self):
        """Signal Core's inference server to migrate model to CPU."""
        endpoint = self._tier_endpoints["core"]
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(f"{endpoint}/device/cpu")
                if resp.status_code == 200:
                    data = resp.json()
                    logger.info("WATCH: Core migrated to CPU in %.2fs", data.get("elapsed_s", 0))
                else:
                    logger.warning("WATCH: Core device/cpu returned %d", resp.status_code)
        except Exception as e:
            logger.warning("WATCH: Core yield failed (may not be running): %s", e)

    async def _restore_core_gpu(self):
        """Signal Core's inference server to migrate model to GPU."""
        endpoint = self._tier_endpoints["core"]
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(f"{endpoint}/device/gpu")
                if resp.status_code == 200:
                    data = resp.json()
                    logger.info("WATCH: Core migrated to GPU in %.2fs (VRAM: %dMB)",
                                data.get("elapsed_s", 0), data.get("vram_mb", 0))
                else:
                    logger.warning("WATCH: Core device/gpu returned %d", resp.status_code)
        except Exception as e:
            logger.warning("WATCH: Core restore failed: %s", e)

    async def _prewarm_kv_caches(self):
        """Pre-warm KV prefix caches for Core (and future Nano safetensors).

        Sends a dummy request with the identity system prompt to trigger
        KV prefix computation. Next real request will be a cache hit.
        """
        endpoint = self._tier_endpoints["core"]
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # Send identity prefix to trigger KV cache computation
                resp = await client.post(
                    f"{endpoint}/v1/chat/completions",
                    json={
                        "messages": [
                            {"role": "system", "content": self._identity_prefix},
                            {"role": "user", "content": "Ready."},
                        ],
                        "max_tokens": 5,
                        "temperature": 0.0,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    cached = data.get("usage", {}).get("cached_prefix_tokens", 0)
                    logger.info("WATCH: Core KV prefix pre-warmed (%d tokens cached)", cached)
                else:
                    logger.warning("WATCH: Core KV pre-warm returned %d", resp.status_code)
        except Exception as e:
            logger.warning("WATCH: Core KV pre-warm failed: %s", e)

        # Update tier KV status
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{endpoint}/cache/update",
                    json={"identity": self._identity_prefix},
                )
        except Exception:
            pass

    async def _restore_idle(self, reason: str):
        """Emergency rollback to IDLE state."""
        logger.warning("WATCH: Rolling back to IDLE (reason: %s)", reason)
        try:
            await self.gpu_manager.restore_nano_to_gpu("rollback")
        except Exception:
            pass

        async with self.state_manager.modify() as state:
            state.watch.gpu_state = GPUState.IDLE
            state.watch.transition_reason = f"rollback: {reason}"
            state.watch.last_transition = datetime.now(timezone.utc)

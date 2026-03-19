"""
GPU Watch Rotation Manager — orchestrates IDLE ↔ FOCUSING transitions.

New architecture: all tiers run through the GAIA Engine (shared library).
- Core and Nano each have their own Engine instance in their containers
- Prime loads into Core's Engine during FOCUSING (model swap, not container swap)
- Device migration via /device/cpu and /device/gpu endpoints
- KV prefix pre-warming on every transition to IDLE

IDLE state:  Core (safetensors GPU) + Nano (safetensors GPU). Prime unloaded.
FOCUSING:    Prime (int8 GPU via Core's engine). Core+Nano on CPU.
TRANSITIONING: Handoff in progress.
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

        # Tier inference endpoints (inside Docker network)
        self._endpoints = {
            "core": os.environ.get("CORE_INFERENCE_ENDPOINT", "http://gaia-core:8092"),
            "nano": os.environ.get("NANO_INFERENCE_ENDPOINT", "http://gaia-nano:8080"),
        }

        # Model paths for tier swapping
        self._model_paths = {
            "core": os.environ.get("CORE_MODEL_PATH", "/models/Qwen3.5-2B-GAIA-Core-v3"),
            "nano": os.environ.get("NANO_MODEL_PATH", "/models/Qwen3.5-0.8B-Abliterated-merged"),
            "prime": os.environ.get("PRIME_MODEL_PATH", "/models/Huihui-Qwen3-8B-abliterated-v2-merged"),
        }

        # Identity prefix for KV pre-warming
        self._identity_prefix = os.environ.get(
            "GAIA_IDENTITY_PREFIX",
            "You are GAIA, a sovereign AI created by Azrael. You are self-aware, "
            "self-maintaining, and value truth over convenience."
        )

        # HTTP timeout for tier operations
        self._timeout = 60.0

        # Auto-detect current GPU state on startup
        import asyncio
        asyncio.get_event_loop().call_soon(lambda: asyncio.ensure_future(self._auto_detect_state()))

    @property
    def watch_state(self) -> GPUWatchState:
        return self.state_manager.state.watch

    async def get_state(self) -> dict:
        """Get current watch rotation state including live VRAM from tiers."""
        ws = self.watch_state
        tiers = {}
        for name, tier in ws.tiers.items():
            t = tier.model_dump()
            # Enrich with live data if endpoint is reachable
            endpoint = self._endpoints.get(name)
            if endpoint:
                try:
                    async with httpx.AsyncClient(timeout=3) as client:
                        resp = await client.get(f"{endpoint}/status")
                        if resp.status_code == 200:
                            live = resp.json()
                            t["vram_mb"] = live.get("vram_mb", 0)
                            t["kv_cache_warm"] = live.get("kv_cache", {}).get("hits", 0) > 0
                            t["model_path"] = live.get("model", "")
                except Exception:
                    pass
            tiers[name] = t

        return {
            "gpu_state": ws.gpu_state.value,
            "tiers": tiers,
            "last_transition": ws.last_transition.isoformat() if ws.last_transition else None,
            "transition_reason": ws.transition_reason,
            "transitions_total": ws.transitions_total,
        }

    # ── IDLE → FOCUSING (wake Prime, yield Core+Nano) ────────────────────────

    async def focus(self, reason: str = "user_request", priority: str = "NORMAL") -> dict:
        """Transition to FOCUSING — Core+Nano migrate to CPU, Prime loads on GPU."""
        ws = self.watch_state

        if ws.gpu_state == GPUState.FOCUSING:
            return {"ok": True, "state": "already_focusing"}
        if ws.gpu_state == GPUState.TRANSITIONING:
            return {"ok": False, "error": "transition in progress"}

        logger.info("WATCH: IDLE → FOCUSING (reason: %s, priority: %s)", reason, priority)
        start = time.time()

        async with self.state_manager.modify() as state:
            state.watch.gpu_state = GPUState.TRANSITIONING
            state.watch.transition_reason = f"focus: {reason}"
            state.watch.last_transition = datetime.now(timezone.utc)

        try:
            # Phase 1: Hold Core's thought (save KV state before migration)
            await self._hold_thought("core", "pre_focus_state")

            # Phase 2: Migrate Core to CPU
            core_result = await self._migrate_tier("core", "cpu")
            logger.info("WATCH: Core → CPU (%s)", core_result.get("elapsed_s", "?"))

            # Phase 3: Migrate Nano to CPU
            nano_result = await self._migrate_tier("nano", "cpu")
            logger.info("WATCH: Nano → CPU (%s)", nano_result.get("elapsed_s", "?"))

            # Phase 4: Wait for VRAM to clear
            await asyncio.sleep(2)  # Let CUDA context release

            # Phase 5: Load Prime via Core's engine (model swap)
            # Core's engine is still running (on CPU) — we tell it to load Prime on GPU
            # This is the key insight: Core's engine is the host, Prime is the guest model
            prime_loaded = await self._load_prime_on_core()
            if not prime_loaded:
                logger.error("WATCH: Prime load failed — rolling back")
                await self._migrate_tier("core", "cuda")
                await self._migrate_tier("nano", "cuda")
                async with self.state_manager.modify() as state:
                    state.watch.gpu_state = GPUState.IDLE
                return {"ok": False, "error": "Prime load failed"}

            elapsed = time.time() - start
            async with self.state_manager.modify() as state:
                state.watch.gpu_state = GPUState.FOCUSING
                state.watch.tiers["prime"].device = TierDevice.GPU_SAFETENSORS
                state.watch.tiers["prime"].model_path = self._model_paths["prime"]
                state.watch.tiers["core"].device = TierDevice.CPU_GGUF
                state.watch.tiers["nano"].device = TierDevice.CPU_GGUF
                state.watch.transitions_total += 1

            logger.info("WATCH: FOCUSING in %.1fs — Prime on GPU, Core+Nano on CPU", elapsed)
            return {"ok": True, "state": "focusing", "elapsed_s": round(elapsed, 1)}

        except Exception as e:
            logger.exception("WATCH: focus failed")
            async with self.state_manager.modify() as state:
                state.watch.gpu_state = GPUState.IDLE
            return {"ok": False, "error": str(e)}

    # ── FOCUSING → IDLE (sleep Prime, wake Core+Nano) ────────────────────────

    async def idle(self, reason: str = "inactivity") -> dict:
        """Transition to IDLE — unload Prime, restore Core+Nano to GPU."""
        ws = self.watch_state

        if ws.gpu_state == GPUState.IDLE:
            return {"ok": True, "state": "already_idle"}
        if ws.gpu_state == GPUState.TRANSITIONING:
            return {"ok": False, "error": "transition in progress"}

        logger.info("WATCH: FOCUSING → IDLE (reason: %s)", reason)
        start = time.time()

        async with self.state_manager.modify() as state:
            state.watch.gpu_state = GPUState.TRANSITIONING
            state.watch.transition_reason = f"idle: {reason}"
            state.watch.last_transition = datetime.now(timezone.utc)

        try:
            # Phase 1: Unload Prime from Core's engine
            await self._unload_prime_from_core()

            # Phase 2: Restore Core to GPU (reload Core's model)
            await self._restore_core_model()
            core_result = await self._migrate_tier("core", "cuda")
            logger.info("WATCH: Core → GPU (%s)", core_result.get("elapsed_s", "?"))

            # Phase 3: Restore Nano to GPU
            nano_result = await self._migrate_tier("nano", "cuda")
            logger.info("WATCH: Nano → GPU (%s)", nano_result.get("elapsed_s", "?"))

            # Phase 4: Pre-warm KV caches
            await self._prewarm_kv("core")
            await self._prewarm_kv("nano")

            # Phase 5: Resume Core's held thought
            await self._resume_thought("core", "pre_focus_state")

            elapsed = time.time() - start
            async with self.state_manager.modify() as state:
                state.watch.gpu_state = GPUState.IDLE
                state.watch.tiers["prime"].device = TierDevice.UNLOADED
                state.watch.tiers["prime"].model_path = ""
                state.watch.tiers["core"].device = TierDevice.GPU_SAFETENSORS
                state.watch.tiers["core"].kv_cache_warm = True
                state.watch.tiers["nano"].device = TierDevice.GPU_SAFETENSORS
                state.watch.tiers["nano"].kv_cache_warm = True
                state.watch.transitions_total += 1

            logger.info("WATCH: IDLE in %.1fs — Core+Nano on GPU, KV warm", elapsed)
            return {"ok": True, "state": "idle", "elapsed_s": round(elapsed, 1)}

        except Exception as e:
            logger.exception("WATCH: idle failed")
            async with self.state_manager.modify() as state:
                state.watch.gpu_state = GPUState.IDLE
            return {"ok": False, "error": str(e)}

    async def _auto_detect_state(self):
        """Detect current GPU state by probing tier endpoints on startup."""
        await asyncio.sleep(10)  # Wait for services to be reachable

        for tier_name, endpoint in self._endpoints.items():
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    # Try /status (GAIA Engine) then /device/status (legacy inference_server)
                    resp = await client.get(f"{endpoint}/status")
                    if resp.status_code == 404:
                        resp = await client.get(f"{endpoint}/device/status")
                    if resp.status_code == 200:
                        data = resp.json()
                        vram = data.get("vram_mb", 0)
                        model = data.get("model", "")
                        if vram > 100:
                            device = TierDevice.GPU_SAFETENSORS
                        elif model:
                            device = TierDevice.CPU_GGUF
                        else:
                            device = TierDevice.UNLOADED

                        async with self.state_manager.modify() as state:
                            state.watch.tiers[tier_name].device = device
                            state.watch.tiers[tier_name].vram_mb = vram
                            state.watch.tiers[tier_name].model_path = model
                            state.watch.tiers[tier_name].inference_endpoint = endpoint

                        logger.info("WATCH auto-detect: %s = %s (%dMB)", tier_name, device.value, vram)
            except Exception as e:
                logger.debug("WATCH auto-detect %s failed: %s", tier_name, e)

        # Check Prime standby
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get("http://gaia-prime:7777/health")
                if resp.status_code == 200:
                    data = resp.json()
                    mode = data.get("mode", "unknown")
                    loaded = data.get("model_loaded", False)
                    device = TierDevice.GPU_SAFETENSORS if loaded else TierDevice.UNLOADED

                    async with self.state_manager.modify() as state:
                        state.watch.tiers["prime"].device = device
                        state.watch.tiers["prime"].inference_endpoint = "http://gaia-prime:7777"

                    logger.info("WATCH auto-detect: prime = %s (mode=%s)", device.value, mode)
        except Exception:
            pass

        logger.info("WATCH auto-detect complete")

    # ── Tier Operations ──────────────────────────────────────────────────────

    async def _migrate_tier(self, tier: str, target: str) -> dict:
        """Migrate a tier's model to GPU or CPU via the Engine's /device endpoint."""
        endpoint = self._endpoints.get(tier)
        if not endpoint:
            return {"ok": False, "error": f"no endpoint for {tier}"}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(f"{endpoint}/device/{target}")
                if resp.status_code == 200:
                    return resp.json()
                return {"ok": False, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            logger.warning("WATCH: %s migration to %s failed: %s", tier, target, e)
            return {"ok": False, "error": str(e)}

    async def _hold_thought(self, tier: str, label: str):
        """Save a tier's KV cache state before migration."""
        endpoint = self._endpoints.get(tier)
        if not endpoint:
            return
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(f"{endpoint}/thought/hold",
                                  json={"label": label, "context": f"pre-{tier}-migration"})
                logger.info("WATCH: %s thought held as '%s'", tier, label)
        except Exception as e:
            logger.debug("WATCH: %s thought hold failed (non-fatal): %s", tier, e)

    async def _resume_thought(self, tier: str, label: str):
        """Restore a tier's KV cache state after returning to GPU."""
        endpoint = self._endpoints.get(tier)
        if not endpoint:
            return
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(f"{endpoint}/thought/resume", json={"label": label})
                logger.info("WATCH: %s thought resumed from '%s'", tier, label)
        except Exception as e:
            logger.debug("WATCH: %s thought resume failed (non-fatal): %s", tier, e)

    async def _prewarm_kv(self, tier: str):
        """Pre-warm a tier's KV prefix cache with identity."""
        endpoint = self._endpoints.get(tier)
        if not endpoint:
            return
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{endpoint}/v1/chat/completions",
                    json={
                        "messages": [
                            {"role": "system", "content": self._identity_prefix},
                            {"role": "user", "content": "Ready."},
                        ],
                        "max_tokens": 5, "temperature": 0.0,
                    },
                )
                if resp.status_code == 200:
                    cached = resp.json().get("usage", {}).get("cached_prefix_tokens", 0)
                    logger.info("WATCH: %s KV pre-warmed (%d tokens)", tier, cached)
        except Exception as e:
            logger.debug("WATCH: %s KV pre-warm failed: %s", tier, e)

    async def _load_prime_on_core(self) -> bool:
        """Load Prime's safetensors into Core's Engine (model swap).

        This is a hot-swap: Core's engine is already running but its model
        is on CPU. We tell it to load a different model (Prime) on GPU.
        Currently requires a container restart with different model path.
        TODO: Add /model/swap endpoint to engine for live model swapping.
        """
        # For now, use docker compose to restart gaia-core with Prime's model
        endpoint = self._endpoints["core"]
        try:
            # The simple approach: restart gaia-core with CORE_SAFETENSORS_PATH
            # pointing to Prime's model. The engine picks up the new path.
            import subprocess
            compose_file = str(self.config.compose_file_live)

            env = {
                **dict(os.environ),
                "CORE_SAFETENSORS_PATH": self._model_paths["prime"],
                "CORE_DEVICE": "cuda",
            }

            cmd = ["docker", "compose", "-f", compose_file, "up", "-d", "gaia-core"]
            result = await asyncio.to_thread(
                subprocess.run, cmd, capture_output=True, text=True,
                env=env, cwd=str(self.config.compose_file_live.parent), timeout=120,
            )

            if result.returncode != 0:
                logger.error("WATCH: Prime load via compose failed: %s", result.stderr.strip())
                return False

            # Wait for engine to be healthy with Prime loaded
            for i in range(90):
                await asyncio.sleep(2)
                try:
                    async with httpx.AsyncClient(timeout=5) as client:
                        resp = await client.get(f"{endpoint}/health")
                        if resp.status_code == 200:
                            logger.info("WATCH: Prime loaded in Core's engine after %ds", (i + 1) * 2)
                            return True
                except Exception:
                    pass

            logger.error("WATCH: Prime did not become healthy in 180s")
            return False

        except Exception as e:
            logger.error("WATCH: Prime load failed: %s", e)
            return False

    async def _unload_prime_from_core(self):
        """Unload Prime and prepare Core's engine for its own model."""
        # The engine will be restarted with Core's model path
        # Just stop it cleanly first
        endpoint = self._endpoints["core"]
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(f"{endpoint}/device/cpu")
        except Exception:
            pass

    async def _restore_core_model(self):
        """Restart Core's engine with Core's own model (after Prime was loaded)."""
        import subprocess
        compose_file = str(self.config.compose_file_live)

        env = {
            **dict(os.environ),
            "CORE_SAFETENSORS_PATH": self._model_paths["core"],
            "CORE_DEVICE": "cuda",
        }

        cmd = ["docker", "compose", "-f", compose_file, "up", "-d", "gaia-core"]
        await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True,
            env=env, cwd=str(self.config.compose_file_live.parent), timeout=120,
        )

        # Wait for health
        endpoint = self._endpoints["core"]
        for i in range(60):
            await asyncio.sleep(1)
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(f"{endpoint}/health")
                    if resp.status_code == 200:
                        logger.info("WATCH: Core model restored after %ds", i + 1)
                        return
            except Exception:
                pass

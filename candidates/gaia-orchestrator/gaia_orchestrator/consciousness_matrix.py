"""
Consciousness Matrix — three-state resource manager for cognitive tiers.

Each tier (Nano, Core, Prime) exists in one of three consciousness states:
  3 = Conscious (GPU, safetensors, fast, full activation capture)
  2 = Subconscious (CPU, GGUF, moderate speed, always-available)
  1 = Unconscious (unloaded, no resources)

The matrix tracks target vs actual state for each tier, with dynamic
GPU/RAM polling to validate transitions. The Orchestrator uses this
to coordinate hot-swaps between tiers.

Key principles:
- Any tier CAN be in any state (no biological hard constraints)
- States are operational preferences, not survival requirements
- Like dolphin unihemispheric sleep — parts rest independently
- Even all-unconscious isn't death (Groq fallback exists)
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Optional

import httpx

logger = logging.getLogger("GAIA.Orchestrator.Consciousness")


class ConsciousnessLevel(IntEnum):
    UNCONSCIOUS = 1   # Unloaded, no resources
    SUBCONSCIOUS = 2  # CPU (GGUF via llama-server)
    CONSCIOUS = 3     # GPU (safetensors via GAIA Engine)


@dataclass
class TierState:
    """Live state of a single cognitive tier."""
    tier: str
    target: ConsciousnessLevel = ConsciousnessLevel.UNCONSCIOUS
    actual: ConsciousnessLevel = ConsciousnessLevel.UNCONSCIOUS
    gpu_mb: int = 0
    ram_mb: int = 0
    healthy: bool = False
    last_probe: float = 0.0
    transitioning: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.target == self.actual and self.healthy and not self.transitioning

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "target": self.target.name.lower(),
            "actual": self.actual.name.lower(),
            "target_level": int(self.target),
            "actual_level": int(self.actual),
            "gpu_mb": self.gpu_mb,
            "ram_mb": self.ram_mb,
            "healthy": self.healthy,
            "ok": self.ok,
            "transitioning": self.transitioning,
            "error": self.error,
        }


class ConsciousnessMatrix:
    """
    Tracks and manages consciousness states for all cognitive tiers.

    The matrix is the single source of truth for what state each tier
    SHOULD be in (target) and what state it IS in (actual). The
    Orchestrator calls transition methods to change targets, and the
    matrix validates by polling actual resource usage.
    """

    def __init__(self):
        # Tier engine endpoints
        self._endpoints = {
            "nano": os.environ.get("NANO_INFERENCE_ENDPOINT", "http://gaia-nano:8080"),
            "core": os.environ.get("CORE_INFERENCE_ENDPOINT", "http://gaia-core:8092"),
            "prime": os.environ.get("PRIME_INFERENCE_ENDPOINT", "http://gaia-prime:7777"),
        }

        # Safetensors model paths (for state 3 = Conscious/GPU)
        self._gpu_models = {
            "nano": os.environ.get("NANO_SAFETENSORS_PATH", "/models/Qwen3.5-0.8B-Abliterated"),
            "core": os.environ.get("CORE_SAFETENSORS_PATH", "/models/Qwen3.5-2B-GAIA-Core-v3"),
            "prime": os.environ.get("PRIME_MODEL_PATH", "/models/Huihui-Qwen3-8B-GAIA-Prime-adaptive"),
        }

        # GGUF model paths (for state 2 = Subconscious/CPU)
        self._cpu_models = {
            "nano": os.environ.get("NANO_GGUF_PATH", "/models/Qwen3.5-0.8B-Abliterated-Q8_0.gguf"),
            "core": os.environ.get("CORE_GGUF_PATH", "/models/Qwen3.5-2B-BF16.gguf"),
            "prime": os.environ.get("PRIME_GGUF_PATH", "/models/Huihui-Qwen3-8B-GAIA-Prime-identity-Q8_0.gguf"),
        }

        # The matrix — one entry per tier
        self._tiers: Dict[str, TierState] = {
            "nano": TierState(tier="nano"),
            "core": TierState(tier="core"),
            "prime": TierState(tier="prime"),
        }

        self._lock = asyncio.Lock()
        self._poll_task: Optional[asyncio.Task] = None

    # ── Public API ────────────────────────────────────────────────────

    def get_matrix(self) -> Dict[str, dict]:
        """Return the full consciousness matrix."""
        return {tier: state.to_dict() for tier, state in self._tiers.items()}

    def get_tier(self, tier: str) -> Optional[TierState]:
        return self._tiers.get(tier)

    async def set_target(self, tier: str, level: ConsciousnessLevel) -> dict:
        """Set the target consciousness level for a tier.

        This doesn't immediately change the state — it sets the target
        and then executes the transition to reach it.
        """
        if tier not in self._tiers:
            return {"ok": False, "error": f"Unknown tier: {tier}"}

        state = self._tiers[tier]
        old_target = state.target
        state.target = level

        if old_target == level:
            return {"ok": True, "message": "already at target", "tier": tier, "level": level.name}

        logger.info("Consciousness target: %s %s → %s",
                     tier, old_target.name, level.name)

        # Execute the transition
        result = await self._transition_tier(tier, old_target, level)
        return result

    async def probe_all(self) -> Dict[str, dict]:
        """Probe all tiers and update actual states."""
        for tier in self._tiers:
            await self._probe_tier(tier)
        return self.get_matrix()

    async def start_continuous_poll(self, interval: float = 10.0):
        """Start a background task that continuously validates the matrix."""
        if self._poll_task and not self._poll_task.done():
            return
        self._poll_task = asyncio.create_task(self._poll_loop(interval))
        logger.info("Consciousness matrix continuous poll started (%.0fs interval)", interval)

    def stop_poll(self):
        if self._poll_task:
            self._poll_task.cancel()

    # ── Preset Configurations ─────────────────────────────────────────

    async def awake(self) -> dict:
        """AWAKE: Core=3, Nano=3, Prime=2"""
        results = {}
        results["nano"] = await self.set_target("nano", ConsciousnessLevel.CONSCIOUS)
        results["core"] = await self.set_target("core", ConsciousnessLevel.CONSCIOUS)
        results["prime"] = await self.set_target("prime", ConsciousnessLevel.SUBCONSCIOUS)
        return {"configuration": "awake", "results": results}

    async def focusing(self) -> dict:
        """FOCUSING: Nano=3, Core=2, Prime=3"""
        results = {}
        # Demote Core first to free GPU
        results["core"] = await self.set_target("core", ConsciousnessLevel.SUBCONSCIOUS)
        # Then promote Prime
        results["prime"] = await self.set_target("prime", ConsciousnessLevel.CONSCIOUS)
        results["nano"] = await self.set_target("nano", ConsciousnessLevel.CONSCIOUS)
        return {"configuration": "focusing", "results": results}

    async def sleep(self) -> dict:
        """SLEEP: Nano=2, Core=2, Prime=1"""
        results = {}
        results["prime"] = await self.set_target("prime", ConsciousnessLevel.UNCONSCIOUS)
        results["core"] = await self.set_target("core", ConsciousnessLevel.SUBCONSCIOUS)
        results["nano"] = await self.set_target("nano", ConsciousnessLevel.SUBCONSCIOUS)
        return {"configuration": "sleep", "results": results}

    async def deep_sleep(self) -> dict:
        """DEEP SLEEP: All → 1 (Nano stays 2 for wake detection)"""
        results = {}
        results["prime"] = await self.set_target("prime", ConsciousnessLevel.UNCONSCIOUS)
        results["core"] = await self.set_target("core", ConsciousnessLevel.UNCONSCIOUS)
        results["nano"] = await self.set_target("nano", ConsciousnessLevel.SUBCONSCIOUS)
        return {"configuration": "deep_sleep", "results": results}

    async def training(self, tier: str = "prime") -> dict:
        """TRAINING: Target tier → 1 (free GPU), others → 2"""
        results = {}
        for t in self._tiers:
            if t == tier:
                results[t] = await self.set_target(t, ConsciousnessLevel.UNCONSCIOUS)
            else:
                results[t] = await self.set_target(t, ConsciousnessLevel.SUBCONSCIOUS)
        return {"configuration": f"training_{tier}", "results": results}

    # ── Internal: Tier Probing ────────────────────────────────────────

    async def _probe_tier(self, tier: str):
        """Probe a tier's engine endpoint and update actual state."""
        state = self._tiers[tier]
        endpoint = self._endpoints.get(tier)
        if not endpoint:
            return

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{endpoint}/health")
                if resp.status_code == 200:
                    data = resp.json()
                    state.healthy = True
                    state.last_probe = time.time()

                    # Determine consciousness level from health response
                    # Manager responses: {managed:true, model_loaded:bool, mode:str, backend:str}
                    # GGUF worker responses: {status:"ok"} (no model_loaded field)
                    # Standalone engine: {engine:"gaia", model_loaded:bool, ...}
                    managed = data.get("managed", False)
                    model_loaded = data.get("model_loaded")
                    mode = data.get("mode", "")
                    backend = data.get("backend", "none")
                    worker_pid = data.get("worker_pid")

                    if managed:
                        # Managed engine — check if worker is active
                        if mode == "active" and model_loaded:
                            if backend == "gguf":
                                state.actual = ConsciousnessLevel.SUBCONSCIOUS
                            else:
                                state.actual = ConsciousnessLevel.CONSCIOUS
                        elif mode == "standby" or model_loaded is False:
                            state.actual = ConsciousnessLevel.UNCONSCIOUS
                        elif worker_pid is not None:
                            # Worker running but mode unclear — probably loading
                            state.actual = ConsciousnessLevel.CONSCIOUS
                        else:
                            state.actual = ConsciousnessLevel.UNCONSCIOUS
                    elif data.get("status") == "ok" and "model_loaded" not in data:
                        # GGUF llama-server health (proxied through manager)
                        # If we get here with status=ok, a worker IS running
                        state.actual = ConsciousnessLevel.SUBCONSCIOUS
                    elif model_loaded:
                        state.actual = ConsciousnessLevel.CONSCIOUS
                    else:
                        state.actual = ConsciousnessLevel.UNCONSCIOUS

                    state.gpu_mb = data.get("vram_mb", 0)
                    state.error = ""
                else:
                    state.healthy = False
                    state.actual = ConsciousnessLevel.UNCONSCIOUS
                    state.error = f"HTTP {resp.status_code}"
        except Exception as e:
            state.healthy = False
            state.actual = ConsciousnessLevel.UNCONSCIOUS
            state.error = str(e)[:100]

    # ── Internal: State Transitions ───────────────────────────────────

    async def _transition_tier(
        self, tier: str, from_level: ConsciousnessLevel, to_level: ConsciousnessLevel
    ) -> dict:
        """Execute a consciousness state transition for a single tier."""
        state = self._tiers[tier]
        endpoint = self._endpoints[tier]
        state.transitioning = True
        state.error = ""

        try:
            if to_level == ConsciousnessLevel.UNCONSCIOUS:
                # Unload everything
                return await self._unload_tier(tier, endpoint, state)

            elif to_level == ConsciousnessLevel.SUBCONSCIOUS:
                # Load GGUF on CPU
                # First unload current if conscious (GPU)
                if from_level == ConsciousnessLevel.CONSCIOUS:
                    await self._unload_tier(tier, endpoint, state)
                    await asyncio.sleep(1)  # brief pause for VRAM release
                return await self._load_tier_cpu(tier, endpoint, state)

            elif to_level == ConsciousnessLevel.CONSCIOUS:
                # Load safetensors on GPU
                if from_level == ConsciousnessLevel.SUBCONSCIOUS:
                    await self._unload_tier(tier, endpoint, state)
                    await asyncio.sleep(1)
                return await self._load_tier_gpu(tier, endpoint, state)

        except Exception as e:
            state.error = str(e)[:200]
            logger.error("Transition failed for %s: %s", tier, e)
            return {"ok": False, "tier": tier, "error": str(e)}
        finally:
            state.transitioning = False
            await self._probe_tier(tier)

    async def _unload_tier(self, tier: str, endpoint: str, state: TierState) -> dict:
        """Unload a tier's model (any state → unconscious)."""
        logger.info("Unloading %s", tier)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(f"{endpoint}/model/unload")
                data = resp.json() if resp.status_code in (200, 409) else {}
                state.actual = ConsciousnessLevel.UNCONSCIOUS
                state.gpu_mb = 0
                state.ram_mb = 0
                logger.info("Unloaded %s: %s", tier, data.get("message", "ok"))
                return {"ok": True, "tier": tier, "action": "unloaded"}
        except Exception as e:
            logger.warning("Unload %s failed: %s", tier, e)
            return {"ok": False, "tier": tier, "error": str(e)}

    async def _load_tier_gpu(self, tier: str, endpoint: str, state: TierState) -> dict:
        """Load a tier on GPU (safetensors → conscious)."""
        model = self._gpu_models.get(tier)
        if not model:
            return {"ok": False, "tier": tier, "error": "no GPU model configured"}

        logger.info("Loading %s to GPU: %s", tier, model)
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                # Check if already loaded
                health = await client.get(f"{endpoint}/health")
                if health.status_code == 200:
                    hdata = health.json()
                    if hdata.get("model_loaded") and hdata.get("mode") == "active":
                        state.actual = ConsciousnessLevel.CONSCIOUS
                        logger.info("%s already loaded on GPU", tier)
                        return {"ok": True, "tier": tier, "action": "already_loaded_gpu"}

                resp = await client.post(
                    f"{endpoint}/model/load",
                    json={"model": model, "device": "cuda"},
                )
                if resp.status_code in (200, 409):
                    data = resp.json()
                    if data.get("ok") or resp.status_code == 409:
                        state.actual = ConsciousnessLevel.CONSCIOUS
                        logger.info("Loaded %s to GPU", tier)
                        return {"ok": True, "tier": tier, "action": "loaded_gpu", "model": model}
                return {"ok": False, "tier": tier, "error": f"HTTP {resp.status_code}: {resp.text[:100]}"}
        except Exception as e:
            return {"ok": False, "tier": tier, "error": str(e)}

    async def _load_tier_cpu(self, tier: str, endpoint: str, state: TierState) -> dict:
        """Load a tier on CPU (GGUF → subconscious)."""
        model = self._cpu_models.get(tier)
        if not model:
            return {"ok": False, "tier": tier, "error": "no GGUF model configured"}

        logger.info("Loading %s to CPU (GGUF): %s", tier, model)
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{endpoint}/model/load",
                    json={"model": model, "device": "cpu"},
                )
                if resp.status_code in (200, 409):
                    data = resp.json()
                    if data.get("ok") or resp.status_code == 409:
                        state.actual = ConsciousnessLevel.SUBCONSCIOUS
                        logger.info("Loaded %s to CPU (GGUF)", tier)
                        return {"ok": True, "tier": tier, "action": "loaded_cpu", "model": model}
                return {"ok": False, "tier": tier, "error": f"HTTP {resp.status_code}: {resp.text[:100]}"}
        except Exception as e:
            return {"ok": False, "tier": tier, "error": str(e)}

    # ── Internal: Continuous Poll ─────────────────────────────────────

    async def _poll_loop(self, interval: float):
        """Continuously probe tiers and validate matrix."""
        while True:
            try:
                await self.probe_all()
                # Log any mismatches
                for tier, state in self._tiers.items():
                    if not state.ok and not state.transitioning:
                        logger.warning(
                            "Matrix mismatch: %s target=%s actual=%s healthy=%s error=%s",
                            tier, state.target.name, state.actual.name,
                            state.healthy, state.error[:50] if state.error else ""
                        )
            except Exception as e:
                logger.debug("Poll error: %s", e)
            await asyncio.sleep(interval)

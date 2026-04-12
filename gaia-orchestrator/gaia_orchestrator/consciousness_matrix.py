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

    def __init__(self, lifecycle_machine=None):
        self._lifecycle_machine = lifecycle_machine

        # Load defaults from shared constants, with env var overrides
        cfg = self._load_config()

        # Tier engine endpoints
        self._endpoints = {
            "nano": os.environ.get("NANO_INFERENCE_ENDPOINT", cfg["inference_endpoints"].get("nano", "http://gaia-nano:8080")),
            "core": os.environ.get("CORE_INFERENCE_ENDPOINT", cfg["inference_endpoints"].get("core", "http://gaia-core:8092")),
            "prime": os.environ.get("PRIME_INFERENCE_ENDPOINT", cfg["inference_endpoints"].get("prime", "http://gaia-prime:7777")),
        }

        # Safetensors model paths (for state 3 = Conscious/GPU)
        self._gpu_models = {
            "nano": os.environ.get("NANO_SAFETENSORS_PATH", cfg["gpu_models"].get("nano", "/models/nano")),
            "core": os.environ.get("CORE_SAFETENSORS_PATH", cfg["gpu_models"].get("core", "/models/core")),
            "prime": os.environ.get("PRIME_MODEL_PATH", cfg["gpu_models"].get("prime", "/models/prime")),
        }

        # GGUF model paths (for state 2 = Subconscious/CPU)
        self._cpu_models = {
            "nano": os.environ.get("NANO_GGUF_PATH", cfg["cpu_models"].get("nano", "/models/nano.gguf")),
            "core": os.environ.get("CORE_GGUF_PATH", cfg["cpu_models"].get("core", "/models/core.gguf")),
            "prime": os.environ.get("PRIME_GGUF_PATH", cfg["cpu_models"].get("prime", "/models/prime.gguf")),
        }

        # The matrix — one entry per tier
        self._tiers: Dict[str, TierState] = {
            "nano": TierState(tier="nano"),
            "core": TierState(tier="core"),
            "prime": TierState(tier="prime"),
        }

        # Default startup preset — configurable via CONSCIOUSNESS_DEFAULT_PRESET env
        # Options: "awake", "sleep", "unconscious"
        # "awake" = Core+Nano GPU, Prime CPU (normal operation)
        # "unconscious" = everything starts unloaded (manual control)
        self._default_preset = os.environ.get("CONSCIOUSNESS_DEFAULT_PRESET", "awake")

        # Skill adapters to auto-load when a tier enters Subconscious (CPU) mode.
        # GGUF LoRA adapters loaded via /adapter/load with configurable scale.
        self._cpu_adapters = {
            "prime": os.environ.get("PRIME_CPU_ADAPTER", ""),
            "core": os.environ.get("CORE_CPU_ADAPTER", ""),
            "nano": os.environ.get("NANO_CPU_ADAPTER", ""),
        }
        self._adapter_scale = float(os.environ.get("CPU_ADAPTER_SCALE", "0.5"))

        # GPU adapters — identity/skill LoRA loaded after safetensors model loads.
        # Used for models where adapter is not merged into base weights.
        self._gpu_adapters = {
            "prime": os.environ.get("PRIME_GPU_ADAPTER", ""),
            "core": os.environ.get("CORE_GPU_ADAPTER", ""),
            "nano": os.environ.get("NANO_GPU_ADAPTER", ""),
        }
        self._gpu_adapter_scale = float(os.environ.get("GPU_ADAPTER_SCALE", "1.0"))

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

        # Only skip if target matches AND actual matches (truly at target)
        if old_target == level and state.actual == level:
            return {"ok": True, "message": "already at target", "tier": tier, "level": level.name}

        logger.info("Consciousness target: %s %s → %s (actual=%s)",
                     tier, old_target.name, level.name, state.actual.name)

        # Execute the transition using ACTUAL state, not old target.
        # The old target may not reflect reality (e.g., load failed, worker crashed).
        result = await self._transition_tier(tier, state.actual, level)
        return result

    async def probe_all(self) -> Dict[str, dict]:
        """Probe all tiers and update actual states."""
        for tier in self._tiers:
            await self._probe_tier(tier)
        return self.get_matrix()

    async def start_continuous_poll(self, interval: float = 10.0):
        """Start a background task that continuously validates the matrix.

        On first start, applies the default preset (CONSCIOUSNESS_DEFAULT_PRESET env)
        to set tier targets. Auto-reconcile in the poll loop will then load any
        tiers that aren't at their target level.
        """
        if self._poll_task and not self._poll_task.done():
            return

        # Apply startup preset — sets targets so auto-reconcile can load tiers
        preset = self._default_preset
        if preset != "unconscious":
            presets = {
                "awake": {"nano": ConsciousnessLevel.CONSCIOUS, "core": ConsciousnessLevel.CONSCIOUS, "prime": ConsciousnessLevel.SUBCONSCIOUS},
                "sleep": {"nano": ConsciousnessLevel.SUBCONSCIOUS, "core": ConsciousnessLevel.SUBCONSCIOUS, "prime": ConsciousnessLevel.UNCONSCIOUS},
            }
            if preset in presets:
                for tier, level in presets[preset].items():
                    self._tiers[tier].target = level
                logger.info("Consciousness startup preset: %s (targets: %s)",
                            preset, {t: l.name for t, l in presets[preset].items()})

        self._poll_task = asyncio.create_task(self._poll_loop(interval))
        logger.info("Consciousness matrix continuous poll started (%.0fs interval)", interval)

    def stop_poll(self):
        if self._poll_task:
            self._poll_task.cancel()

    # ── Lifecycle FSM Sync ──────────────────────────────────────────

    # Map consciousness configuration names to lifecycle states
    _CONFIG_TO_LIFECYCLE = {
        "awake": "awake",
        "focusing": "focusing",
        "sleep": "sleep",
        "deep_sleep": "deep_sleep",
        "training": "meditation",
    }

    async def _sync_lifecycle(self, config_name: str) -> Optional[dict]:
        """Sync the lifecycle FSM state after a consciousness transition.

        The consciousness matrix handles the actual tier load/unload work.
        This method updates the lifecycle state machine so /lifecycle/state
        reflects the new configuration.
        """
        if not self._lifecycle_machine:
            return None

        try:
            from gaia_common.lifecycle.states import LifecycleState

            target_name = self._CONFIG_TO_LIFECYCLE.get(config_name)
            if not target_name:
                return None

            target = LifecycleState(target_name)
            current = LifecycleState(self._lifecycle_machine._snapshot.state)

            # Already at target — no transition needed
            if current == target:
                return {"lifecycle_sync": "already_at_target", "state": target.value}

            # Use set_state_external since the consciousness matrix has already
            # performed the actual tier load/unload work. We just need the FSM
            # to reflect the new state without re-executing tier actions.
            result = await self._lifecycle_machine.set_state_external(
                target=target,
                reason=f"consciousness_matrix:{config_name}",
            )
            if result.ok:
                logger.info("Lifecycle synced: %s → %s", current.value, target.value)
                return {"lifecycle_sync": "ok", "state": target.value}
            else:
                logger.warning("Lifecycle sync failed: %s", result.error)
                return {"lifecycle_sync": "error", "error": result.error}
        except Exception as e:
            logger.warning("Lifecycle sync failed for %s: %s", config_name, e)
            return {"lifecycle_sync": "error", "error": str(e)[:100]}

    # ── Config Loading ──────────────────────────────────────────────

    @staticmethod
    def _load_config() -> dict:
        """Load tier paths and endpoints from gaia_constants.json."""
        defaults = {
            "inference_endpoints": {"nano": "http://gaia-nano:8080", "core": "http://gaia-core:8092", "prime": "http://gaia-prime:7777"},
            "gpu_models": {"nano": "/models/nano", "core": "/models/core", "prime": "/models/prime"},
            "cpu_models": {"nano": "/models/nano.gguf", "core": "/models/core.gguf", "prime": "/models/prime.gguf"},
        }
        try:
            from gaia_common.config import Config
            cfg = Config.get_instance()
            defaults["inference_endpoints"] = dict(cfg.inference_endpoints)
            for tier in ("nano", "core", "prime"):
                merged = cfg.model_path(tier, "merged")
                if merged:
                    defaults["gpu_models"][tier] = merged
                gguf = cfg.model_path(tier, "gguf")
                if gguf:
                    defaults["cpu_models"][tier] = gguf
        except Exception:
            pass
        return defaults

    # ── Preset Configurations ─────────────────────────────────────────

    # Tier targets for each configuration.
    # Used by both the public preset methods and _apply_configuration().
    # Gemma 4 VRAM reality on 16GB GPU:
    #   Nano NF4 (E2B): 6.3GB | Core NF4 (E4B): 8.8GB | Prime MoE: 4.6GB
    #   Nano + Core = 15.1GB → OOM (no KV cache room)
    #   Only one large tier on GPU at a time; others serve from GGUF/CPU
    _PRESETS = {
        "awake": {"nano": ConsciousnessLevel.CONSCIOUS, "core": ConsciousnessLevel.SUBCONSCIOUS, "prime": ConsciousnessLevel.SUBCONSCIOUS},
        "operating": {"nano": ConsciousnessLevel.SUBCONSCIOUS, "core": ConsciousnessLevel.CONSCIOUS, "prime": ConsciousnessLevel.SUBCONSCIOUS},
        "focusing": {"nano": ConsciousnessLevel.SUBCONSCIOUS, "core": ConsciousnessLevel.SUBCONSCIOUS, "prime": ConsciousnessLevel.CONSCIOUS},
        "sleep": {"nano": ConsciousnessLevel.SUBCONSCIOUS, "core": ConsciousnessLevel.SUBCONSCIOUS, "prime": ConsciousnessLevel.UNCONSCIOUS},
        "deep_sleep": {"nano": ConsciousnessLevel.SUBCONSCIOUS, "core": ConsciousnessLevel.UNCONSCIOUS, "prime": ConsciousnessLevel.UNCONSCIOUS},
        "meditation": {"nano": ConsciousnessLevel.SUBCONSCIOUS, "core": ConsciousnessLevel.SUBCONSCIOUS, "prime": ConsciousnessLevel.UNCONSCIOUS},
    }

    async def _apply_configuration(self, config_name: str, sync_lifecycle: bool = True) -> dict:
        """Apply a consciousness configuration to all tiers.

        Args:
            config_name: One of the preset names (awake, focusing, sleep, etc.)
            sync_lifecycle: If True (default), sync the lifecycle FSM after
                applying. Set to False when the LifecycleMachine is the caller
                (it manages its own FSM state to avoid deadlock).

        Returns:
            Dict with configuration name, per-tier results, and optional lifecycle sync.
        """
        targets = self._PRESETS.get(config_name)
        if not targets:
            return {"ok": False, "error": f"Unknown configuration: {config_name}"}

        results = {}
        for tier, level in targets.items():
            results[tier] = await self.set_target(tier, level)

        result = {"configuration": config_name, "results": results}

        if sync_lifecycle:
            lifecycle = await self._sync_lifecycle(config_name)
            if lifecycle:
                result["lifecycle"] = lifecycle

        # Notify gaia-core to refresh its model pool after tier changes.
        # This clears stale gpu_prime entries that cause ReadTimeout errors.
        await self._notify_core_refresh_pool()

        return result

    async def apply_for_lifecycle(self, config_name: str) -> dict:
        """Apply a configuration on behalf of the LifecycleMachine.

        Skips lifecycle sync (the FSM is already managing its own state).
        This is the method LifecycleMachine._execute_transition() calls
        to delegate actual tier load/unload work.
        """
        logger.info("Consciousness applying config for lifecycle: %s", config_name)
        return await self._apply_configuration(config_name, sync_lifecycle=False)

    async def awake(self) -> dict:
        """AWAKE: Core=3, Nano=3, Prime=2"""
        return await self._apply_configuration("awake")

    async def focusing(self) -> dict:
        """FOCUSING: Nano=3, Core=2, Prime=3"""
        return await self._apply_configuration("focusing")

    async def sleep(self) -> dict:
        """SLEEP: Nano=2, Core=2, Prime=1"""
        return await self._apply_configuration("sleep")

    async def deep_sleep(self) -> dict:
        """DEEP SLEEP: All → 1 (Nano stays 2 for wake detection)"""
        return await self._apply_configuration("deep_sleep")

    async def training(self, tier: str = "prime") -> dict:
        """TRAINING: Target tier → 1 (free GPU), others → 2"""
        # Build custom targets for the specific training tier
        results = {}
        for t in self._tiers:
            if t == tier:
                results[t] = await self.set_target(t, ConsciousnessLevel.UNCONSCIOUS)
            else:
                results[t] = await self.set_target(t, ConsciousnessLevel.SUBCONSCIOUS)
        result = {"configuration": f"training_{tier}", "results": results}
        lifecycle = await self._sync_lifecycle("training")
        if lifecycle:
            result["lifecycle"] = lifecycle
        return result

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

                    device = data.get("device", "unknown")

                    if managed:
                        # Managed engine — check if worker is active
                        if model_loaded:
                            if backend in ("gguf", "cpp") or device == "cpu" or data.get("has_gpu") is False:
                                state.actual = ConsciousnessLevel.SUBCONSCIOUS
                            elif device in ("cuda", "gpu") or data.get("has_gpu") is True:
                                state.actual = ConsciousnessLevel.CONSCIOUS
                            elif mode == "active":
                                state.actual = ConsciousnessLevel.CONSCIOUS
                            else:
                                # Loaded but can't determine device — assume conscious
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
                # First unload current if anything is loaded
                if from_level in (ConsciousnessLevel.CONSCIOUS, ConsciousnessLevel.SUBCONSCIOUS):
                    unload_result = await self._unload_tier(tier, endpoint, state)
                    if not unload_result.get("ok"):
                        logger.warning("Unload %s before CPU load returned: %s", tier, unload_result)
                    # Wait for engine manager to be ready after unload
                    await self._wait_for_manager_ready(tier, endpoint)
                return await self._load_tier_cpu(tier, endpoint, state)

            elif to_level == ConsciousnessLevel.CONSCIOUS:
                # Load safetensors on GPU
                if from_level in (ConsciousnessLevel.CONSCIOUS, ConsciousnessLevel.SUBCONSCIOUS):
                    unload_result = await self._unload_tier(tier, endpoint, state)
                    if not unload_result.get("ok"):
                        logger.warning("Unload %s before GPU load returned: %s", tier, unload_result)
                    await self._wait_for_manager_ready(tier, endpoint)
                return await self._load_tier_gpu(tier, endpoint, state)

        except Exception as e:
            state.error = str(e)[:200]
            logger.error("Transition failed for %s: %s", tier, e)
            return {"ok": False, "tier": tier, "error": str(e)}
        finally:
            state.transitioning = False
            await self._probe_tier(tier)

    async def _wait_for_manager_ready(self, tier: str, endpoint: str,
                                       timeout: float = 10.0):
        """Wait for the engine manager to be responsive in standby mode.

        After an unload, the manager needs a moment to finish killing the
        worker and return to standby. This polls /health until we get a
        managed response, ensuring the subsequent /model/load won't hit a
        blocked or unavailable server.
        """
        deadline = asyncio.get_event_loop().time() + timeout
        attempt = 0
        while asyncio.get_event_loop().time() < deadline:
            attempt += 1
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(f"{endpoint}/health")
                    if resp.status_code == 200:
                        data = resp.json()
                        # Manager is ready when it responds with managed=true
                        if data.get("managed"):
                            logger.debug("Manager %s ready after %d attempts (mode=%s)",
                                         tier, attempt, data.get("mode"))
                            return
            except Exception:
                pass
            await asyncio.sleep(0.5)
        logger.warning("Manager %s not confirmed ready after %.0fs — proceeding anyway", tier, timeout)

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

                        # Auto-load identity/skill adapter if configured
                        adapter_path = self._gpu_adapters.get(tier, "")
                        if adapter_path:
                            await self._load_adapter(tier, endpoint, adapter_path,
                                                     scale=self._gpu_adapter_scale)

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

                        # Auto-load skill adapter if configured
                        adapter_path = self._cpu_adapters.get(tier, "")
                        if adapter_path:
                            await self._load_adapter(tier, endpoint, adapter_path)

                        return {"ok": True, "tier": tier, "action": "loaded_cpu", "model": model}
                return {"ok": False, "tier": tier, "error": f"HTTP {resp.status_code}: {resp.text[:100]}"}
        except Exception as e:
            return {"ok": False, "tier": tier, "error": str(e)}

    async def _load_adapter(self, tier: str, endpoint: str, adapter_path: str) -> None:
        """Load a GGUF LoRA adapter on a tier's engine."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{endpoint}/adapter/load",
                    json={"adapter_path": adapter_path, "scale": self._adapter_scale},
                )
                if resp.status_code == 200 and resp.json().get("ok"):
                    logger.info("Loaded adapter on %s: %s (scale=%.2f)",
                                tier, adapter_path, self._adapter_scale)
                else:
                    logger.warning("Adapter load on %s returned %d: %s",
                                   tier, resp.status_code, resp.text[:100])
        except Exception as e:
            logger.warning("Adapter load on %s failed: %s", tier, e)

    async def _notify_core_refresh_pool(self):
        """Tell gaia-core to clear stale GPU model entries from its pool.

        Non-blocking — if Core is unreachable we log and continue.
        """
        core_endpoint = self._endpoints.get("core", "").replace(":8092", ":6415")
        if not core_endpoint:
            # Fallback: use the service endpoint from config
            try:
                from gaia_common.config import Config
                core_endpoint = Config.get_instance().get_endpoint("core")
            except Exception:
                core_endpoint = "http://gaia-core:6415"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(f"{core_endpoint}/refresh_pool")
                if resp.status_code == 200:
                    data = resp.json()
                    removed = data.get("removed", {})
                    if removed:
                        logger.info("Core model pool refreshed: removed %s", list(removed.keys()))
                    else:
                        logger.debug("Core model pool refresh: no stale entries")
                else:
                    logger.debug("Core /refresh_pool returned %d", resp.status_code)
        except Exception as e:
            logger.debug("Core /refresh_pool call failed (non-blocking): %s", e)

    # ── Internal: Continuous Poll ─────────────────────────────────────

    def _is_meditation_active(self) -> bool:
        """Check if the lifecycle FSM is in MEDITATION (study owns GPU).

        During MEDITATION, tiers are intentionally unloaded. Auto-reconcile
        must NOT reload them or it will fight with training for VRAM.
        """
        if not self._lifecycle_machine:
            return False
        try:
            from gaia_common.lifecycle.states import LifecycleState
            current = self._lifecycle_machine._snapshot.state
            if isinstance(current, str):
                current = LifecycleState(current)
            return current == LifecycleState.MEDITATION
        except Exception:
            return False

    async def _poll_loop(self, interval: float):
        """Continuously probe tiers, validate matrix, and auto-reconcile drift."""
        while True:
            try:
                await self.probe_all()

                # Skip auto-reconcile during MEDITATION — tiers are intentionally
                # unloaded while Study owns the GPU for training.
                if self._is_meditation_active():
                    logger.debug("Consciousness poll: MEDITATION active — skipping auto-reconcile")
                    await asyncio.sleep(interval)
                    continue

                # Auto-reconcile: if target > actual, load the tier
                for tier, state in self._tiers.items():
                    if state.ok or state.transitioning:
                        continue
                    if state.target > state.actual and state.healthy:
                        logger.warning(
                            "Matrix mismatch: %s target=%s actual=%s — auto-reconciling",
                            tier, state.target.name, state.actual.name,
                        )
                        try:
                            result = await self._transition_tier(
                                tier, state.actual, state.target,
                            )
                            if result.get("ok"):
                                logger.info("Auto-reconciled %s → %s", tier, state.target.name)
                            else:
                                logger.warning("Auto-reconcile %s failed: %s", tier, result.get("error", ""))
                        except Exception as e:
                            logger.warning("Auto-reconcile %s error: %s", tier, e)
                    elif not state.ok and not state.transitioning:
                        logger.warning(
                            "Matrix mismatch: %s target=%s actual=%s healthy=%s error=%s",
                            tier, state.target.name, state.actual.name,
                            state.healthy, state.error[:50] if state.error else ""
                        )
            except Exception as e:
                logger.debug("Poll error: %s", e)
            await asyncio.sleep(interval)

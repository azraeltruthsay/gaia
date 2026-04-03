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

from gaia_common.utils.maintenance import is_maintenance_active

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

        # Tier engine endpoints
        self._endpoints = {
            "nano": os.environ.get("NANO_INFERENCE_ENDPOINT", "http://gaia-nano:8080"),
            "core": os.environ.get("CORE_INFERENCE_ENDPOINT", "http://gaia-core:8092"),
            "prime": os.environ.get("PRIME_INFERENCE_ENDPOINT", "http://gaia-prime:7777"),
        }

        # Safetensors model paths (for state 3 = Conscious/GPU)
        self._gpu_models = {
            "nano": os.environ.get("NANO_SAFETENSORS_PATH", "/models/nano"),
            "core": os.environ.get("CORE_SAFETENSORS_PATH", "/models/core"),
            "prime": os.environ.get("PRIME_MODEL_PATH", "/models/prime"),
        }

        # GGUF model paths (for state 2 = Subconscious/CPU)
        self._cpu_models = {
            "nano": os.environ.get("NANO_GGUF_PATH", "/models/nano.gguf"),
            "core": os.environ.get("CORE_GGUF_PATH", "/models/core.gguf"),
            "prime": os.environ.get("PRIME_GGUF_PATH", "/models/prime.gguf"),
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

        self._lock = asyncio.Lock()
        self._poll_task: Optional[asyncio.Task] = None
        # Cooldown: suppress auto-reconcile after execute_shift to prevent
        # the poll loop from immediately undoing a transition
        self._shift_cooldown_until: float = 0.0

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

    # ── Clutch Protocol (Lifecycle-Delegated Execution) ────────────────

    async def execute_shift(self, from_state, to_state) -> dict:
        """Execute a full tier reconfiguration for a lifecycle state transition.

        Called by LifecycleMachine._execute_transition. This is THE method
        that moves models. Implements the clutch protocol:
          1. ENGAGE CLUTCH — drain changing tiers (finish in-flight inference)
          2. SHIFT GEARS — unload/load tiers to match target state
          3. RELEASE CLUTCH — resume inference on new configuration
        """
        from gaia_common.lifecycle.states import TIER_EXPECTATIONS

        target_expectations = TIER_EXPECTATIONS.get(to_state, {})
        from_expectations = TIER_EXPECTATIONS.get(from_state, {})
        actions = []

        # Compute which tiers are changing
        tiers_changing = self._compute_changing_tiers(from_expectations, target_expectations)
        logger.info("Clutch: %s → %s (changing: %s)", from_state, to_state, tiers_changing)

        # Phase 1: ENGAGE CLUTCH — drain changing tiers
        if tiers_changing:
            drain_results = await self._engage_clutch(tiers_changing)
            actions.append(f"clutch:engaged({','.join(tiers_changing)})")
            for tier, result in drain_results.items():
                if result.get("ok"):
                    logger.info("Clutch: %s drained", tier)
                else:
                    logger.warning("Clutch: %s drain failed: %s", tier, result.get("error", ""))

        try:
            # Phase 2: SHIFT GEARS — compute target consciousness levels
            tier_targets = self._expectations_to_consciousness(target_expectations)

            # Unload phase (downgrading tiers — free VRAM first)
            did_unload = False
            for tier, target_level in tier_targets.items():
                current_level = self._tiers[tier].actual
                if target_level < current_level:
                    result = await self._transition_tier(tier, current_level, target_level)
                    actions.append(f"{tier}:{current_level.name}->{target_level.name}")
                    if result.get("ok"):
                        did_unload = True
                    else:
                        logger.warning("Clutch: %s downgrade failed: %s", tier, result.get("error"))

            # CUDA cleanup delay if we unloaded anything from GPU
            if did_unload:
                await asyncio.sleep(1.5)

            # Load phase (upgrading tiers)
            for tier, target_level in tier_targets.items():
                current_level = self._tiers[tier].actual
                if target_level > current_level:
                    result = await self._transition_tier(tier, current_level, target_level)
                    actions.append(f"{tier}:{current_level.name}->{target_level.name}")
                    if not result.get("ok"):
                        raise RuntimeError(f"{tier} load failed: {result.get('error', 'unknown')}")

            # Verify
            await self.probe_all()

            # Update targets to match the new state
            for tier, level in tier_targets.items():
                self._tiers[tier].target = level

            # Suppress auto-reconcile for 60s so the poll loop doesn't
            # immediately undo this transition by reverting to startup targets
            self._shift_cooldown_until = time.time() + 30.0
            logger.info("Clutch: shift complete, auto-reconcile suppressed for 30s")

            return {"ok": True, "actions": actions}

        except Exception as e:
            logger.error("Clutch: shift failed: %s", e)
            return {"ok": False, "error": str(e), "actions": actions}

        finally:
            # Phase 3: RELEASE CLUTCH — always resume, even on error
            if tiers_changing:
                await self._release_clutch(tiers_changing)
                actions.append("clutch:released")

    async def _engage_clutch(self, tiers: list) -> dict:
        """Drain inference on specified tiers before model swap.

        Sends POST /inference/drain to each tier's engine. If the endpoint
        doesn't exist (pre-Phase-0 engine), proceeds without drain.
        """
        results = {}
        for tier in tiers:
            endpoint = self._endpoints.get(tier)
            if not endpoint:
                continue
            try:
                async with httpx.AsyncClient(timeout=35.0) as client:
                    resp = await client.post(
                        f"{endpoint}/inference/drain",
                        json={"timeout_s": 30},
                    )
                    results[tier] = resp.json() if resp.status_code == 200 else {"ok": False, "error": f"HTTP {resp.status_code}"}
            except Exception as e:
                # Engine may not have drain endpoint yet — proceed gracefully
                logger.debug("Clutch: drain %s failed (may not support drain yet): %s", tier, e)
                results[tier] = {"ok": False, "error": str(e)[:100]}
        return results

    async def _release_clutch(self, tiers: list) -> dict:
        """Resume inference on specified tiers after model swap."""
        results = {}
        for tier in tiers:
            endpoint = self._endpoints.get(tier)
            if not endpoint:
                continue
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post(f"{endpoint}/inference/resume")
                    results[tier] = resp.json() if resp.status_code == 200 else {"ok": False}
            except Exception:
                pass  # Engine may have been unloaded — that's fine
        return results

    @staticmethod
    def _compute_changing_tiers(from_exp: dict, to_exp: dict) -> list:
        """Determine which tiers are changing state between lifecycle states."""
        changing = []
        all_tiers = set(list(from_exp.keys()) + list(to_exp.keys()))
        for tier in all_tiers:
            if tier in ("study", "audio"):
                continue  # Not managed by consciousness matrix
            from_device = from_exp[tier].device if tier in from_exp else "unloaded"
            to_device = to_exp[tier].device if tier in to_exp else "unloaded"
            if from_device != to_device:
                changing.append(tier)
        return changing

    @staticmethod
    def _expectations_to_consciousness(expectations: dict) -> dict:
        """Map lifecycle TierExpectations to ConsciousnessLevel per tier."""
        mapping = {
            "gpu": ConsciousnessLevel.CONSCIOUS,
            "cpu": ConsciousnessLevel.SUBCONSCIOUS,
            "unloaded": ConsciousnessLevel.UNCONSCIOUS,
        }
        result = {}
        for tier, exp in expectations.items():
            if tier in ("study", "audio"):
                continue  # Not managed by consciousness matrix
            result[tier] = mapping.get(exp.device, ConsciousnessLevel.UNCONSCIOUS)
        return result

    # ── Preset Configurations ─────────────────────────────────────────

    async def awake(self) -> dict:
        """AWAKE: Core+Nano GPU, Prime CPU.

        Delegates through the lifecycle machine (which calls back to execute_shift).
        Falls back to direct execution if lifecycle machine isn't available.
        """
        if self._lifecycle_machine:
            from gaia_common.lifecycle.states import TransitionTrigger, LifecycleState
            result = await self._lifecycle_machine.transition(
                TransitionTrigger.USER_REQUEST, target=LifecycleState.AWAKE,
                reason="consciousness_matrix:awake")
            return {"configuration": "awake", "lifecycle": result.__dict__ if hasattr(result, '__dict__') else str(result)}
        return await self._direct_awake()

    async def _direct_awake(self) -> dict:
        """Direct execution fallback for awake (no lifecycle machine)."""
        results = {}
        results["nano"] = await self.set_target("nano", ConsciousnessLevel.CONSCIOUS)
        results["core"] = await self.set_target("core", ConsciousnessLevel.CONSCIOUS)
        results["prime"] = await self.set_target("prime", ConsciousnessLevel.SUBCONSCIOUS)
        return {"configuration": "awake", "results": results}

    async def focusing(self) -> dict:
        """FOCUSING: Core+Nano off GPU, Prime on GPU.

        Delegates through the lifecycle machine (which calls back to execute_shift).
        The clutch protocol handles VRAM safety — drain, unload, load, resume.
        """
        if self._lifecycle_machine:
            from gaia_common.lifecycle.states import TransitionTrigger, LifecycleState
            result = await self._lifecycle_machine.transition(
                TransitionTrigger.USER_REQUEST, target=LifecycleState.FOCUSING,
                reason="consciousness_matrix:focusing")
            return {"configuration": "focusing", "lifecycle": result.__dict__ if hasattr(result, '__dict__') else str(result)}
        return await self._direct_focusing()

    async def _direct_focusing(self) -> dict:
        """Direct execution fallback for focusing (no lifecycle machine)."""
        results = {}
        results["core"] = await self.set_target("core", ConsciousnessLevel.UNCONSCIOUS)
        results["nano"] = await self.set_target("nano", ConsciousnessLevel.SUBCONSCIOUS)
        await asyncio.sleep(2)
        results["prime"] = await self.set_target("prime", ConsciousnessLevel.CONSCIOUS)
        return {"configuration": "focusing", "results": results}

    async def sleep(self) -> dict:
        """SLEEP: Nano+Core CPU, Prime unloaded."""
        if self._lifecycle_machine:
            from gaia_common.lifecycle.states import TransitionTrigger, LifecycleState
            result = await self._lifecycle_machine.transition(
                TransitionTrigger.USER_REQUEST, target=LifecycleState.SLEEP,
                reason="consciousness_matrix:sleep")
            return {"configuration": "sleep", "lifecycle": result.__dict__ if hasattr(result, '__dict__') else str(result)}
        return await self._direct_sleep()

    async def _direct_sleep(self) -> dict:
        """Direct execution fallback for sleep (no lifecycle machine)."""
        results = {}
        results["prime"] = await self.set_target("prime", ConsciousnessLevel.UNCONSCIOUS)
        results["core"] = await self.set_target("core", ConsciousnessLevel.SUBCONSCIOUS)
        results["nano"] = await self.set_target("nano", ConsciousnessLevel.SUBCONSCIOUS)
        return {"configuration": "sleep", "results": results}

    async def deep_sleep(self) -> dict:
        """DEEP SLEEP: All unloaded except Nano CPU for wake detection."""
        if self._lifecycle_machine:
            from gaia_common.lifecycle.states import TransitionTrigger, LifecycleState
            result = await self._lifecycle_machine.transition(
                TransitionTrigger.USER_REQUEST, target=LifecycleState.DEEP_SLEEP,
                reason="consciousness_matrix:deep_sleep")
            return {"configuration": "deep_sleep", "lifecycle": result.__dict__ if hasattr(result, '__dict__') else str(result)}
        return await self._direct_deep_sleep()

    async def _direct_deep_sleep(self) -> dict:
        """Direct execution fallback for deep_sleep (no lifecycle machine)."""
        results = {}
        results["prime"] = await self.set_target("prime", ConsciousnessLevel.UNCONSCIOUS)
        results["core"] = await self.set_target("core", ConsciousnessLevel.UNCONSCIOUS)
        results["nano"] = await self.set_target("nano", ConsciousnessLevel.SUBCONSCIOUS)
        return {"configuration": "deep_sleep", "results": results}

    async def training(self, tier: str = "prime") -> dict:
        """TRAINING: Target tier unloaded (free GPU), others CPU."""
        if self._lifecycle_machine:
            from gaia_common.lifecycle.states import TransitionTrigger, LifecycleState
            result = await self._lifecycle_machine.transition(
                TransitionTrigger.TRAINING_SCHEDULED, target=None,
                reason=f"consciousness_matrix:training_{tier}")
            return {"configuration": f"training_{tier}", "lifecycle": result.__dict__ if hasattr(result, '__dict__') else str(result)}
        return await self._direct_training(tier)

    async def _direct_training(self, tier: str = "prime") -> dict:
        """Direct execution fallback for training (no lifecycle machine)."""
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

                    device = data.get("device", "unknown")

                    if managed:
                        # Managed engine — check if worker is active
                        if mode == "active" and model_loaded:
                            if backend == "gguf" or device == "cpu":
                                state.actual = ConsciousnessLevel.SUBCONSCIOUS
                            elif device == "cuda" or device == "gpu":
                                state.actual = ConsciousnessLevel.CONSCIOUS
                            else:
                                # Unknown device — infer from backend
                                state.actual = ConsciousnessLevel.CONSCIOUS if backend == "engine" else ConsciousnessLevel.SUBCONSCIOUS
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
                # Try fast migration to CPU first (keeps weights in RAM for quick reload)
                if from_level == ConsciousnessLevel.CONSCIOUS:
                    migrate_result = await self._try_migrate(tier, endpoint, "cpu")
                    if migrate_result and migrate_result.get("ok"):
                        state.actual = ConsciousnessLevel.UNCONSCIOUS
                        state.gpu_mb = 0
                        logger.info("Fast path: %s migrated GPU→CPU (%.1fs)",
                                    tier, migrate_result.get("elapsed_s", 0))
                        return {"ok": True, "tier": tier, "action": "migrated_to_cpu",
                                "elapsed_s": migrate_result.get("elapsed_s", 0)}
                # Fallback: full unload
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
                # Try fast migration from CPU first (if worker still has weights in RAM)
                migrate_result = await self._try_migrate(tier, endpoint, "cuda")
                if migrate_result and not migrate_result.get("ok"):
                    logger.info("Migration %s→GPU declined: %s", tier, migrate_result.get("error", "unknown"))
                if migrate_result and migrate_result.get("ok"):
                    state.actual = ConsciousnessLevel.CONSCIOUS
                    state.gpu_mb = migrate_result.get("vram_mb", 0)
                    logger.info("Fast path: %s migrated CPU→GPU (%.1fs, %dMB)",
                                tier, migrate_result.get("elapsed_s", 0),
                                migrate_result.get("vram_mb", 0))
                    return {"ok": True, "tier": tier, "action": "migrated_to_gpu",
                            "elapsed_s": migrate_result.get("elapsed_s", 0)}
                # Fallback: full load from disk
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

    async def _try_migrate(self, tier: str, endpoint: str, target_device: str) -> Optional[dict]:
        """Try to migrate a tier's model between GPU and CPU via /model/migrate.

        Returns the migration result, or None if migration isn't supported
        (e.g., GGUF backend, no active worker, endpoint not available).
        This is the fast path (~5s) vs full unload+reload (~95s).
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{endpoint}/model/migrate",
                    json={"device": target_device},
                )
                if resp.status_code == 200:
                    return resp.json()
                logger.info("Migration %s→%s returned HTTP %d — using fallback path",
                            tier, target_device, resp.status_code)
        except Exception as e:
            logger.info("Migration %s→%s failed: %s — using fallback path", tier, target_device, e)
        return None

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
        """Load a tier on GPU (safetensors → conscious).

        If the tier already has a GGUF/CPU model loaded, unloads it first
        and waits for the manager to return to standby before loading GPU.
        The GAIA Engine auto-detects when NF4 quantization is needed based
        on available VRAM.
        """
        model = self._gpu_models.get(tier)
        if not model:
            return {"ok": False, "tier": tier, "error": "no GPU model configured"}

        logger.info("Loading %s to GPU: %s", tier, model)
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                # Check current state — may need to unload GGUF first
                health = await client.get(f"{endpoint}/health")
                if health.status_code == 200:
                    hdata = health.json()
                    if hdata.get("model_loaded") and hdata.get("device") == "cuda":
                        state.actual = ConsciousnessLevel.CONSCIOUS
                        logger.info("%s already loaded on GPU", tier)
                        return {"ok": True, "tier": tier, "action": "already_loaded_gpu"}
                    _has_model = hdata.get("model_loaded") or hdata.get("status") == "ok"
                    if _has_model:
                        # Model loaded but not on GPU (GGUF/CPU) — unload first
                        logger.info("%s has model on %s — unloading before GPU load",
                                    tier, hdata.get("device", "cpu"))
                        await self._unload_tier(tier, endpoint, state)
                        await self._wait_for_manager_ready(tier, endpoint)

                resp = await client.post(
                    f"{endpoint}/model/load",
                    json={"model": model, "device": "cuda"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("ok"):
                        state.actual = ConsciousnessLevel.CONSCIOUS
                        logger.info("Loaded %s to GPU", tier)
                        return {"ok": True, "tier": tier, "action": "loaded_gpu", "model": model}
                if resp.status_code == 409:
                    # 409 = already loaded — verify it's actually on GPU
                    h2 = await client.get(f"{endpoint}/health")
                    if h2.status_code == 200 and h2.json().get("device") == "cuda":
                        state.actual = ConsciousnessLevel.CONSCIOUS
                        logger.info("Loaded %s to GPU (409 = already loaded)", tier)
                        return {"ok": True, "tier": tier, "action": "loaded_gpu", "model": model}
                    logger.warning("%s 409 but not on GPU — load may have failed", tier)
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

    # ── Internal: Continuous Poll ─────────────────────────────────────

    async def _poll_loop(self, interval: float):
        """Continuously probe tiers, validate matrix, and auto-reconcile drift."""
        while True:
            try:
                await self.probe_all()

                # Skip auto-reconcile during maintenance, active transition, or cooldown
                if is_maintenance_active():
                    await asyncio.sleep(15)
                    continue
                if time.time() < self._shift_cooldown_until:
                    logger.debug("Poll: skipping auto-reconcile — shift cooldown active")
                    await asyncio.sleep(interval)
                    continue
                if self._lifecycle_machine:
                    from gaia_common.lifecycle.states import LifecycleState as _LS
                    current = self._lifecycle_machine._snapshot.state
                    if current == _LS.TRANSITIONING or current == "transitioning":
                        logger.debug("Poll: skipping auto-reconcile — lifecycle is TRANSITIONING")
                        await asyncio.sleep(interval)
                        continue

                # Auto-reconcile: if target != actual, load/unload the tier
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
                                # Sync lifecycle after drift recovery
                                await self._sync_lifecycle(
                                    self._infer_config_from_matrix())
                            else:
                                logger.warning("Auto-reconcile %s failed: %s", tier, result.get("error", ""))
                        except Exception as e:
                            logger.warning("Auto-reconcile %s error: %s", tier, e)
                    elif not state.ok and not state.transitioning:
                        logger.debug(
                            "Matrix mismatch: %s target=%s actual=%s healthy=%s error=%s",
                            tier, state.target.name, state.actual.name,
                            state.healthy, state.error[:50] if state.error else ""
                        )
            except Exception as e:
                logger.debug("Poll error: %s", e)
            await asyncio.sleep(interval)

    def _infer_config_from_matrix(self) -> str:
        """Infer the current configuration name from the consciousness matrix state."""
        nano = self._tiers["nano"].actual
        core = self._tiers["core"].actual
        prime = self._tiers["prime"].actual

        if prime == ConsciousnessLevel.CONSCIOUS:
            return "focusing"
        if core == ConsciousnessLevel.CONSCIOUS and nano == ConsciousnessLevel.CONSCIOUS:
            return "awake"
        if core == ConsciousnessLevel.SUBCONSCIOUS and nano == ConsciousnessLevel.SUBCONSCIOUS:
            return "sleep"
        if core == ConsciousnessLevel.UNCONSCIOUS and nano == ConsciousnessLevel.SUBCONSCIOUS:
            return "deep_sleep"
        return "awake"  # Default fallback

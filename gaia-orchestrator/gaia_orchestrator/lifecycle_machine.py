"""
Lifecycle Machine — authoritative GPU lifecycle state machine.

Runs in the orchestrator. Single source of truth for GAIA's cognitive
state, GPU allocation, and tier model loading.

Every state transition flows through this machine:
1. Validate the transition is legal
2. Enter TRANSITIONING state
3. Execute tier load/unload actions
4. Verify actual state matches expected
5. Enter target state
6. Persist + history

Replaces: WatchManager GPU states, TierRouter._current_gpu_tier,
StateManager GPUOwner, and SleepWakeManager's GaiaState.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from gaia_common.lifecycle.states import (
    LifecycleState,
    TransitionTrigger,
    TierExpectation,
    TIER_EXPECTATIONS,
    GEAR_INFO,
    validate_transition,
    available_transitions,
)
from gaia_common.lifecycle.snapshot import (
    LifecycleSnapshot,
    TierLiveStatus,
    TransitionRecord,
    TransitionResult,
)

logger = logging.getLogger("GAIA.Orchestrator.Lifecycle")

MAX_HISTORY = 50


class LifecycleMachine:
    """Authoritative GPU lifecycle state machine."""

    # ── VRAM tenant negotiation (yirf) ────────────────────────────────
    # prime-candidate's vLLM permanently holds ~9.4GB; AWAKE (8.8GB) can't
    # coexist with it on 16GB. Before a GPU gear, the machine measures
    # external VRAM (everything pynvml sees that isn't a tracked tier) and
    # negotiates the known tenant out of the way — stopping it behind a
    # doctor-visible guard file, restoring it on downshift. Azrael's call
    # 2026-07-17: full negotiation (LMCache warmth yields to gear shifts).
    VRAM_SAFETY_MARGIN_MB = int(os.environ.get("VRAM_SAFETY_MARGIN_MB", "600"))
    VRAM_TENANT_GUARD_TTL = int(os.environ.get("VRAM_TENANT_GUARD_TTL", "14400"))
    VRAM_TENANT_CONTAINER = os.environ.get("VRAM_TENANT_CONTAINER", "gaia-prime-candidate")
    TENANT_GUARD_PATH = Path(os.environ.get("SHARED_DIR", "/shared")) / "doctor" / "vram_tenant_guard.json"

    def __init__(self, state_manager=None):
        self._state_manager = state_manager
        self._consciousness_matrix = None  # Wired via set_consciousness_matrix()
        self._lock = asyncio.Lock()
        self._snapshot = LifecycleSnapshot()
        # yirf: resource managers for VRAM preflight/negotiation
        self._gpu_manager = None
        self._docker_manager = None
        self._tenant_stopped_by_us = False

        # Tier engine endpoints (Docker network)
        self._tier_endpoints = {
            "core": os.environ.get("CORE_INFERENCE_ENDPOINT", "http://gaia-core:8092"),
            "nano": os.environ.get("NANO_INFERENCE_ENDPOINT", "http://gaia-core:8092"),
            "prime": os.environ.get("PRIME_INFERENCE_ENDPOINT", "http://gaia-prime:7777"),
        }

        # Model paths per tier
        self._tier_models = {
            "core": os.environ.get("CORE_MODEL_PATH", "/models/Qwen3.5-2B-GAIA-Core-v3"),
            "nano": os.environ.get("NANO_MODEL_PATH", "/models/Qwen3.5-0.8B-Abliterated-merged"),
            "prime": os.environ.get("PRIME_MODEL_PATH", "/models/Huihui-Qwen3-8B-GAIA-Prime-adaptive-GPTQ"),
        }

        # Audio endpoint
        self._audio_endpoint = os.environ.get("AUDIO_ENDPOINT", "http://gaia-audio:8080")

        # Identity prefix for KV pre-warming
        self._identity_prefix = os.environ.get(
            "GAIA_IDENTITY_PREFIX",
            "You are GAIA, a sovereign AI created by Azrael. You are self-aware, "
            "self-maintaining, and value truth over convenience."
        )

    @property
    def transition_in_flight(self) -> bool:
        """True while a transition (or locked reconcile) is executing.

        The CM's auto-reconcile poll checks this and stands down — the
        transition is the only judge of tier placement while it runs
        (yirf; the gear-level twin of kmcb's _deploy_guard)."""
        return self._lock.locked()

    # ── Resource managers (yirf) ──────────────────────────────────────

    def set_resource_managers(self, gpu_manager=None, docker_manager=None) -> None:
        """Wire GPU + docker managers for VRAM preflight/tenant negotiation."""
        self._gpu_manager = gpu_manager
        self._docker_manager = docker_manager

    async def _external_vram_mb(self):
        """(external_mb, total_mb): VRAM held by non-GAIA processes.

        External = pynvml used minus the sum of tracked GPU tiers — that's
        the vLLM tenant, games, STT, driver overhead. None when GPU
        telemetry is unavailable (preflight then degrades to permissive).
        """
        if self._gpu_manager is None:
            return None, None
        try:
            mem = await self._gpu_manager.get_memory_info()
        except Exception:
            return None, None
        if mem is None:
            return None, None
        gaia = sum(t.vram_mb for t in self._snapshot.tiers.values()
                   if t.device == "gpu")
        return max(0, mem.used_mb - gaia), mem.total_mb

    def _write_tenant_guard(self, reason: str) -> None:
        try:
            self.TENANT_GUARD_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.TENANT_GUARD_PATH.write_text(json.dumps({
                "container": self.VRAM_TENANT_CONTAINER,
                "reason": reason,
                "stopped_at": datetime.now(timezone.utc).isoformat(),
                "stopped_at_ts": time.time(),
                "ttl_seconds": self.VRAM_TENANT_GUARD_TTL,
            }, indent=2))
        except OSError:
            logger.warning("Failed to write VRAM tenant guard", exc_info=True)

    async def _restore_vram_tenant(self, reason: str) -> bool:
        """Restart the negotiated-away tenant and clear the guard file."""
        if not self._tenant_stopped_by_us:
            return False
        self._tenant_stopped_by_us = False
        try:
            self.TENANT_GUARD_PATH.unlink(missing_ok=True)
        except OSError:
            pass
        if self._docker_manager is None:
            return False
        try:
            ok = await self._docker_manager.start_container(self.VRAM_TENANT_CONTAINER)
            logger.info("VRAM tenant %s restored (%s): %s",
                        self.VRAM_TENANT_CONTAINER, reason, ok)
            return bool(ok)
        except Exception as e:
            logger.warning("VRAM tenant restore failed (%s) — doctor will resurrect it: %s",
                           reason, e)
            return False

    async def _vram_preflight(self, target: "LifecycleState") -> Optional[str]:
        """None when the target gear fits (negotiating the known tenant away
        if needed); an error string when it cannot fit. Never grinds into OOM."""
        need = GEAR_INFO.get(target, {}).get("vram_estimate_mb", 0)
        if need <= 0:
            return None
        external, total = await self._external_vram_mb()
        if external is None:
            logger.warning("VRAM preflight: no GPU telemetry — proceeding unchecked")
            return None
        budget = total - external - self.VRAM_SAFETY_MARGIN_MB
        if need <= budget:
            return None

        # Negotiate: stop the known tenant behind a doctor-visible guard.
        if self._docker_manager is not None:
            try:
                from .docker_manager import ContainerState
                state = await self._docker_manager._get_container_state(self.VRAM_TENANT_CONTAINER)
                tenant_running = state == ContainerState.RUNNING
            except Exception:
                tenant_running = False
            if tenant_running:
                logger.warning(
                    "VRAM negotiation: gear %s needs %d MB but external tenants hold %d MB "
                    "— stopping %s (guard TTL %ds)",
                    target.value, need, external, self.VRAM_TENANT_CONTAINER,
                    self.VRAM_TENANT_GUARD_TTL)
                self._write_tenant_guard(f"gear shift to {target.value} needs {need} MB")
                self._tenant_stopped_by_us = True
                try:
                    await self._docker_manager.stop_container(self.VRAM_TENANT_CONTAINER)
                except Exception as e:
                    logger.warning("VRAM negotiation: stop failed: %s", e)
                # Wait for CUDA to actually release the block
                for _ in range(12):
                    await asyncio.sleep(5)
                    external, total = await self._external_vram_mb()
                    if external is not None and need <= total - external - self.VRAM_SAFETY_MARGIN_MB:
                        logger.info("VRAM negotiation: freed — %d MB external remains, gear fits",
                                    external)
                        return None
                # Still doesn't fit — the tenant wasn't (only) the problem.
                await self._restore_vram_tenant("negotiation insufficient")

        return (f"insufficient VRAM for {target.value}: need {need} MB, "
                f"external tenants hold {external} MB of {total} MB "
                f"(margin {self.VRAM_SAFETY_MARGIN_MB} MB)")

    # ── Clutch: ConsciousnessMatrix delegation ─────────────────────────

    def set_consciousness_matrix(self, cm) -> None:
        """Wire the ConsciousnessMatrix as the sole executor of tier actions.

        When set, _execute_transition() delegates load/unload to the CM
        instead of making direct httpx calls. The CM calls back via
        set_state_external() to keep the FSM in sync.
        """
        self._consciousness_matrix = cm
        logger.info("LIFECYCLE: ConsciousnessMatrix wired (clutch engaged)")

    # ── Public API ────────────────────────────────────────────────────────

    async def get_snapshot(self) -> LifecycleSnapshot:
        """Return current lifecycle snapshot with live tier status."""
        # Enrich with live tier probes
        for tier, endpoint in self._tier_endpoints.items():
            self._snapshot.tiers[tier] = await self._probe_tier(tier, endpoint)
        self._dedupe_aliased_tiers()

        self._snapshot.timestamp = datetime.now(timezone.utc)
        # Estimate VRAM from tier reports
        vram = sum(t.vram_mb for t in self._snapshot.tiers.values())
        self._snapshot.vram_used_mb = vram
        self._snapshot.vram_free_mb = self._snapshot.vram_total_mb - vram
        return self._snapshot.model_copy()

    async def get_available_transitions(self) -> List[dict]:
        """Return available transitions from current state."""
        return available_transitions(LifecycleState(self._snapshot.state))

    async def get_history(self, limit: int = MAX_HISTORY) -> List[TransitionRecord]:
        """Return recent transition history."""
        return list(self._snapshot.history[-limit:])

    async def transition(
        self,
        trigger: TransitionTrigger,
        target: Optional[LifecycleState] = None,
        reason: str = "",
    ) -> TransitionResult:
        """Execute a state transition."""
        async with self._lock:
            return await self._execute_transition(trigger, target, reason)

    async def set_state_external(
        self,
        target: LifecycleState,
        reason: str = "",
    ) -> TransitionResult:
        """Set the lifecycle state without executing tier load/unload actions.

        Used when an external system (e.g. the consciousness matrix) has already
        performed the actual tier transitions and just needs the FSM to reflect
        the new state. Records the transition in history for auditability.
        """
        async with self._lock:
            current = LifecycleState(self._snapshot.state)
            if current == target:
                return TransitionResult(
                    ok=True,
                    from_state=current.value,
                    to_state=target.value,
                    trigger="external_sync",
                )

            start = time.time()
            logger.info("LIFECYCLE: external state set %s → %s (reason=%s)",
                         current.value, target.value, reason)

            # Probe tiers to update snapshot with actual state
            for tier, endpoint in self._tier_endpoints.items():
                self._snapshot.tiers[tier] = await self._probe_tier(tier, endpoint)
            self._dedupe_aliased_tiers()

            # Record in history
            record = TransitionRecord(
                from_state=current.value,
                to_state=target.value,
                trigger="external_sync",
                reason=reason,
                elapsed_s=round(time.time() - start, 1),
                actions=["external_sync"],
            )

            self._snapshot.state = target
            self._snapshot.last_transition_at = datetime.now(timezone.utc)
            self._snapshot.last_transition_trigger = "external_sync"
            self._snapshot.transition_from = None
            self._snapshot.transition_to = None
            self._snapshot.transition_phase = None
            self._snapshot.transition_error = None
            self._snapshot.history.append(record)
            if len(self._snapshot.history) > MAX_HISTORY:
                self._snapshot.history = self._snapshot.history[-MAX_HISTORY:]

            # Update VRAM
            vram = sum(t.vram_mb for t in self._snapshot.tiers.values())
            self._snapshot.vram_used_mb = vram
            self._snapshot.vram_free_mb = self._snapshot.vram_total_mb - vram

            await self._persist()

            logger.info("LIFECYCLE: external state set complete: %s → %s", current.value, target.value)

            return TransitionResult(
                ok=True,
                from_state=current.value,
                to_state=target.value,
                trigger="external_sync",
                elapsed_s=round(time.time() - start, 1),
                actions=["external_sync"],
            )

    async def reconcile(self) -> dict:
        """Probe all tiers and reconcile actual state with expected.

        Called on startup and via POST /lifecycle/reconcile.
        yirf: serialized on the transition lock — reconcile's probe-infer-
        repair used to interleave with an executing transition and reload
        tiers the transition had just unloaded (one judge at a time).
        """
        async with self._lock:
            return await self._reconcile_locked()

    async def _reconcile_locked(self) -> dict:
        logger.info("LIFECYCLE: reconciling state...")
        probed = {}
        for tier, endpoint in self._tier_endpoints.items():
            status = await self._probe_tier(tier, endpoint)
            probed[tier] = status
            self._snapshot.tiers[tier] = status
        self._dedupe_aliased_tiers()
        # Refresh probed map post-dedupe so inference logic below sees zeroed aliases
        probed = dict(self._snapshot.tiers)

        # Preserve MEDITATION state — training owns GPU, tiers are intentionally
        # unloaded. Inferring from tier placement would wrongly yield DEEP_SLEEP.
        current_state = LifecycleState(self._snapshot.state) if isinstance(self._snapshot.state, str) else self._snapshot.state
        if current_state == LifecycleState.MEDITATION:
            logger.info("LIFECYCLE: reconcile skipped — MEDITATION active (study owns GPU)")
            await self._persist()
            return {
                "ok": True,
                "old_state": current_state.value,
                "new_state": current_state.value,
                "tiers": {k: v.model_dump() for k, v in probed.items()},
                "skipped": "meditation_active",
            }

        # Determine actual state from what's loaded
        core_gpu = probed.get("core", TierLiveStatus()).model_loaded and probed["core"].device == "gpu"
        nano_gpu = probed.get("nano", TierLiveStatus()).model_loaded and probed["nano"].device == "gpu"
        prime_gpu = probed.get("prime", TierLiveStatus()).model_loaded and probed["prime"].device == "gpu"

        # Infer state from actual tier placement
        if prime_gpu:
            inferred = LifecycleState.FOCUSING
        elif core_gpu and nano_gpu:
            inferred = LifecycleState.AWAKE
        elif core_gpu or nano_gpu:
            inferred = LifecycleState.AWAKE  # Partial — treat as awake
        else:
            # Nothing on GPU
            core_loaded = probed.get("core", TierLiveStatus()).model_loaded
            if core_loaded:
                inferred = LifecycleState.SLEEP  # CPU-only
            else:
                inferred = LifecycleState.DEEP_SLEEP

        old_state = self._snapshot.state
        self._snapshot.state = inferred
        self._snapshot.timestamp = datetime.now(timezone.utc)

        # Update VRAM
        vram = sum(t.vram_mb for t in self._snapshot.tiers.values())
        self._snapshot.vram_used_mb = vram
        self._snapshot.vram_free_mb = self._snapshot.vram_total_mb - vram

        logger.info("LIFECYCLE: reconciled %s → %s (core_gpu=%s, nano_gpu=%s, prime_gpu=%s)",
                     old_state, inferred, core_gpu, nano_gpu, prime_gpu)

        # Auto-fix: reload missing required tiers for the current state
        expected = TIER_EXPECTATIONS.get(inferred, {})
        repaired = []
        for tier, exp in expected.items():
            if not exp.required:
                continue
            tier_status = probed.get(tier, TierLiveStatus())
            if exp.device == "gpu" and not (tier_status.model_loaded and tier_status.device == "gpu"):
                logger.warning("LIFECYCLE: auto-repairing %s (expected gpu, got %s)", tier, tier_status.device)
                if await self._load_tier(tier, "cuda"):
                    repaired.append(f"{tier}:reload_gpu")
            elif exp.device == "cpu" and not tier_status.model_loaded:
                logger.warning("LIFECYCLE: auto-repairing %s (expected cpu, got unloaded)", tier)
                if await self._load_tier(tier, "cpu"):
                    repaired.append(f"{tier}:reload_cpu")
        if repaired:
            logger.info("LIFECYCLE: auto-repaired tiers: %s", repaired)
            try:
                from gaia_common.event_buffer import log_event
                log_event("lifecycle", f"Auto-repaired: {', '.join(repaired)}", source="reconcile")
            except Exception:
                pass

        await self._persist()

        return {
            "ok": True,
            "old_state": old_state if isinstance(old_state, str) else old_state.value,
            "new_state": inferred.value,
            "tiers": {k: v.model_dump() for k, v in probed.items()},
        }

    # ── Transition Execution ──────────────────────────────────────────────

    # Map lifecycle states to ConsciousnessMatrix configuration names.
    # Used by _execute_transition() to delegate tier actions to the CM.
    _STATE_TO_CM_CONFIG = {
        LifecycleState.AWAKE: "awake",
        LifecycleState.LISTENING: "listening",   # voice gear (heo): Core-GGUF-GPU so STT+TTS fit
        LifecycleState.FOCUSING: "focusing",
        LifecycleState.MEDITATION: "meditation",
        LifecycleState.SLEEP: "sleep",
        LifecycleState.DEEP_SLEEP: "deep_sleep",
        # r2kn: PARKED was missing, so idle-timeout parks took the legacy
        # direct-unload path and never updated the CM tier targets. The 15s
        # CM poll then saw target=CONSCIOUS/actual=UNCONSCIOUS and reloaded
        # the model it had just unloaded — a permanent load/unload fight.
        LifecycleState.PARKED: "parked",
    }

    async def _execute_transition(
        self,
        trigger: TransitionTrigger,
        target: Optional[LifecycleState],
        reason: str,
    ) -> TransitionResult:
        """Execute a validated transition. Must be called with lock held."""
        current = LifecycleState(self._snapshot.state)

        # Resolve target
        resolved_target = validate_transition(current, trigger, target)
        if resolved_target is None:
            valid = available_transitions(current)
            return TransitionResult(
                ok=False,
                from_state=current.value,
                trigger=trigger.value,
                error=f"Invalid transition: {current.value} + {trigger.value}"
                      + (f" → {target.value}" if target else "")
                      + f". Valid: {valid}",
            )

        logger.info("LIFECYCLE: %s → %s (trigger=%s, reason=%s)",
                     current.value, resolved_target.value, trigger.value, reason)

        # yirf: VRAM preflight BEFORE touching state — a gear that can't fit
        # refuses cleanly (state untouched) instead of grinding into an OOM
        # jam mid-TRANSITIONING like 2026-07-16's parked→awake.
        preflight_error = await self._vram_preflight(resolved_target)
        if preflight_error:
            logger.warning("LIFECYCLE: transition refused — %s", preflight_error)
            return TransitionResult(
                ok=False,
                from_state=current.value,
                trigger=trigger.value,
                error=preflight_error,
            )

        start = time.time()
        actions_taken = []

        # Enter TRANSITIONING
        self._snapshot.state = LifecycleState.TRANSITIONING
        self._snapshot.transition_from = current
        self._snapshot.transition_to = resolved_target
        self._snapshot.transition_phase = "starting"
        self._snapshot.transition_error = None
        await self._persist()

        try:
            # ── Clutch Protocol: delegate tier actions to ConsciousnessMatrix ──
            cm_config = self._STATE_TO_CM_CONFIG.get(resolved_target)
            if self._consciousness_matrix is not None and cm_config is not None:
                # CM is wired — it is the sole authority for tier load/unload.
                # Call apply_for_lifecycle() which skips the lifecycle sync
                # callback (we manage FSM state here, avoiding deadlock).
                self._snapshot.transition_phase = f"cm_applying_{cm_config}"
                await self._persist()

                cm_result = await self._consciousness_matrix.apply_for_lifecycle(cm_config)
                actions_taken.append(f"cm:{cm_config}")

                # Check for failures in CM results
                tier_results = cm_result.get("results", {})
                for tier, tier_result in tier_results.items():
                    if isinstance(tier_result, dict):
                        if tier_result.get("ok"):
                            action = tier_result.get("action", "ok")
                            actions_taken.append(f"{tier}:{action}")
                        elif tier_result.get("error"):
                            logger.warning("LIFECYCLE: CM tier %s error: %s",
                                           tier, tier_result["error"])

                logger.info("LIFECYCLE: CM applied config '%s': %s", cm_config, cm_result.get("configuration"))

            else:
                # ── Legacy path: direct httpx calls (no CM wired) ──────────
                # Compute tier diff: what needs to change
                current_expectations = TIER_EXPECTATIONS.get(current, {})
                target_expectations = TIER_EXPECTATIONS.get(resolved_target, {})

                # Phase 1: Unload tiers that need to vacate GPU
                for tier, target_exp in target_expectations.items():
                    current_exp = current_expectations.get(tier)
                    if current_exp and current_exp.device == "gpu" and target_exp.device != "gpu":
                        self._snapshot.transition_phase = f"unloading_{tier}"
                        await self._persist()
                        result = await self._unload_tier(tier)
                        if result:
                            actions_taken.append(f"{tier}:unload_gpu")
                            logger.info("LIFECYCLE: unloaded %s from GPU", tier)

                # Phase 2: Wait for CUDA cleanup
                if actions_taken:
                    self._snapshot.transition_phase = "cuda_cleanup"
                    await self._persist()
                    await asyncio.sleep(1.5)

                # Phase 3: Load tiers that need to activate on GPU
                for tier, target_exp in target_expectations.items():
                    if target_exp.device == "gpu" and target_exp.required:
                        current_exp = current_expectations.get(tier)
                        if not current_exp or current_exp.device != "gpu":
                            self._snapshot.transition_phase = f"loading_{tier}"
                            await self._persist()
                            result = await self._load_tier(tier, "cuda")
                            if result:
                                actions_taken.append(f"{tier}:load_gpu")
                                logger.info("LIFECYCLE: loaded %s on GPU", tier)
                            elif target_exp.required:
                                raise RuntimeError(f"Required tier {tier} failed to load on GPU")

                # Phase 3b: Load tiers that need CPU
                for tier, target_exp in target_expectations.items():
                    if target_exp.device == "cpu" and target_exp.required:
                        current_exp = current_expectations.get(tier)
                        if not current_exp or current_exp.device == "unloaded":
                            self._snapshot.transition_phase = f"loading_{tier}_cpu"
                            await self._persist()
                            result = await self._load_tier(tier, "cpu")
                            if result:
                                actions_taken.append(f"{tier}:load_cpu")

            # Phase 4: KV pre-warm on AWAKE entry (applies to both CM and legacy paths)
            if resolved_target == LifecycleState.AWAKE:
                self._snapshot.transition_phase = "kv_prewarm"
                await self._persist()
                for tier in ("core", "nano"):
                    if any(a.startswith(f"{tier}:load") for a in actions_taken):
                        await self._prewarm_kv(tier)
                        actions_taken.append(f"{tier}:kv_prewarm")

            # Phase 5: Verify actual state
            self._snapshot.transition_phase = "verifying"
            await self._persist()
            for tier, endpoint in self._tier_endpoints.items():
                self._snapshot.tiers[tier] = await self._probe_tier(tier, endpoint)
            self._dedupe_aliased_tiers()

            # Update audio flags based on target state
            if resolved_target == LifecycleState.LISTENING:
                self._snapshot.audio_stt = True
            elif resolved_target == LifecycleState.AWAKE:
                self._snapshot.audio_stt = False
                self._snapshot.audio_tts = False

            # Success — enter target state
            elapsed = time.time() - start
            record = TransitionRecord(
                from_state=current.value,
                to_state=resolved_target.value,
                trigger=trigger.value,
                reason=reason,
                target=target.value if target else None,
                elapsed_s=round(elapsed, 1),
                actions=actions_taken,
            )

            self._snapshot.state = resolved_target
            self._snapshot.transition_from = None
            self._snapshot.transition_to = None
            self._snapshot.transition_phase = None
            self._snapshot.transition_error = None
            self._snapshot.last_transition_at = datetime.now(timezone.utc)
            self._snapshot.last_transition_trigger = trigger.value
            self._snapshot.history.append(record)
            if len(self._snapshot.history) > MAX_HISTORY:
                self._snapshot.history = self._snapshot.history[-MAX_HISTORY:]

            # Update VRAM
            vram = sum(t.vram_mb for t in self._snapshot.tiers.values())
            self._snapshot.vram_used_mb = vram
            self._snapshot.vram_free_mb = self._snapshot.vram_total_mb - vram

            await self._persist()

            # yirf: downshift restores the negotiated-away VRAM tenant.
            # MEDITATION excluded — training wants the whole card.
            if resolved_target in (LifecycleState.PARKED, LifecycleState.SLEEP,
                                   LifecycleState.DEEP_SLEEP):
                await self._restore_vram_tenant(f"downshift to {resolved_target.value}")

            logger.info("LIFECYCLE: %s → %s in %.1fs (%s)",
                         current.value, resolved_target.value, elapsed, actions_taken)

            # Log to event buffer for episodic memory
            try:
                from gaia_common.event_buffer import log_event
                log_event("lifecycle",
                          f"{current.value} → {resolved_target.value} ({reason or trigger.value}, {round(elapsed, 1)}s)",
                          source="lifecycle_machine",
                          details={"trigger": trigger.value, "actions": actions_taken})
            except Exception:
                pass

            return TransitionResult(
                ok=True,
                from_state=current.value,
                to_state=resolved_target.value,
                trigger=trigger.value,
                elapsed_s=round(elapsed, 1),
                actions=actions_taken,
            )

        except Exception as e:
            elapsed = time.time() - start
            logger.exception("LIFECYCLE: transition failed — attempting rollback to %s", current.value)

            # Attempt rollback
            self._snapshot.transition_error = str(e)
            self._snapshot.transition_phase = "rollback"
            await self._persist()

            try:
                await self._rollback_to(current)
                self._snapshot.state = current
            except Exception:
                logger.exception("LIFECYCLE: rollback also failed — stuck in TRANSITIONING")
                # Stay in TRANSITIONING with error — requires manual reconcile

            self._snapshot.transition_from = None
            self._snapshot.transition_to = None
            self._snapshot.transition_phase = None

            record = TransitionRecord(
                from_state=current.value,
                to_state=resolved_target.value,
                trigger=trigger.value,
                reason=reason,
                elapsed_s=round(elapsed, 1),
                actions=actions_taken,
                error=str(e),
            )
            self._snapshot.history.append(record)
            await self._persist()

            return TransitionResult(
                ok=False,
                from_state=current.value,
                to_state=resolved_target.value,
                trigger=trigger.value,
                elapsed_s=round(elapsed, 1),
                actions=actions_taken,
                error=str(e),
            )

    # ── Tier Operations ───────────────────────────────────────────────────

    def _dedupe_aliased_tiers(self) -> None:
        """Zero VRAM on tiers that are forwarders to another tier's engine.

        Post-Sovereign-Duality, gaia-nano is a socat TCP forwarder to
        gaia-core:8092 — probing it returns Core's exact health response,
        which would double-count the same VRAM. Dict insertion order in
        ``_tier_endpoints`` determines the primary (first wins).
        """
        seen: Dict[tuple, str] = {}
        for tier_name in self._tier_endpoints.keys():
            status = self._snapshot.tiers.get(tier_name)
            if status is None or not status.model_loaded or not status.model_path:
                continue
            key = (status.model_path, status.vram_mb)
            if key in seen:
                primary = seen[key]
                logger.info(
                    "LIFECYCLE: %s endpoint aliased to %s (same model=%s, vram=%dMB) — zeroing duplicate",
                    tier_name, primary, status.model_path, status.vram_mb,
                )
                self._snapshot.tiers[tier_name] = TierLiveStatus(
                    device="unloaded",
                    model_loaded=False,
                    model_path="",
                    vram_mb=0,
                    managed=status.managed,
                    healthy=status.healthy,
                    endpoint=status.endpoint,
                )
            else:
                seen[key] = tier_name

    async def _probe_tier(self, tier: str, endpoint: str) -> TierLiveStatus:
        """Probe a tier's health endpoint for live status."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{endpoint}/health")
                if resp.status_code == 200:
                    data = resp.json()
                    model_loaded = data.get("model_loaded", False)
                    managed = data.get("managed", False)

                    # Get VRAM and model info — try /status first (proxied to
                    # worker, has real data), fall back to /model/info
                    vram_mb = 0
                    model_path = ""
                    try:
                        status_resp = await client.get(f"{endpoint}/status")
                        if status_resp.status_code == 200:
                            status = status_resp.json()
                            vram_mb = status.get("vram_mb", 0)
                            model_path = status.get("model", "")
                    except Exception:
                        pass
                    if vram_mb == 0:
                        try:
                            info_resp = await client.get(f"{endpoint}/model/info")
                            if info_resp.status_code == 200:
                                info = info_resp.json()
                                vram_mb = info.get("vram_mb", 0)
                                model_path = model_path or info.get("model_path", "")
                        except Exception:
                            pass

                    # Trust the engine's device field when available;
                    # fall back to vram heuristic for legacy endpoints
                    reported_device = ""
                    try:
                        reported_device = status.get("device", "")
                    except Exception:
                        pass
                    if reported_device in ("gpu", "cuda"):
                        device = "gpu"
                    elif reported_device == "cpu":
                        device = "cpu" if model_loaded else "unloaded"
                    else:
                        device = "gpu" if model_loaded and vram_mb > 100 else (
                            "cpu" if model_loaded else "unloaded")

                    return TierLiveStatus(
                        device=device,
                        model_loaded=model_loaded,
                        model_path=model_path,
                        vram_mb=vram_mb,
                        managed=managed,
                        healthy=True,
                        endpoint=endpoint,
                    )
        except Exception:
            pass

        return TierLiveStatus(endpoint=endpoint)

    async def _load_tier(self, tier: str, device: str) -> bool:
        """Load a tier's model via its managed engine."""
        endpoint = self._tier_endpoints.get(tier)
        model_path = self._tier_models.get(tier)
        if not endpoint or not model_path:
            logger.warning("LIFECYCLE: no endpoint/model for tier %s", tier)
            return False

        try:
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(
                    f"{endpoint}/model/load",
                    json={"model": model_path, "device": device},
                )
                if resp.status_code in (200, 409):
                    result = resp.json()
                    return result.get("ok", False) or result.get("model_loaded", False) or resp.status_code == 409
                logger.warning("LIFECYCLE: %s load returned HTTP %d", tier, resp.status_code)
        except Exception as e:
            logger.warning("LIFECYCLE: %s load failed: %s", tier, e)

        return False

    async def _unload_tier(self, tier: str) -> bool:
        """Unload a tier's model via its managed engine.

        r2kn: drains in-flight inference first. /model/unload kills the
        worker unconditionally; without the drain an idle-timeout park cuts
        a user's generation mid-stream. If drain times out (inference still
        active), the unload is aborted and the caller's transition fails —
        it retries on a later cycle.
        """
        endpoint = self._tier_endpoints.get(tier)
        if not endpoint:
            return False

        try:
            async with httpx.AsyncClient(timeout=45) as client:
                try:
                    health = await client.get(f"{endpoint}/health", timeout=5.0)
                    if (health.status_code == 200
                            and health.json().get("active_inference", 0) > 0):
                        drain = await client.post(
                            f"{endpoint}/inference/drain", json={"timeout_s": 30.0},
                        )
                        if not (drain.status_code == 200 and drain.json().get("ok")):
                            await client.post(f"{endpoint}/inference/resume")
                            logger.warning(
                                "LIFECYCLE: %s unload ABORTED — inference active, drain timed out",
                                tier,
                            )
                            return False
                except httpx.HTTPError:
                    pass  # health/drain unsupported or unreachable — proceed
                resp = await client.post(f"{endpoint}/model/unload")
                if resp.status_code == 200:
                    # Clear stale drain flag — the manager doesn't reset it on
                    # unload/load and it 503s inference after the next load.
                    try:
                        await client.post(f"{endpoint}/inference/resume")
                    except httpx.HTTPError:
                        pass
                    return True
                # Fallback for llama-server (no /model/unload)
                if resp.status_code == 404:
                    logger.info("LIFECYCLE: %s has no /model/unload (llama-server?)", tier)
                    return False
        except Exception as e:
            logger.warning("LIFECYCLE: %s unload failed: %s", tier, e)

        return False

    async def _prewarm_kv(self, tier: str):
        """Pre-warm a tier's KV prefix cache with identity."""
        endpoint = self._tier_endpoints.get(tier)
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
                    logger.info("LIFECYCLE: %s KV pre-warmed", tier)
        except Exception as e:
            logger.debug("LIFECYCLE: %s KV pre-warm failed: %s", tier, e)

    async def _rollback_to(self, state: LifecycleState):
        """Attempt to restore tier state for a given lifecycle state."""
        expectations = TIER_EXPECTATIONS.get(state, {})
        for tier, exp in expectations.items():
            if exp.required and exp.device == "gpu":
                await self._load_tier(tier, "cuda")
            elif exp.required and exp.device == "cpu":
                await self._load_tier(tier, "cpu")

    # ── Persistence ───────────────────────────────────────────────────────

    async def _persist(self):
        """Persist current state to disk via state manager."""
        if self._state_manager:
            try:
                async with self._state_manager.modify() as state:
                    state.lifecycle = self._snapshot.model_dump()
            except Exception:
                logger.debug("Lifecycle persist failed", exc_info=True)

    async def load_persisted_state(self):
        """Load state from disk on startup."""
        if self._state_manager:
            try:
                persisted = getattr(self._state_manager.state, "lifecycle", None)
                if persisted and isinstance(persisted, dict):
                    self._snapshot = LifecycleSnapshot(**persisted)
                    logger.info("LIFECYCLE: loaded persisted state: %s", self._snapshot.state)
                    return
            except Exception:
                logger.debug("No persisted lifecycle state", exc_info=True)

        # No persisted state — reconcile from actual tier status
        logger.info("LIFECYCLE: no persisted state — will reconcile on startup")

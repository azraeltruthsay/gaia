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
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx

from gaia_common.lifecycle.states import (
    LifecycleState,
    TransitionTrigger,
    TierExpectation,
    TIER_EXPECTATIONS,
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

    def __init__(self, state_manager=None):
        self._state_manager = state_manager
        self._lock = asyncio.Lock()
        self._snapshot = LifecycleSnapshot()

        # Tier engine endpoints (Docker network)
        self._tier_endpoints = {
            "core": os.environ.get("CORE_INFERENCE_ENDPOINT", "http://gaia-core:8092"),
            "nano": os.environ.get("NANO_INFERENCE_ENDPOINT", "http://gaia-nano:8080"),
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

    # ── Public API ────────────────────────────────────────────────────────

    async def get_snapshot(self) -> LifecycleSnapshot:
        """Return current lifecycle snapshot with live tier status."""
        # Enrich with live tier probes
        for tier, endpoint in self._tier_endpoints.items():
            self._snapshot.tiers[tier] = await self._probe_tier(tier, endpoint)

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
        """
        logger.info("LIFECYCLE: reconciling state...")
        probed = {}
        for tier, endpoint in self._tier_endpoints.items():
            status = await self._probe_tier(tier, endpoint)
            probed[tier] = status
            self._snapshot.tiers[tier] = status

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

            # Phase 4: KV pre-warm on AWAKE entry
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
        """Unload a tier's model via its managed engine."""
        endpoint = self._tier_endpoints.get(tier)
        if not endpoint:
            return False

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(f"{endpoint}/model/unload")
                if resp.status_code == 200:
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

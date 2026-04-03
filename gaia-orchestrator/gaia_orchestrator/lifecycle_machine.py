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

from gaia_common.utils.maintenance import is_maintenance_active
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

        # Consciousness matrix reference — set after both are initialized
        # to break circular dependency. The lifecycle machine DECLARES intent;
        # the consciousness matrix EXECUTES tier loads/unloads.
        self._consciousness_matrix = None

    def set_consciousness_matrix(self, cm):
        """Wire the consciousness matrix after both objects are initialized."""
        self._consciousness_matrix = cm
        logger.info("LIFECYCLE: consciousness matrix linked (clutch architecture active)")

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
        Observe-only: probes tiers and infers state, but does NOT load/unload.
        Auto-repair is the consciousness matrix's responsibility.
        """
        logger.info("LIFECYCLE: reconciling state (observe-only)...")
        probed = {}
        for tier, endpoint in self._tier_endpoints.items():
            status = await self._probe_tier(tier, endpoint)
            probed[tier] = status
            self._snapshot.tiers[tier] = status

        inferred = self._infer_state(probed)
        old_state = self._snapshot.state

        # If stuck in TRANSITIONING, force out to inferred state
        if old_state == LifecycleState.TRANSITIONING:
            logger.warning("LIFECYCLE: was stuck in TRANSITIONING — forcing to %s", inferred.value)
            self._snapshot.transition_from = None
            self._snapshot.transition_to = None
            self._snapshot.transition_phase = None
            self._snapshot.transition_error = None

        self._snapshot.state = inferred
        self._snapshot.timestamp = datetime.now(timezone.utc)

        # Update VRAM
        vram = sum(t.vram_mb for t in self._snapshot.tiers.values())
        self._snapshot.vram_used_mb = vram
        self._snapshot.vram_free_mb = self._snapshot.vram_total_mb - vram

        logger.info("LIFECYCLE: reconciled %s → %s", old_state, inferred.value)
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
        """Execute a validated transition. Must be called with lock held.

        The lifecycle machine DECLARES intent and delegates execution to
        the consciousness matrix (the "gearbox"). It does not send any
        HTTP requests to tier engines directly.
        """
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
        self._snapshot.transition_phase = "pending_execution"
        self._snapshot.transition_error = None
        await self._persist()

        try:
            # Delegate to consciousness matrix for actual tier load/unload
            if self._consciousness_matrix:
                self._snapshot.transition_phase = "executing"
                await self._persist()

                result = await asyncio.wait_for(
                    self._consciousness_matrix.execute_shift(
                        from_state=current,
                        to_state=resolved_target,
                    ),
                    timeout=120.0,
                )
                if not result.get("ok"):
                    raise RuntimeError(result.get("error", "execution failed"))
                actions_taken = result.get("actions", [])
            else:
                logger.warning("LIFECYCLE: no consciousness matrix — cannot execute tier actions")
                actions_taken = ["no_executor"]

            # KV pre-warm on AWAKE entry
            if resolved_target == LifecycleState.AWAKE:
                self._snapshot.transition_phase = "kv_prewarm"
                await self._persist()
                for tier in ("core", "nano"):
                    if any(f"{tier}:" in a and "CONSCIOUS" in a for a in actions_taken):
                        await self._prewarm_kv(tier)
                        actions_taken.append(f"{tier}:kv_prewarm")

            # Verify actual state
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

        except asyncio.TimeoutError:
            elapsed = time.time() - start
            logger.error("LIFECYCLE: transition timed out after 120s — inferring actual state")
            await self._infer_state_from_probes()

            record = TransitionRecord(
                from_state=current.value,
                to_state=resolved_target.value,
                trigger=trigger.value,
                reason=reason,
                elapsed_s=round(elapsed, 1),
                actions=actions_taken,
                error="execution timeout (120s)",
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
                error="execution timeout (120s)",
            )

        except Exception as e:
            elapsed = time.time() - start
            logger.exception("LIFECYCLE: transition failed — inferring actual state")

            # Instead of rollback (which can also fail and leave us stuck),
            # probe what's actually loaded and set state accordingly
            await self._infer_state_from_probes()

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

    # ── State Inference ─────────────────────────────────────────────────

    def _infer_state(self, probed: dict) -> LifecycleState:
        """Infer lifecycle state from actual tier placement."""
        core_gpu = probed.get("core", TierLiveStatus()).model_loaded and probed["core"].device == "gpu"
        nano_gpu = probed.get("nano", TierLiveStatus()).model_loaded and probed["nano"].device == "gpu"
        prime_gpu = probed.get("prime", TierLiveStatus()).model_loaded and probed["prime"].device == "gpu"

        if prime_gpu:
            return LifecycleState.FOCUSING
        elif core_gpu and nano_gpu:
            return LifecycleState.AWAKE
        elif core_gpu or nano_gpu:
            return LifecycleState.AWAKE  # Partial — treat as awake
        else:
            core_loaded = probed.get("core", TierLiveStatus()).model_loaded
            if core_loaded:
                return LifecycleState.SLEEP  # CPU-only
            else:
                return LifecycleState.DEEP_SLEEP

    async def _infer_state_from_probes(self):
        """Probe all tiers and set state to best-matching lifecycle state.

        Used after failed transitions instead of rollback — we accept reality
        rather than trying to force a state that may fail again.
        """
        probed = {}
        for tier, endpoint in self._tier_endpoints.items():
            status = await self._probe_tier(tier, endpoint)
            probed[tier] = status
            self._snapshot.tiers[tier] = status

        inferred = self._infer_state(probed)
        logger.info("LIFECYCLE: inferred state from probes: %s", inferred.value)

        self._snapshot.state = inferred
        self._snapshot.transition_from = None
        self._snapshot.transition_to = None
        self._snapshot.transition_phase = None
        self._snapshot.timestamp = datetime.now(timezone.utc)

        vram = sum(t.vram_mb for t in self._snapshot.tiers.values())
        self._snapshot.vram_used_mb = vram
        self._snapshot.vram_free_mb = self._snapshot.vram_total_mb - vram

        await self._persist()

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

"""
GAIA Sleep/Wake State Machine.

Manages six public states + two internal transient phases:

Public states:
    OFFLINE → ACTIVE → DROWSY → ASLEEP → DREAMING / DISTRACTED

Internal phases (not in the public enum):
    _FINISHING_TASK, _WAKING

Design decisions:
- DROWSY is cancellable: a message arriving during checkpoint writing
  aborts the transition and returns to ACTIVE.
- Waking uses a parallel strategy: CPU Lite handles the first queued
  message while Prime boots in the background (~37-60 s from tmpfs).
- prime.md checkpoint is the KV cache replacement.
- DREAMING = GPU handed off to Study (training); canned response only.
- DISTRACTED = CPU or GPU under sustained load (>42% for 5 s); canned response.

This module lives in gaia-core (not gaia-common) to avoid a circular
dependency.  gaia-common provides low-level primitives (IdleMonitor,
TaskQueue); gaia-core owns the orchestration.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from gaia_core.cognition.prime_checkpoint import PrimeCheckpointManager

logger = logging.getLogger("GAIA.SleepWake")


class GaiaState(Enum):
    OFFLINE = "offline"
    ACTIVE = "active"
    DROWSY = "drowsy"  # Checkpoint in progress — cancellable
    ASLEEP = "asleep"  # GPU offloaded, running sleep tasks
    DREAMING = "dreaming"  # GPU handed to Study for training
    DISTRACTED = "distracted"  # System under sustained load


class _TransientPhase(Enum):
    """Internal waking sub-phases — not visible in the public state enum."""
    NONE = "none"
    FINISHING_TASK = "finishing_task"
    WAKING = "waking"


# Canned responses for states that don't forward to the model
CANNED_DREAMING = (
    "I'm studying right now and can't chat — "
    "I'll be back once my training session wraps up!"
)
CANNED_DISTRACTED = (
    "I'm a little occupied at the moment — "
    "give me a few minutes and I'll get back to you!"
)


class SleepWakeManager:
    """Manages GAIA's sleep/wake state transitions with cognitive continuity."""

    def __init__(self, config, model_pool=None, idle_monitor=None, timeline_store=None) -> None:
        self.config = config
        self.state = GaiaState.ACTIVE
        self._phase = _TransientPhase.NONE
        self.current_task: Optional[Dict[str, Any]] = None
        self.wake_signal_pending = False
        self._timeline = timeline_store
        self.prime_available = False
        self.model_pool = model_pool
        self.idle_monitor = idle_monitor
        self.checkpoint_manager = PrimeCheckpointManager(config, timeline_store=timeline_store)
        self.last_state_change = datetime.now(timezone.utc)
        self.dreaming_handoff_id: Optional[str] = None
        self.voice_active: bool = False

        logger.info("SleepWakeManager initialized")

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_state(self) -> GaiaState:
        return self.state

    def should_transition_to_drowsy(self, idle_minutes: float) -> bool:
        """Check whether we should begin the sleep transition."""
        if self.state != GaiaState.ACTIVE:
            return False
        threshold = getattr(self.config, "SLEEP_IDLE_THRESHOLD_MINUTES", 5)
        return idle_minutes >= threshold

    # ------------------------------------------------------------------
    # Timeline event helper
    # ------------------------------------------------------------------

    def _emit_state_change(self, from_state: str, to_state: str, reason: str = "") -> None:
        """Emit a state_change event to the timeline store (best-effort)."""
        if self._timeline is not None:
            try:
                self._timeline.append("state_change", {
                    "from": from_state,
                    "to": to_state,
                    "reason": reason,
                })
            except Exception:
                logger.debug("Timeline state_change emit failed", exc_info=True)

        # Notify gaia-audio of mute/unmute based on state transitions
        self._notify_audio_state(to_state)

    def _notify_audio_state(self, to_state: str) -> None:
        """Notify gaia-audio to sleep/wake based on sleep state (best-effort).

        Uses /sleep (GPU model unload) and /wake (eager GPU reload) for
        ASLEEP/DREAMING/ACTIVE transitions.  Falls back silently on failure.
        """
        try:
            constants = getattr(self.config, "constants", {})
            audio_cfg = constants.get("INTEGRATIONS", {}).get("audio", {})
            if not audio_cfg.get("enabled") or not audio_cfg.get("mute_on_sleep"):
                return

            audio_endpoint = audio_cfg.get("endpoint", "http://gaia-audio:8080")

            if to_state in ("asleep", "dreaming"):
                if self.voice_active:
                    logger.info("Skipping audio sleep — voice channel active")
                    return
                action = "sleep"
                timeout = 15.0  # GPU model unload may take a few seconds
            elif to_state == "active":
                action = "wake"
                timeout = 5.0   # /wake returns immediately (reload is background)
            else:
                return

            import httpx
            with httpx.Client(timeout=timeout) as client:
                client.post(f"{audio_endpoint}/{action}")
            logger.debug("Audio %s signal sent", action)
        except Exception:
            logger.debug("Audio state notification failed (non-fatal)", exc_info=True)

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def initiate_drowsy(self, current_packet=None) -> bool:
        """Transition ACTIVE → DROWSY → ASLEEP.

        During DROWSY, Prime writes its cognitive checkpoint.
        If a wake signal arrives before the checkpoint completes,
        the sleep is cancelled and we return to ACTIVE.

        Returns True if we entered ASLEEP, False otherwise.
        """
        if self.state != GaiaState.ACTIVE:
            logger.warning("Cannot enter DROWSY from state: %s", self.state)
            return False

        self.state = GaiaState.DROWSY
        self.last_state_change = datetime.now(timezone.utc)
        self._emit_state_change("active", "drowsy", "idle threshold reached")
        logger.info("Entering DROWSY — writing checkpoint...")

        # Early bail-out: if a wake signal already arrived, skip checkpoint entirely
        if self.wake_signal_pending:
            logger.info("Wake signal already pending — skipping checkpoint, returning to ACTIVE")
            self.state = GaiaState.ACTIVE
            self.wake_signal_pending = False
            self.last_state_change = datetime.now(timezone.utc)
            return False

        try:
            # Rotate FIRST so the backup captures the *previous* checkpoint
            self.checkpoint_manager.rotate_checkpoints()
            self.checkpoint_manager.create_checkpoint(current_packet, model_pool=self.model_pool)

            # Check if we were interrupted during checkpoint write
            if self.wake_signal_pending:
                logger.info("Message arrived during DROWSY — cancelling sleep")
                # Mark the freshly-written checkpoint as consumed to prevent stale injection
                self.checkpoint_manager.mark_consumed()
                self.state = GaiaState.ACTIVE
                self.wake_signal_pending = False
                self.last_state_change = datetime.now(timezone.utc)
                return False

            # Checkpoint complete — enter ASLEEP
            self.state = GaiaState.ASLEEP
            self.last_state_change = datetime.now(timezone.utc)
            self._emit_state_change("drowsy", "asleep", "checkpoint complete")
            logger.info("Checkpoint written — entering ASLEEP")
            return True

        except Exception:
            logger.error("Checkpoint failed — staying ACTIVE", exc_info=True)
            self.state = GaiaState.ACTIVE
            self.last_state_change = datetime.now(timezone.utc)
            return False

    def receive_wake_signal(self) -> None:
        """Called by gaia-web (via POST /sleep/wake) when a message is queued."""
        self.wake_signal_pending = True

        # Only reset idle timer when waking from a sleep state (DROWSY/ASLEEP).
        # When already ACTIVE, /process_packet handles mark_active() itself —
        # resetting here would prevent the idle countdown from ever completing.
        if self.idle_monitor is not None and self.state in (
            GaiaState.DROWSY, GaiaState.ASLEEP,
        ):
            self.idle_monitor.mark_active()

        if self.state == GaiaState.DROWSY:
            # initiate_drowsy() will notice the flag and cancel
            logger.info("Wake signal during DROWSY — will cancel checkpoint")

        elif self.state == GaiaState.ASLEEP:
            logger.info("Wake signal during ASLEEP")
            if self._phase == _TransientPhase.FINISHING_TASK:
                logger.info("Already FINISHING_TASK — wake deferred")
            elif self.current_task and not self.current_task.get("interruptible", True):
                task_id = self.current_task.get("task_id", "unknown")
                logger.info("Non-interruptible task running: %s", task_id)
                self._phase = _TransientPhase.FINISHING_TASK
            else:
                self.transition_to_waking()

        elif self.state == GaiaState.DREAMING:
            logger.info("Wake signal during DREAMING — deferred until study completes")
            # Don't transition — let exit_dreaming() handle it

        elif self.state == GaiaState.DISTRACTED:
            logger.info("Wake signal during DISTRACTED — noted, will check on recheck")

        elif self.state == GaiaState.ACTIVE:
            logger.debug("Wake signal received but already ACTIVE")
            self.wake_signal_pending = False

    def set_voice_active(self, active: bool) -> None:
        """Called by gaia-web when GAIA joins/leaves a Discord voice channel.

        Joining voice triggers an implicit wake signal so Prime begins booting.
        Leaving voice while sleeping triggers the deferred audio sleep signal.
        """
        prev = self.voice_active
        self.voice_active = active
        logger.info("Voice active: %s → %s (state: %s)", prev, active, self.state)

        if active:
            # Voice join = implicit wake signal
            self.receive_wake_signal()
        elif not active and self.state in (GaiaState.ASLEEP, GaiaState.DROWSY):
            # Left voice while sleeping — now safe to mute audio
            self._notify_audio_state(self.state.value)

    def transition_to_waking(self) -> None:
        """Move to internal WAKING phase. Begins parallel wake strategy."""
        if self.state != GaiaState.ASLEEP:
            logger.warning("Cannot wake from state: %s", self.state)
            return
        self._phase = _TransientPhase.WAKING
        self.last_state_change = datetime.now(timezone.utc)
        logger.info("Entering WAKING phase — starting parallel wake")

    def complete_wake(self) -> Dict[str, Any]:
        """Finish waking: load checkpoint, format as REVIEW context, go ACTIVE.

        Returns a dict with ``checkpoint_loaded``, ``context``, and
        ``timestamp`` keys.
        """
        if self.state != GaiaState.ASLEEP or self._phase != _TransientPhase.WAKING:
            logger.warning("Cannot complete wake from state: %s (phase: %s)", self.state, self._phase)
            return {"checkpoint_loaded": False}

        try:
            checkpoint = self.checkpoint_manager.load_latest()
            review_context = self._format_checkpoint_as_review(checkpoint)

            # Mark consumed so prompt_builder doesn't inject stale context
            if checkpoint:
                self.checkpoint_manager.mark_consumed()

            self.state = GaiaState.ACTIVE
            self._phase = _TransientPhase.NONE
            self.wake_signal_pending = False
            self.prime_available = True
            self.last_state_change = datetime.now(timezone.utc)
            self._emit_state_change("asleep", "active", "wake signal processed")

            logger.info("Wake complete, context restored")
            return {
                "checkpoint_loaded": bool(checkpoint),
                "context": review_context,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception:
            logger.error("Wake completion failed", exc_info=True)
            self.state = GaiaState.ACTIVE
            self._phase = _TransientPhase.NONE
            self.wake_signal_pending = False
            return {"checkpoint_loaded": False}

    # ------------------------------------------------------------------
    # DREAMING transitions (orchestrator-driven)
    # ------------------------------------------------------------------

    def enter_dreaming(self, handoff_id: str) -> bool:
        """ASLEEP → DREAMING when orchestrator hands GPU to Study."""
        if self.state != GaiaState.ASLEEP:
            logger.warning("Cannot enter DREAMING from state: %s", self.state)
            return False
        self.state = GaiaState.DREAMING
        self.dreaming_handoff_id = handoff_id
        self.last_state_change = datetime.now(timezone.utc)
        self._emit_state_change("asleep", "dreaming", f"handoff {handoff_id}")
        logger.info("Entering DREAMING (handoff %s)", handoff_id)
        return True

    def exit_dreaming(self) -> bool:
        """DREAMING → ASLEEP when Study returns GPU to Prime."""
        if self.state != GaiaState.DREAMING:
            logger.warning("Cannot exit DREAMING from state: %s", self.state)
            return False
        self.state = GaiaState.ASLEEP
        self.dreaming_handoff_id = None
        self.last_state_change = datetime.now(timezone.utc)
        self._emit_state_change("dreaming", "asleep", "study complete")
        logger.info("Exiting DREAMING — back to ASLEEP")

        # If a wake signal arrived while dreaming, start waking now
        if self.wake_signal_pending:
            logger.info("Pending wake signal — transitioning to WAKING")
            self.transition_to_waking()
        return True

    # ------------------------------------------------------------------
    # DISTRACTED transitions (resource-driven)
    # ------------------------------------------------------------------

    def enter_distracted(self) -> bool:
        """ASLEEP → DISTRACTED when sustained CPU/GPU load detected."""
        if self.state != GaiaState.ASLEEP:
            logger.warning("Cannot enter DISTRACTED from state: %s", self.state)
            return False
        self.state = GaiaState.DISTRACTED
        self.last_state_change = datetime.now(timezone.utc)
        self._emit_state_change("asleep", "distracted", "sustained load")
        logger.info("Entering DISTRACTED — system under load")
        return True

    def exit_distracted(self) -> bool:
        """DISTRACTED → ASLEEP when load drops below threshold."""
        if self.state != GaiaState.DISTRACTED:
            logger.warning("Cannot exit DISTRACTED from state: %s", self.state)
            return False
        self.state = GaiaState.ASLEEP
        self.last_state_change = datetime.now(timezone.utc)
        self._emit_state_change("distracted", "asleep", "load subsided")
        logger.info("Exiting DISTRACTED — back to ASLEEP")

        # If a wake signal arrived while distracted, start waking
        if self.wake_signal_pending:
            logger.info("Pending wake signal — transitioning to WAKING")
            self.transition_to_waking()
        return True

    # ------------------------------------------------------------------
    # OFFLINE transition
    # ------------------------------------------------------------------

    def initiate_offline(self) -> None:
        """ANY → OFFLINE for graceful shutdown."""
        prev = self.state
        self.state = GaiaState.OFFLINE
        self._phase = _TransientPhase.NONE
        self.last_state_change = datetime.now(timezone.utc)
        self._emit_state_change(prev.value, "offline", "shutdown")
        logger.info("Entering OFFLINE from %s", prev)

    # ------------------------------------------------------------------
    # Status / monitoring
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        status: Dict[str, Any] = {
            "state": self.state.value,
            "phase": self._phase.value,
            "wake_signal_pending": self.wake_signal_pending,
            "prime_available": self.prime_available,
            "current_task": self.current_task.get("task_id") if self.current_task else None,
            "last_state_change": self.last_state_change.isoformat(),
            "seconds_in_state": (now - self.last_state_change).total_seconds(),
        }
        status["voice_active"] = self.voice_active
        if self.dreaming_handoff_id:
            status["dreaming_handoff_id"] = self.dreaming_handoff_id
        return status

    def get_canned_response(self) -> Optional[str]:
        """Return a canned response if the current state warrants one, else None."""
        if self.state == GaiaState.DREAMING:
            return CANNED_DREAMING
        if self.state == GaiaState.DISTRACTED:
            return CANNED_DISTRACTED
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_checkpoint_as_review(checkpoint: str) -> str:
        """Format checkpoint as REVIEW material, NOT as a prompt.

        Injected at Tier 1 in prompt_builder alongside session summaries.
        The model must understand this is context restoration, not a user
        message requiring a response.
        """
        if not checkpoint:
            return ""

        return (
            "[SLEEP RESTORATION CONTEXT — Internal Review Only]\n"
            "These are your notes from your last active session before sleep.\n"
            "Use them to restore your working context. Do not respond to them directly.\n"
            "\n"
            f"{checkpoint}\n"
            "\n"
            f"Context restoration timestamp: {datetime.now(timezone.utc).isoformat()}"
        )

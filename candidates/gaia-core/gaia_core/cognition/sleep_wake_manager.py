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

    def __init__(self, config) -> None:
        self.config = config
        self.state = GaiaState.ACTIVE
        self._phase = _TransientPhase.NONE
        self.current_task: Optional[Dict[str, Any]] = None
        self.wake_signal_pending = False
        self.prime_available = False
        self.checkpoint_manager = PrimeCheckpointManager(config)
        self.last_state_change = datetime.now(timezone.utc)
        self.dreaming_handoff_id: Optional[str] = None

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
        logger.info("Entering DROWSY — writing checkpoint...")

        try:
            self.checkpoint_manager.create_checkpoint(current_packet)
            self.checkpoint_manager.rotate_checkpoints()

            # Check if we were interrupted during checkpoint write
            if self.wake_signal_pending:
                logger.info("Message arrived during DROWSY — cancelling sleep")
                self.state = GaiaState.ACTIVE
                self.wake_signal_pending = False
                self.last_state_change = datetime.now(timezone.utc)
                return False

            # Checkpoint complete — enter ASLEEP
            self.state = GaiaState.ASLEEP
            self.last_state_change = datetime.now(timezone.utc)
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

            self.state = GaiaState.ACTIVE
            self._phase = _TransientPhase.NONE
            self.wake_signal_pending = False
            self.prime_available = True
            self.last_state_change = datetime.now(timezone.utc)

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
        logger.info("Entering DISTRACTED — system under load")
        return True

    def exit_distracted(self) -> bool:
        """DISTRACTED → ASLEEP when load drops below threshold."""
        if self.state != GaiaState.DISTRACTED:
            logger.warning("Cannot exit DISTRACTED from state: %s", self.state)
            return False
        self.state = GaiaState.ASLEEP
        self.last_state_change = datetime.now(timezone.utc)
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

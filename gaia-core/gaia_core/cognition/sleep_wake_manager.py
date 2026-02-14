"""
GAIA Sleep/Wake State Machine.

Manages five states:
    AWAKE → DROWSY → SLEEPING → FINISHING_TASK/WAKING → AWAKE

Design decisions:
- DROWSY is cancellable: a message arriving during checkpoint writing
  aborts the transition and returns to AWAKE.
- WAKING uses a parallel strategy: CPU Lite handles the first queued
  message while Prime boots in the background (~37-60 s from tmpfs).
- prime.md checkpoint is the KV cache replacement.

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
    AWAKE = "awake"
    DROWSY = "drowsy"  # Checkpoint in progress — cancellable
    SLEEPING = "sleeping"  # Executing sleep tasks
    FINISHING_TASK = "finishing_task"  # Non-interruptible task completing
    WAKING = "waking"  # Context restoration + Prime boot


class SleepWakeManager:
    """Manages GAIA's sleep/wake state transitions with cognitive continuity."""

    def __init__(self, config) -> None:
        self.config = config
        self.state = GaiaState.AWAKE
        self.current_task: Optional[Dict[str, Any]] = None
        self.wake_signal_pending = False
        self.prime_available = False
        self.checkpoint_manager = PrimeCheckpointManager(config)
        self.last_state_change = datetime.now(timezone.utc)

        logger.info("SleepWakeManager initialized")

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_state(self) -> GaiaState:
        return self.state

    def should_transition_to_drowsy(self, idle_minutes: float) -> bool:
        """Check whether we should begin the sleep transition."""
        if self.state != GaiaState.AWAKE:
            return False
        threshold = getattr(self.config, "SLEEP_IDLE_THRESHOLD_MINUTES", 5)
        return idle_minutes >= threshold

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def initiate_drowsy(self, current_packet=None) -> bool:
        """Transition AWAKE → DROWSY → SLEEPING.

        During DROWSY, Prime writes its cognitive checkpoint.
        If a wake signal arrives before the checkpoint completes,
        the sleep is cancelled and we return to AWAKE.

        Returns True if we entered SLEEPING, False otherwise.
        """
        if self.state != GaiaState.AWAKE:
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
                self.state = GaiaState.AWAKE
                self.wake_signal_pending = False
                self.last_state_change = datetime.now(timezone.utc)
                return False

            # Checkpoint complete — enter SLEEPING
            self.state = GaiaState.SLEEPING
            self.last_state_change = datetime.now(timezone.utc)
            logger.info("Checkpoint written — entering SLEEPING")
            return True

        except Exception:
            logger.error("Checkpoint failed — staying AWAKE", exc_info=True)
            self.state = GaiaState.AWAKE
            self.last_state_change = datetime.now(timezone.utc)
            return False

    def receive_wake_signal(self) -> None:
        """Called by gaia-web (via POST /sleep/wake) when a message is queued."""
        self.wake_signal_pending = True

        if self.state == GaiaState.DROWSY:
            # initiate_drowsy() will notice the flag and cancel
            logger.info("Wake signal during DROWSY — will cancel checkpoint")

        elif self.state == GaiaState.SLEEPING:
            logger.info("Wake signal during SLEEPING")
            if self.current_task and not self.current_task.get("interruptible", True):
                task_id = self.current_task.get("task_id", "unknown")
                logger.info("Non-interruptible task running: %s", task_id)
                self.state = GaiaState.FINISHING_TASK
            else:
                self.transition_to_waking()

        elif self.state == GaiaState.AWAKE:
            logger.debug("Wake signal received but already AWAKE")
            self.wake_signal_pending = False

    def transition_to_waking(self) -> None:
        """Move to WAKING state.  Begins parallel wake strategy."""
        if self.state not in (GaiaState.SLEEPING, GaiaState.FINISHING_TASK):
            logger.warning("Cannot wake from state: %s", self.state)
            return
        self.state = GaiaState.WAKING
        self.last_state_change = datetime.now(timezone.utc)
        logger.info("Entering WAKING — starting parallel wake")

    def complete_wake(self) -> Dict[str, Any]:
        """Finish waking: load checkpoint, format as REVIEW context, go AWAKE.

        Returns a dict with ``checkpoint_loaded``, ``context``, and
        ``timestamp`` keys.
        """
        if self.state != GaiaState.WAKING:
            logger.warning("Cannot complete wake from state: %s", self.state)
            return {"checkpoint_loaded": False}

        try:
            checkpoint = self.checkpoint_manager.load_latest()
            review_context = self._format_checkpoint_as_review(checkpoint)

            self.state = GaiaState.AWAKE
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
            self.state = GaiaState.AWAKE
            self.wake_signal_pending = False
            return {"checkpoint_loaded": False}

    # ------------------------------------------------------------------
    # Status / monitoring
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        return {
            "state": self.state.value,
            "wake_signal_pending": self.wake_signal_pending,
            "prime_available": self.prime_available,
            "current_task": self.current_task.get("task_id") if self.current_task else None,
            "last_state_change": self.last_state_change.isoformat(),
            "seconds_in_state": (now - self.last_state_change).total_seconds(),
        }

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

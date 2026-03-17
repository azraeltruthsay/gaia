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
import os
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from gaia_core.cognition.prime_checkpoint import PrimeCheckpointManager
from gaia_core.cognition.council_notes import CouncilNoteManager

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
    "I'm studying right now — give me a moment to wrap up "
    "and I'll be right with you!"
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
        self._council_notes = CouncilNoteManager(config, timeline_store=timeline_store)
        self.last_state_change = datetime.now(timezone.utc)
        self.dreaming_handoff_id: Optional[str] = None
        self._preemption_initiated: bool = False
        self.voice_active: bool = False
        self._task_scheduler = None  # Set via set_task_scheduler()
        self.auto_sleep_enabled: bool = True

        logger.info("SleepWakeManager initialized")

    def set_task_scheduler(self, scheduler) -> None:
        """Store a reference to the SleepTaskScheduler for wake signal forwarding."""
        self._task_scheduler = scheduler

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_state(self) -> GaiaState:
        return self.state

    def should_transition_to_drowsy(self, idle_minutes: float) -> bool:
        """Check whether we should begin the sleep transition."""
        if self.state != GaiaState.ACTIVE:
            return False
        if not self.auto_sleep_enabled:
            return False
        # Respect time-boxed sleep hold (e.g., during CFR ingest / penpal pipeline)
        if self._is_hold_active():
            return False
        # Read from SLEEP_CYCLE config section, fall back to 30 minutes
        sleep_cfg = getattr(self.config, "SLEEP_CYCLE", None) or {}
        threshold = sleep_cfg.get("idle_threshold_minutes", 30) if isinstance(sleep_cfg, dict) else 30
        return idle_minutes >= threshold

    def set_auto_sleep(self, enabled: bool) -> None:
        """Enable or disable automatic sleep transitions."""
        self.auto_sleep_enabled = enabled
        self._sleep_hold_until = None  # Clear any hold when manually toggling
        logger.info("Auto-sleep %s", "enabled" if enabled else "disabled")

    def hold_wake(self, minutes: int = 30, reason: str = "") -> dict:
        """Temporarily suppress auto-sleep for the given duration.

        Auto-expires after `minutes` (max 120). The hold is checked in
        should_transition_to_drowsy() and automatically cleared when expired.
        Unlike set_auto_sleep(False), this is time-boxed and self-healing.
        """
        minutes = max(1, min(minutes, 120))  # Clamp 1-120
        self._sleep_hold_until = datetime.now(timezone.utc) + timedelta(minutes=minutes)
        self._sleep_hold_reason = reason or "long-form operation"
        # Also ensure we're ACTIVE
        if self.state != GaiaState.ACTIVE:
            self.receive_wake_signal(reason=f"hold_wake: {reason}")
        logger.info("Sleep hold active for %d minutes (reason: %s)", minutes, self._sleep_hold_reason)
        return {
            "hold_active": True,
            "expires_at": self._sleep_hold_until.isoformat(),
            "minutes": minutes,
            "reason": self._sleep_hold_reason,
        }

    def release_hold(self) -> dict:
        """Release a sleep hold early."""
        was_held = getattr(self, "_sleep_hold_until", None) is not None
        self._sleep_hold_until = None
        self._sleep_hold_reason = ""
        if was_held:
            logger.info("Sleep hold released")
        return {"hold_active": False, "was_held": was_held}

    def _is_hold_active(self) -> bool:
        """Check if a sleep hold is currently active (not expired)."""
        hold_until = getattr(self, "_sleep_hold_until", None)
        if hold_until is None:
            return False
        if datetime.now(timezone.utc) >= hold_until:
            # Auto-expire
            logger.info("Sleep hold expired (was: %s)", getattr(self, "_sleep_hold_reason", ""))
            self._sleep_hold_until = None
            self._sleep_hold_reason = ""
            return False
        return True

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
        """Notify gaia-audio to sleep/wake based on sleep state (best-effort)."""
        try:
            audio_cfg = self.config.INTEGRATIONS.get("audio", {})
            if not audio_cfg.get("enabled") or not audio_cfg.get("mute_on_sleep"):
                return

            audio_endpoint = self.config.get_endpoint("audio")

            if to_state in ("asleep", "dreaming"):
                if getattr(self, "voice_active", False):
                    logger.info("Skipping audio sleep — voice channel active")
                    return
                action = "sleep"
                timeout = self.config.get_timeout("HTTP_DEFAULT", 15.0)
            elif to_state == "active":
                action = "wake"
                timeout = self.config.get_timeout("HTTP_QUICK", 5.0)
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

        # Immediately signal the task scheduler so interruptible tasks can
        # bail out without waiting for the current handler to complete.
        if self._task_scheduler is not None:
            try:
                self._task_scheduler.signal_wake()
            except Exception:
                logger.debug("Failed to signal task scheduler wake", exc_info=True)

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
            logger.info("Wake signal during DREAMING — preemption will be triggered by sleep cycle loop")
            # Don't transition here — _handle_dreaming() in SleepCycleLoop
            # will detect wake_signal_pending and initiate GPU preemption

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

            # Load Council notes written since last sleep
            council_context = ""
            consumed_note_paths = []
            sleep_timestamp = self.checkpoint_manager.get_sleep_timestamp()
            pending = self._council_notes.read_pending_notes(since=sleep_timestamp)
            if pending:
                council_context = self._council_notes.format_notes_for_prime(pending)
                consumed_note_paths = [n["path"] for n in pending]
                logger.info("Loaded %d Council notes for Prime", len(pending))

            # LMCache status: check if disk-persisted KV chunks exist
            lmcache_status = self._check_lmcache_status()

            self.state = GaiaState.ACTIVE
            self._phase = _TransientPhase.NONE
            self.wake_signal_pending = False
            self.prime_available = True
            self.last_state_change = datetime.now(timezone.utc)
            self._emit_state_change("asleep", "active", "wake signal processed")

            # Log memory layer status
            if lmcache_status["kv_warm"] and checkpoint:
                logger.info("Wake complete — dual memory: LMCache KV warm (%d chunks) + prime.md checkpoint",
                            lmcache_status["chunk_count"])
            elif lmcache_status["kv_warm"]:
                logger.info("Wake complete — LMCache KV warm (%d chunks), no prime.md checkpoint",
                            lmcache_status["chunk_count"])
                logger.warning("LMCache has KV state but prime.md is missing/stale — semantic backup gap")
            elif checkpoint:
                logger.info("Wake complete — prime.md checkpoint loaded, LMCache cold (first wake or cache evicted)")
            else:
                logger.info("Wake complete — cold start (no LMCache, no checkpoint)")

            return {
                "checkpoint_loaded": bool(checkpoint),
                "context": review_context,
                "council_context": council_context,
                "council_note_paths": consumed_note_paths,
                "council_notes_count": len(consumed_note_paths),
                "lmcache": lmcache_status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception:
            logger.error("Wake completion failed", exc_info=True)
            self.state = GaiaState.ACTIVE
            self._phase = _TransientPhase.NONE
            self.wake_signal_pending = False
            return {"checkpoint_loaded": False}

    def get_pending_council_context(self):
        """Return pending council context without consuming notes.

        Used by agent_core to check for council notes on Prime's first
        response after waking.
        """
        sleep_timestamp = self.checkpoint_manager.get_sleep_timestamp()
        pending = self._council_notes.read_pending_notes(since=sleep_timestamp)
        if not pending:
            return None
        return {
            "council_context": self._council_notes.format_notes_for_prime(pending),
            "council_note_paths": [n["path"] for n in pending],
        }

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
        self._preemption_initiated = False
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
        self._preemption_initiated = False
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
        status["auto_sleep_enabled"] = self.auto_sleep_enabled
        sleep_cfg = getattr(self.config, "SLEEP_CYCLE", None) or {}
        status["idle_threshold_minutes"] = sleep_cfg.get("idle_threshold_minutes", 30) if isinstance(sleep_cfg, dict) else 30
        # Sleep hold info
        hold_until = getattr(self, "_sleep_hold_until", None)
        if hold_until and datetime.now(timezone.utc) < hold_until:
            status["sleep_hold"] = {
                "active": True,
                "expires_at": hold_until.isoformat(),
                "reason": getattr(self, "_sleep_hold_reason", ""),
            }
        status["voice_active"] = self.voice_active
        status["preemption_initiated"] = self._preemption_initiated
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
    def _check_lmcache_status() -> Dict[str, Any]:
        """Check whether LMCache has persisted KV chunks on disk."""
        kvcache_dir = Path(os.environ.get("LMCACHE_DISK_PATH", "/kvcache"))
        try:
            if kvcache_dir.is_dir():
                chunks = list(kvcache_dir.glob("*"))
                return {
                    "kv_warm": len(chunks) > 0,
                    "chunk_count": len(chunks),
                    "disk_path": str(kvcache_dir),
                }
        except Exception:
            logger.debug("LMCache disk check failed", exc_info=True)
        return {"kv_warm": False, "chunk_count": 0, "disk_path": str(kvcache_dir)}

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

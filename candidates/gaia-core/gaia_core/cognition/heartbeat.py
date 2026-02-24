"""
Thought Seed Heartbeat — regular-interval daemon that triages dormant seeds.

Runs independently of the sleep cycle on a configurable timer (default 20 min).
For each unreviewed seed, Lite performs a three-way triage:

    ARCHIVE — permanently dismiss (dead end)
    PENDING — defer for later review
    ACT     — expand with Lite, optionally wake Prime, run through agent_core

This replaces the old thought_seed_review and initiative_cycle sleep tasks,
keeping sleep focused on pure memory maintenance.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger("GAIA.Heartbeat")

# Session ID for heartbeat-initiated turns (stays internal)
HEARTBEAT_SESSION_ID = "gaia_heartbeat_session"

# Triage system prompt for Lite
_TRIAGE_SYSTEM_PROMPT = """\
You are GAIA's thought seed triage system. You will be shown a thought seed — \
a dormant idea that was generated during a previous conversation.

Respond with EXACTLY one of these words on the first line:
  ARCHIVE — This seed is no longer relevant, too vague, or not worth pursuing.
  PENDING — This seed has potential but should be revisited later.
  ACT     — This seed is actionable and worth expanding on right now.

On the second line, write a single sentence justifying your decision."""

_EXPAND_SYSTEM_PROMPT = """\
You are GAIA's thought seed expansion system. Given a thought seed and its \
context, expand it into a clear, actionable prompt that GAIA can process \
through her cognitive loop. The prompt should be specific, grounded, and \
self-contained — it will be fed directly into GAIA's reasoning engine.

Write ONLY the expanded prompt, nothing else."""


class ThoughtSeedHeartbeat:
    """Daemon thread that triages thought seeds on a regular interval."""

    def __init__(
        self,
        config,
        model_pool=None,
        agent_core=None,
        sleep_wake_manager=None,
        timeline_store=None,
        session_manager=None,
    ) -> None:
        self.config = config
        self.model_pool = model_pool
        self.agent_core = agent_core
        self.sleep_wake_manager = sleep_wake_manager
        self._timeline = timeline_store

        self._interval = getattr(config, "HEARTBEAT_INTERVAL_SECONDS", 1200)
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Temporal awareness subsystems (Phase 1 consciousness framework)
        self._lite_journal = None
        self._temporal_state_manager = None
        self._tick_count = 0
        self._bake_interval = getattr(config, "TEMPORAL_BAKE_INTERVAL_TICKS", 3)

        journal_enabled = getattr(config, "LITE_JOURNAL_ENABLED", True)
        temporal_enabled = getattr(config, "TEMPORAL_STATE_ENABLED", True)

        if journal_enabled:
            try:
                from gaia_core.cognition.lite_journal import LiteJournal
                self._lite_journal = LiteJournal(
                    config=config,
                    model_pool=model_pool,
                    timeline_store=timeline_store,
                    sleep_wake_manager=sleep_wake_manager,
                )
            except Exception:
                logger.warning("Failed to init LiteJournal", exc_info=True)

        if temporal_enabled:
            try:
                from gaia_core.cognition.temporal_state_manager import TemporalStateManager
                self._temporal_state_manager = TemporalStateManager(
                    config=config,
                    model_pool=model_pool,
                    timeline_store=timeline_store,
                    session_manager=session_manager,
                    lite_journal=self._lite_journal,
                )
            except Exception:
                logger.warning("Failed to init TemporalStateManager", exc_info=True)

        # Phase 2: Temporal interviewer (requires baked states)
        self._temporal_interviewer = None
        self._interview_interval = getattr(config, "TEMPORAL_INTERVIEW_INTERVAL_TICKS", 6)
        interview_enabled = getattr(config, "TEMPORAL_INTERVIEW_ENABLED", True)

        if interview_enabled and temporal_enabled and self._temporal_state_manager is not None:
            try:
                from gaia_core.cognition.temporal_interviewer import TemporalInterviewer
                self._temporal_interviewer = TemporalInterviewer(
                    config=config,
                    model_pool=model_pool,
                    temporal_state_manager=self._temporal_state_manager,
                    lite_journal=self._lite_journal,
                    timeline_store=timeline_store,
                )
            except Exception:
                logger.warning("Failed to init TemporalInterviewer", exc_info=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="ThoughtSeedHeartbeat",
        )
        self._thread.start()
        logger.info("Thought seed heartbeat started (interval=%ds)", self._interval)

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None
        logger.info("Thought seed heartbeat stopped")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        # Initial boot delay — let models and services stabilize
        time.sleep(60)
        while self._running:
            try:
                self._tick()
            except Exception:
                logger.error("Heartbeat tick failed", exc_info=True)
            # Sleep in short increments so stop() doesn't block for 20 min
            for _ in range(self._interval):
                if not self._running:
                    return
                time.sleep(1)

    def _tick(self) -> None:
        """One heartbeat cycle: promote overdue, triage unreviewed seeds."""
        from gaia_core.cognition.thought_seed import (
            list_pending_seeds_due,
            list_unreviewed_seeds,
        )

        # 1. Promote overdue pending seeds
        promoted = list_pending_seeds_due()
        if promoted:
            logger.info("Heartbeat: promoted %d overdue pending seeds", len(promoted))

        # 2. Gather unreviewed seeds
        seeds = list_unreviewed_seeds()
        if not seeds:
            # Still run journal + bake + interview even when no seeds to triage
            jw, sb, ic = self._run_temporal_tasks()
            self._emit_heartbeat_tick(
                0, 0, 0, 0,
                journal_written=jw, state_baked=sb, interview_conducted=ic,
            )
            return

        logger.info("Heartbeat: %d seeds to triage", len(seeds))

        # 3. Get Lite model
        llm = None
        if self.model_pool is not None:
            try:
                llm = self.model_pool.get_model_for_role("lite")
            except Exception:
                logger.warning("Heartbeat: could not get Lite model", exc_info=True)
        if llm is None:
            logger.warning("Heartbeat: no LLM available, skipping triage")
            return

        # 4. Triage each seed
        archived, deferred, acted = 0, 0, 0
        for seed_path, seed_data in seeds:
            try:
                decision, reason = self._triage_seed(llm, seed_data)
                filename = seed_path.name

                if decision == "archive":
                    self._do_archive(filename)
                    archived += 1
                elif decision == "act":
                    self._act_on_seed(llm, filename, seed_data)
                    acted += 1
                else:  # pending (default)
                    self._do_defer(filename)
                    deferred += 1
            except Exception:
                logger.error("Heartbeat: failed to triage seed %s", seed_path.name, exc_info=True)

        # --- Temporal awareness tasks (journal + state bake + interview) ---
        journal_written, state_baked, interview_conducted = self._run_temporal_tasks()

        self._emit_heartbeat_tick(
            len(seeds), archived, deferred, acted,
            journal_written=journal_written, state_baked=state_baked,
            interview_conducted=interview_conducted,
        )
        logger.info(
            "Heartbeat complete: %d triaged (archive=%d, pending=%d, act=%d) "
            "journal=%s bake=%s interview=%s",
            len(seeds), archived, deferred, acted,
            journal_written, state_baked, interview_conducted,
        )

    # ------------------------------------------------------------------
    # Temporal awareness tasks
    # ------------------------------------------------------------------

    def _run_temporal_tasks(self) -> tuple[bool, bool, bool]:
        """Run Lite.md journal write, optional KV state bake, and interview.

        Returns (journal_written, state_baked, interview_conducted).
        """
        self._tick_count += 1
        journal_written = False
        state_baked = False
        interview_conducted = False

        # --- Lite.md Journal Entry ---
        if self._lite_journal is not None:
            try:
                self._lite_journal.tick_count = self._tick_count
                entry = self._lite_journal.write_entry()
                if entry:
                    journal_written = True
                    logger.info("Heartbeat: Lite journal entry written (%d chars)", len(entry))
            except Exception:
                logger.error("Heartbeat: Lite journal write failed", exc_info=True)

        # --- Temporal State Bake (every N ticks) ---
        if (self._temporal_state_manager is not None
                and self._tick_count % self._bake_interval == 0):
            try:
                state_path = self._temporal_state_manager.bake_state()
                if state_path:
                    state_baked = True
                    logger.info("Heartbeat: Temporal state baked → %s", state_path)
            except Exception:
                logger.error("Heartbeat: Temporal state bake failed", exc_info=True)

        # --- Temporal Interview (every M ticks) ---
        if (self._temporal_interviewer is not None
                and self._tick_count % self._interview_interval == 0
                and self._tick_count > 0):
            should_interview = True
            if self.sleep_wake_manager is not None:
                try:
                    from gaia_core.cognition.sleep_wake_manager import GaiaState
                    state = self.sleep_wake_manager.get_state()
                    should_interview = state in (GaiaState.ACTIVE, GaiaState.DROWSY)
                except Exception:
                    should_interview = False

            if should_interview:
                try:
                    transcript = self._temporal_interviewer.conduct_interview()
                    if transcript:
                        interview_conducted = True
                        logger.info(
                            "Heartbeat: interview conducted (coherence=%.2f)",
                            transcript.get("coherence", {}).get("overall_coherence", -1),
                        )
                except Exception:
                    logger.error("Heartbeat: temporal interview failed", exc_info=True)

        return journal_written, state_baked, interview_conducted

    # ------------------------------------------------------------------
    # Triage
    # ------------------------------------------------------------------

    def _triage_seed(self, llm, seed_data: Dict[str, Any]) -> tuple[str, str]:
        """Ask Lite to classify a seed as ARCHIVE / PENDING / ACT.

        Returns (decision, reason). Defaults to "pending" on parse failure.
        Knowledge gap seeds are auto-routed to ACT without LLM triage.
        """
        # Fast-path: knowledge gap seeds skip LLM triage and go straight to ACT
        seed_type = seed_data.get("seed_type", "general")
        if seed_type == "knowledge_gap":
            return ("act", "Knowledge gap — auto-routing to research")

        seed_text = seed_data.get("seed", "")
        context = seed_data.get("context", {})
        user_prompt = (
            f"Thought seed: {seed_text}\n"
            f"Original context: {context}\n"
            f"Created: {seed_data.get('created', 'unknown')}"
        )

        try:
            result = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": _TRIAGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=128,
                stream=False,
            )
            text = result["choices"][0]["message"]["content"].strip()
            lines = text.splitlines()
            first_word = lines[0].strip().upper() if lines else ""
            reason = lines[1].strip() if len(lines) > 1 else ""

            if first_word == "ARCHIVE":
                return ("archive", reason)
            elif first_word == "ACT":
                return ("act", reason)
            else:
                return ("pending", reason)
        except Exception:
            logger.warning("Triage LLM call failed, defaulting to pending", exc_info=True)
            return ("pending", "LLM call failed")

    # ------------------------------------------------------------------
    # Archive / Defer helpers
    # ------------------------------------------------------------------

    def _do_archive(self, filename: str) -> None:
        from gaia_core.cognition.thought_seed import archive_seed
        archive_seed(filename)

    def _do_defer(self, filename: str) -> None:
        from gaia_core.cognition.thought_seed import defer_seed
        # Default revisit: 7 days from now
        revisit = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
        defer_seed(filename, revisit_after=revisit)

    # ------------------------------------------------------------------
    # Act path
    # ------------------------------------------------------------------

    def _act_on_seed(self, llm, seed_filename: str, seed_data: Dict[str, Any]) -> None:
        """Expand the seed with Lite, then run through agent_core."""
        # 1. Expand
        expanded = self._expand_seed(llm, seed_data)
        if not expanded:
            logger.warning("Heartbeat: expansion failed for %s, deferring", seed_filename)
            self._do_defer(seed_filename)
            return

        # 2. Check state — can we run a turn?
        if not self._ensure_active(seed_filename):
            return  # Deferred or skipped inside _ensure_active

        # 3. Execute via agent_core
        if self.agent_core is None:
            logger.warning("Heartbeat: no agent_core, deferring seed %s", seed_filename)
            self._do_defer(seed_filename)
            return

        try:
            # ACT seeds go through Prime — that's real cognitive work,
            # so reset the idle timer (run_turn bypasses /process_packet).
            if self.sleep_wake_manager is not None:
                idle_mon = getattr(self.sleep_wake_manager, "idle_monitor", None)
                if idle_mon is not None:
                    idle_mon.mark_active()

            logger.info("Heartbeat: acting on seed %s", seed_filename)
            for _event in self.agent_core.run_turn(
                user_input=expanded,
                session_id=HEARTBEAT_SESSION_ID,
                source="heartbeat",
                destination="log",
            ):
                pass  # Consume generator; results stay internal
        except Exception:
            logger.error("Heartbeat: run_turn failed for seed %s", seed_filename, exc_info=True)

        # 4. Archive the seed after acting
        from gaia_core.cognition.thought_seed import archive_seed
        archive_seed(seed_filename)

    def _expand_seed(self, llm, seed_data: Dict[str, Any]) -> str:
        """Use Lite to expand a thought seed into an actionable prompt."""
        seed_text = seed_data.get("seed", "")
        context = seed_data.get("context", {})
        user_prompt = (
            f"Thought seed: {seed_text}\n"
            f"Original context: {context}"
        )

        try:
            result = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": _EXPAND_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=512,
                stream=False,
            )
            return result["choices"][0]["message"]["content"].strip()
        except Exception:
            logger.warning("Seed expansion LLM call failed", exc_info=True)
            return ""

    def _ensure_active(self, seed_filename: str) -> bool:
        """Ensure GAIA is in an ACTIVE state for running a turn.

        If ASLEEP, sends a wake signal and polls for up to 180s.
        If DREAMING or DISTRACTED, defers the seed.
        If OFFLINE, skips entirely.

        Returns True if ACTIVE and ready, False otherwise.
        """
        if self.sleep_wake_manager is None:
            return True  # No state machine — assume ready

        from gaia_core.cognition.sleep_wake_manager import GaiaState

        state = self.sleep_wake_manager.get_state()

        if state == GaiaState.ACTIVE:
            return True

        if state == GaiaState.ASLEEP:
            logger.info("Heartbeat: waking Prime for seed %s", seed_filename)
            self.sleep_wake_manager.receive_wake_signal()
            # Poll for up to 180s
            for _ in range(90):
                time.sleep(2)
                if self.sleep_wake_manager.get_state() == GaiaState.ACTIVE:
                    return True
            logger.warning("Heartbeat: wake timed out, deferring seed %s", seed_filename)
            self._do_defer(seed_filename)
            return False

        if state in (GaiaState.DREAMING, GaiaState.DISTRACTED):
            logger.info("Heartbeat: GAIA is %s, deferring seed %s", state.value, seed_filename)
            self._do_defer(seed_filename)
            return False

        if state == GaiaState.OFFLINE:
            logger.info("Heartbeat: GAIA is OFFLINE, skipping seed %s", seed_filename)
            return False

        # DROWSY or unknown — defer to be safe
        self._do_defer(seed_filename)
        return False

    # ------------------------------------------------------------------
    # Timeline events
    # ------------------------------------------------------------------

    def _emit_heartbeat_tick(
        self, seeds_found: int, archived: int, deferred: int, acted: int,
        journal_written: bool = False, state_baked: bool = False,
        interview_conducted: bool = False,
    ) -> None:
        if self._timeline is not None:
            try:
                self._timeline.append("heartbeat_tick", {
                    "seeds_found": seeds_found,
                    "archived": archived,
                    "deferred": deferred,
                    "acted": acted,
                    "journal_written": journal_written,
                    "state_baked": state_baked,
                    "interview_conducted": interview_conducted,
                    "tick_number": self._tick_count,
                })
            except Exception:
                logger.debug("Timeline heartbeat_tick emit failed", exc_info=True)

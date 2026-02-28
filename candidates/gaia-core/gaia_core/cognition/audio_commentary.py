"""
Audio Commentary Evaluator — daemon that monitors the audio context buffer
and pushes commentary to the dashboard when GAIA hears something interesting.

Runs as an independent daemon thread (not coupled to heartbeat).
When audio listening is active, periodically reads new buffer entries,
triages them with Lite, and generates commentary via AgentCore.run_turn()
when warranted.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("GAIA.AudioCommentary")

COMMENTARY_SESSION_ID = "gaia_audio_commentary"

_TRIAGE_SYSTEM_PROMPT = """\
You are evaluating ambient audio context to decide if GAIA should comment.
The user asked you to listen. These are recent transcriptions.

Respond with EXACTLY one of:
  SKIP — Nothing worth commenting on (silence, mundane noise, repetitive filler)
  COMMENT — Something interesting, notable, or conversation-worthy

Second line: one-sentence summary of what caught your attention (or why you're skipping)."""


class AudioCommentaryEvaluator:
    """Daemon thread that evaluates audio buffer and generates commentary."""

    def __init__(
        self,
        model_pool=None,
        agent_core=None,
        sleep_wake_manager=None,
        config=None,
    ) -> None:
        self.model_pool = model_pool
        self.agent_core = agent_core
        self.sleep_wake_manager = sleep_wake_manager

        # Load config from constants
        constants = {}
        if config is not None:
            constants = getattr(config, "constants", {})
        audio_cfg = constants.get("AUDIO_COMMENTARY", {})

        self._interval = audio_cfg.get("interval_seconds", 90)
        self._min_gap = audio_cfg.get("min_gap_seconds", 60)
        self._triage_max_tokens = audio_cfg.get("lite_triage_max_tokens", 128)
        self._commentary_max_tokens = audio_cfg.get("commentary_max_tokens", 512)
        self._enabled = audio_cfg.get("enabled", True)

        self._poll_interval = 10  # seconds between checks when listening inactive
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # State tracking
        self._last_evaluated_ts: float = 0.0  # ingested_at of last evaluated entry
        self._last_commented_at: float = 0.0  # time.time() of last commentary

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if not self._enabled:
            logger.info("Audio commentary daemon disabled by config")
            return
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="AudioCommentaryDaemon",
        )
        self._thread.start()
        logger.info(
            "Audio commentary daemon started (interval=%ds, min_gap=%ds)",
            self._interval, self._min_gap,
        )

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None
        logger.info("Audio commentary daemon stopped")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        # Initial boot delay — let models and services stabilize
        time.sleep(30)

        while self._running:
            try:
                listening = self._is_listening()

                if listening:
                    self._evaluate_cycle()
                    # Sleep the full interval in 1s increments (for clean shutdown)
                    for _ in range(self._interval):
                        if not self._running:
                            return
                        time.sleep(1)
                else:
                    # Idle polling — check every _poll_interval seconds
                    for _ in range(self._poll_interval):
                        if not self._running:
                            return
                        time.sleep(1)
            except Exception:
                logger.error("Audio commentary cycle failed", exc_info=True)
                time.sleep(10)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def _evaluate_cycle(self) -> None:
        """One evaluation cycle: read new entries, triage, optionally comment."""
        # Skip if GAIA is asleep
        if not self._is_awake():
            logger.debug("Audio commentary: skipping cycle (GAIA not active)")
            return

        # Enforce minimum gap between commentaries
        now = time.time()
        if self._last_commented_at and (now - self._last_commented_at) < self._min_gap:
            logger.debug("Audio commentary: skipping cycle (min_gap not met)")
            return

        # Get new buffer entries
        new_entries = self._get_new_entries()
        if not new_entries:
            logger.debug("Audio commentary: no new entries since last eval")
            return

        # Update last evaluated timestamp
        self._last_evaluated_ts = max(
            e.get("ingested_at", 0) for e in new_entries
        )

        # Triage with Lite
        should_comment, reason = self._triage(new_entries)
        if not should_comment:
            logger.debug("Audio commentary: SKIP — %s", reason)
            return

        logger.info("Audio commentary: COMMENT — %s", reason)
        self._generate_commentary(new_entries, reason)
        self._last_commented_at = time.time()

    def _triage(self, entries: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Ask Lite to decide if the audio buffer is worth commenting on."""
        if self.model_pool is None:
            return False, "no model pool"

        llm = None
        try:
            llm = self.model_pool.get_model_for_role("lite")
        except Exception:
            logger.debug("Audio commentary: could not get Lite model")
            return False, "Lite model unavailable"

        if llm is None:
            return False, "Lite model unavailable"

        # Format entries for the triage prompt
        entry_text = self._format_entries(entries)
        user_prompt = f"Recent audio transcriptions:\n\n{entry_text}"

        try:
            result = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": _TRIAGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=self._triage_max_tokens,
                stream=False,
            )
            text = result["choices"][0]["message"]["content"].strip()
            lines = text.splitlines()
            first_word = lines[0].strip().upper() if lines else ""
            reason = lines[1].strip() if len(lines) > 1 else ""

            if "COMMENT" in first_word:
                return True, reason
            return False, reason
        except Exception:
            logger.warning("Audio commentary triage failed", exc_info=True)
            return False, "triage call failed"

    def _generate_commentary(
        self, entries: List[Dict[str, Any]], reason: str
    ) -> None:
        """Generate commentary via AgentCore.run_turn() with destination=web."""
        if self.agent_core is None:
            logger.warning("Audio commentary: no agent_core available")
            return

        # Build the prompt — the audio context is already injected by
        # prompt_builder (section 5.7) when _audio_listening_active is True.
        # We just need a user-input that directs the model to comment.
        prompt = (
            f"[Audio commentary trigger] You're actively listening to ambient audio. "
            f"Something caught your attention: {reason}\n\n"
            f"React naturally to what you're hearing. Be conversational, brief, and genuine. "
            f"Don't explain that you're an AI listening — just comment on what you heard."
        )

        try:
            for _event in self.agent_core.run_turn(
                user_input=prompt,
                session_id=COMMENTARY_SESSION_ID,
                source="audio_listen",
                destination="web",
            ):
                pass  # Consume generator; output routes via output_router
            logger.info("Audio commentary turn completed")
        except Exception:
            logger.error("Audio commentary run_turn failed", exc_info=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_listening(self) -> bool:
        """Check if audio listening is currently active."""
        try:
            from gaia_core.main import _audio_listening_active
            return _audio_listening_active
        except ImportError:
            return False

    def _is_awake(self) -> bool:
        """Check if GAIA is in an active/drowsy state (not asleep/offline)."""
        if self.sleep_wake_manager is None:
            return True
        try:
            from gaia_core.cognition.sleep_wake_manager import GaiaState
            state = self.sleep_wake_manager.get_state()
            return state in (GaiaState.ACTIVE, GaiaState.DROWSY)
        except Exception:
            return False

    def _get_new_entries(self) -> List[Dict[str, Any]]:
        """Get buffer entries newer than the last evaluated timestamp."""
        try:
            from gaia_core.main import _audio_context_buffer
            return [
                e for e in _audio_context_buffer
                if e.get("ingested_at", 0) > self._last_evaluated_ts
            ]
        except ImportError:
            return []

    @staticmethod
    def _format_entries(entries: List[Dict[str, Any]]) -> str:
        """Format buffer entries as readable text for the triage prompt."""
        lines = []
        for entry in entries:
            markers = entry.get("context_markers", [])
            marker_str = f" [{', '.join(markers)}]" if markers else ""
            lines.append(f"[{entry.get('timestamp', '??:??')}]{marker_str} {entry.get('text', '')}")
        return "\n".join(lines)

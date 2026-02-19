"""
Temporal Interviewer — Prime interviews past-Lite via KV cache state swapping.

Phase 2 of the Temporal Awareness Framework.  Prime formulates structured
questions about a past moment, Lite answers from its restored KV cache state
(which contains the cognitive context of that moment), and the transcript is
analysed for narrative coherence against the corresponding journal entries.

Interview cycle:
  1.  Select a past state (oldest un-interviewed baked state)
  2.  Hold _LITE_LOCK for the duration of steps 3-6
  3.  Save current Lite state in memory (fast memcpy, not disk)
  4.  Load past state into Lite's KV cache
  5.  Multi-turn Q&A — Prime asks, past-Lite answers (2-4 rounds)
  6.  Restore current Lite state from memory
  7.  Release _LITE_LOCK
  8.  Analyse narrative coherence (journal vs. interview, outside lock)
  9.  Save transcript to disk
  10. Emit timeline event

Thread safety:
  _LITE_LOCK (from temporal_state_manager) is held for steps 3-6 only.
  This prevents lite_journal, bake_state, or intent detection from touching
  Lite while its KV cache is in a past state.  Coherence analysis (step 8)
  uses Prime, not Lite, so the lock is already released.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GAIA.TemporalInterviewer")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PRIME_INTERVIEW_SYSTEM = """\
You are GAIA's temporal self-awareness interviewer.  You are questioning a past \
version of yourself (Lite) to understand what it was experiencing at a specific \
moment in time.

Your goal: extract concrete, specific information about:
- What Lite was doing and processing
- What patterns or anomalies it noticed
- Its emotional / cognitive tone
- Unfinished threads or open questions

Ask ONE clear, specific question.  Do not ask multiple questions at once. \
Build on previous answers when available.  Be curious but focused."""

_PAST_LITE_SYSTEM = """\
You are GAIA-Lite at a specific moment in time.  Answer from your actual \
cognitive state — what you genuinely know, feel, and remember right now. \
Do not fabricate details.  If you are uncertain about something, say so. \
Speak in first person.  Be specific and concrete."""

# Round-specific focus instructions appended to Prime's user message.
_ROUND_FOCUS = [
    "Ask an open-ended orientation question: what is Lite doing, what is on its mind?",
    "Ask about specific details based on the previous answer: patterns noticed, context, unfinished work.",
    "Ask about subjective / emotional state: cognitive tone, what feels important.",
    "Ask a synthesis question: if Lite could leave a message for its future self, what would it be?",
]

_COHERENCE_SYSTEM = """\
You are GAIA's narrative coherence analyst.  Compare two sources about the same \
moment in GAIA's experience: a written journal and a recalled interview."""

_COHERENCE_USER_TEMPLATE = """\
## Journal Entries (written at the time)
{journal_text}

## Interview Transcript (recalled from KV cache state)
{interview_text}

Analyse the coherence between written record and recalled experience.
Respond in EXACTLY this format (one item per line):

TOPIC_OVERLAP: [0.0-1.0] [brief note]
TONE_CONSISTENCY: [0.0-1.0] [brief note]
INFO_LOSS: [topics in journal but missing from interview, comma-separated, or "none"]
INFO_GAIN: [topics in interview but missing from journal, comma-separated, or "none"]
OVERALL: [0.0-1.0] [one-sentence assessment]"""


class TemporalInterviewer:
    """Orchestrates Prime-interviews-past-Lite sessions."""

    INTERVIEW_ROUNDS = 3
    MAX_ROUNDS = 4
    TRANSCRIPT_DIR_NAME = "interviews"

    def __init__(
        self,
        config,
        model_pool=None,
        temporal_state_manager=None,
        lite_journal=None,
        timeline_store=None,
    ) -> None:
        self.config = config
        self.model_pool = model_pool
        self._tsm = temporal_state_manager
        self._lite_journal = lite_journal
        self._timeline = timeline_store

        self.interview_rounds = min(
            getattr(config, "TEMPORAL_INTERVIEW_ROUNDS", self.INTERVIEW_ROUNDS),
            self.MAX_ROUNDS,
        )

        shared_dir = getattr(config, "SHARED_DIR", "/shared")
        self.transcript_dir = (
            Path(shared_dir) / "temporal_states" / self.TRANSCRIPT_DIR_NAME
        )

        try:
            self.transcript_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Could not create transcript dir: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def conduct_interview(
        self, state_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Run a full interview session against a past Lite state.

        If *state_id* is ``None``, selects the oldest un-interviewed state
        that is not the current (newest) state.

        Returns the transcript dict, or ``None`` on failure / nothing to
        interview.
        """
        if self.model_pool is None or self._tsm is None:
            return None

        # --- Select target ---
        if state_id is not None:
            target = self._tsm.get_state_metadata(state_id)
            if target is None:
                logger.warning("Interview: state %s not found", state_id)
                return None
            target["state_id"] = state_id
        else:
            target = self._select_interview_target()

        if target is None:
            logger.debug("Interview: no suitable past state to interview")
            return None

        target_id = target["state_id"]
        state_path = self._tsm.state_dir / f"{target_id}.bin"
        if not state_path.exists():
            logger.warning("Interview: state file missing: %s", state_path)
            return None

        # --- Get Lite model instance ---
        try:
            llm = self.model_pool.get_model_for_role("lite")
        except Exception:
            logger.warning("Interview: could not get Lite model", exc_info=True)
            return None
        if llm is None:
            return None

        start_ms = time.monotonic()

        # --- Interview (under lock) ---
        from gaia_core.cognition.temporal_state_manager import _LITE_LOCK

        rounds: List[Dict[str, str]] = []
        with _LITE_LOCK:
            try:
                # Save current state in memory (fast)
                saved_current = self._tsm.save_current_state_memory(llm)

                # Load past state into KV cache
                loaded = self._tsm._load_lite_state(llm, state_path)
                if not loaded:
                    logger.error("Interview: failed to load past state %s", target_id)
                    # Restore current before returning
                    self._tsm.restore_state_memory(llm, saved_current)
                    return None

                # Multi-turn interview
                try:
                    rounds = self._run_interview_rounds(llm, target)
                except Exception:
                    logger.error("Interview: round execution failed", exc_info=True)
                finally:
                    # ALWAYS restore current state
                    self._tsm.restore_state_memory(llm, saved_current)

            except Exception:
                logger.error("Interview: state swap failed", exc_info=True)
                return None

        elapsed_ms = int((time.monotonic() - start_ms) * 1000)

        if not rounds:
            return None

        # --- Coherence analysis (outside lock, uses Prime) ---
        coherence = self._analyze_coherence(rounds, target)

        # --- Build and save transcript ---
        transcript = self._build_transcript(
            target_id, target, rounds, coherence, elapsed_ms
        )
        self._save_transcript(transcript)
        self._emit_interview_event(transcript)

        logger.info(
            "Interview complete: state=%s rounds=%d coherence=%.2f duration=%dms",
            target_id,
            len(rounds),
            coherence.get("overall_coherence", -1),
            elapsed_ms,
        )

        return transcript

    # ------------------------------------------------------------------
    # Target Selection
    # ------------------------------------------------------------------

    def _select_interview_target(self) -> Optional[Dict[str, Any]]:
        """Select a past state to interview.

        Strategy:
          - List all available states
          - Exclude the most recent (current) state
          - Prefer states that have not been interviewed yet
          - Among candidates, pick the oldest (most temporally distant)
        """
        states = self._tsm.list_states()
        if len(states) < 2:
            # Need at least 2 states (one current, one past)
            return None

        # Exclude the newest (current) state
        candidates = states[:-1]

        # Separate un-interviewed from already-interviewed
        uninterviewed = []
        interviewed = []
        for s in candidates:
            if self._has_transcript(s["state_id"]):
                interviewed.append(s)
            else:
                uninterviewed.append(s)

        # Prefer un-interviewed, oldest first
        if uninterviewed:
            return uninterviewed[0]
        if interviewed:
            return interviewed[0]
        return None

    def _has_transcript(self, state_id: str) -> bool:
        """Check if we already have an interview transcript for this state."""
        return any(self.transcript_dir.glob(f"interview_{state_id}_*.json"))

    # ------------------------------------------------------------------
    # Interview Execution
    # ------------------------------------------------------------------

    def _run_interview_rounds(
        self, llm, state_metadata: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Execute the multi-turn interview: Prime asks, past-Lite answers.

        Returns list of ``{"question": ..., "answer": ...}`` dicts.
        """
        rounds: List[Dict[str, str]] = []
        state_ts = state_metadata.get("timestamp", "unknown time")

        for i in range(self.interview_rounds):
            # --- Prime formulates question ---
            question = self._prime_ask(rounds, state_ts, i)
            if not question:
                logger.warning("Interview: Prime returned empty question at round %d", i)
                break

            # --- Past-Lite answers ---
            answer = self._lite_answer(llm, rounds, question)
            if not answer:
                logger.warning("Interview: Lite returned empty answer at round %d", i)
                answer = "(no response)"

            rounds.append({"question": question, "answer": answer})

        return rounds

    def _prime_ask(
        self,
        previous_rounds: List[Dict[str, str]],
        state_ts: str,
        round_idx: int,
    ) -> str:
        """Have Prime formulate an interview question."""
        if self.model_pool is None:
            return ""

        # Build context for Prime
        focus = _ROUND_FOCUS[min(round_idx, len(_ROUND_FOCUS) - 1)]

        user_parts = [
            f"Interview with past-Lite (state from {state_ts})",
        ]
        if previous_rounds:
            user_parts.append("\nConversation so far:")
            for r in previous_rounds:
                user_parts.append(f"Q: {r['question']}")
                user_parts.append(f"A: {r['answer']}\n")

        user_parts.append(f"Round {round_idx + 1} focus: {focus}")
        user_parts.append("Ask your question.")

        messages = [
            {"role": "system", "content": _PRIME_INTERVIEW_SYSTEM},
            {"role": "user", "content": "\n".join(user_parts)},
        ]

        try:
            result = self.model_pool.forward_to_model(
                "prime",
                messages=messages,
                release=True,
                max_tokens=200,
                temperature=0.4,
            )
            return result["choices"][0]["message"]["content"].strip()
        except Exception:
            logger.warning("Interview: Prime question failed", exc_info=True)
            return ""

    def _lite_answer(
        self,
        llm,
        previous_rounds: List[Dict[str, str]],
        question: str,
    ) -> str:
        """Have past-Lite answer a question (KV cache is in past state)."""
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": _PAST_LITE_SYSTEM},
        ]

        # Include prior rounds as conversation context
        for r in previous_rounds:
            messages.append({"role": "user", "content": r["question"]})
            messages.append({"role": "assistant", "content": r["answer"]})

        messages.append({"role": "user", "content": question})

        try:
            result = llm.create_chat_completion(
                messages=messages,
                temperature=0.3,
                max_tokens=300,
                stream=False,
            )
            return result["choices"][0]["message"]["content"].strip()
        except Exception:
            logger.warning("Interview: Lite answer failed", exc_info=True)
            return ""

    # ------------------------------------------------------------------
    # Narrative Coherence
    # ------------------------------------------------------------------

    def _analyze_coherence(
        self,
        transcript_rounds: List[Dict[str, str]],
        state_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare interview responses against journal entries for coherence.

        Detects topic consistency, emotional tone shifts, information
        loss/gain.  Uses Prime for analysis (outside _LITE_LOCK).
        """
        if self.model_pool is None:
            return self._default_coherence()

        # Collect journal entries
        journal_text = self._get_journal_for_state(state_metadata)
        if not journal_text:
            journal_text = "(no journal entries available for this time period)"

        # Build interview text
        interview_lines = []
        for i, r in enumerate(transcript_rounds, 1):
            interview_lines.append(f"Q{i}: {r['question']}")
            interview_lines.append(f"A{i}: {r['answer']}")
        interview_text = "\n".join(interview_lines)

        user_msg = _COHERENCE_USER_TEMPLATE.format(
            journal_text=journal_text,
            interview_text=interview_text,
        )

        messages = [
            {"role": "system", "content": _COHERENCE_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        try:
            result = self.model_pool.forward_to_model(
                "prime",
                messages=messages,
                release=True,
                max_tokens=300,
                temperature=0.2,
            )
            analysis_text = result["choices"][0]["message"]["content"].strip()
            return self._parse_coherence(analysis_text)
        except Exception:
            logger.warning("Interview: coherence analysis failed", exc_info=True)
            return self._default_coherence()

    def _get_journal_for_state(self, state_metadata: Dict[str, Any]) -> str:
        """Load journal entries temporally close to the state's timestamp."""
        if self._lite_journal is None:
            return ""

        try:
            entries = self._lite_journal.load_recent_entries(n=5)
            if not entries:
                return ""

            # If we have a timestamp, try to find entries near it
            state_ts = state_metadata.get("timestamp", "")
            if state_ts:
                # Filter entries whose timestamp is close to the state's
                matched = []
                for entry in entries:
                    # Extract timestamp from "## Entry: {ts}" header
                    m = re.match(r"## Entry:\s*(\S+)", entry)
                    if m:
                        matched.append(entry)
                # Use matched entries, or fall back to all
                if matched:
                    return "\n\n".join(matched)

            return "\n\n".join(entries)
        except Exception:
            return ""

    def _parse_coherence(self, text: str) -> Dict[str, Any]:
        """Parse structured coherence output from Prime."""
        result = self._default_coherence()

        # TOPIC_OVERLAP: 0.85 Some note
        m = re.search(r"TOPIC_OVERLAP:\s*([\d.]+)", text)
        if m:
            result["topic_overlap"] = min(1.0, max(0.0, float(m.group(1))))

        m = re.search(r"TONE_CONSISTENCY:\s*([\d.]+)", text)
        if m:
            result["tone_consistency"] = min(1.0, max(0.0, float(m.group(1))))

        m = re.search(r"INFO_LOSS:\s*(.+)", text)
        if m:
            val = m.group(1).strip()
            if val.lower() == "none":
                result["information_loss"] = []
            else:
                result["information_loss"] = [
                    s.strip() for s in val.split(",") if s.strip()
                ]

        m = re.search(r"INFO_GAIN:\s*(.+)", text)
        if m:
            val = m.group(1).strip()
            if val.lower() == "none":
                result["information_gain"] = []
            else:
                result["information_gain"] = [
                    s.strip() for s in val.split(",") if s.strip()
                ]

        m = re.search(r"OVERALL:\s*([\d.]+)\s*(.*)", text)
        if m:
            result["overall_coherence"] = min(1.0, max(0.0, float(m.group(1))))
            result["narrative"] = m.group(2).strip()

        return result

    @staticmethod
    def _default_coherence() -> Dict[str, Any]:
        return {
            "topic_overlap": -1.0,
            "tone_consistency": -1.0,
            "information_loss": [],
            "information_gain": [],
            "overall_coherence": -1.0,
            "narrative": "",
        }

    # ------------------------------------------------------------------
    # Transcript Storage
    # ------------------------------------------------------------------

    def _build_transcript(
        self,
        state_id: str,
        state_metadata: Dict[str, Any],
        rounds: List[Dict[str, str]],
        coherence: Dict[str, Any],
        duration_ms: int,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        return {
            "state_id": state_id,
            "interview_timestamp": now.isoformat(),
            "state_timestamp": state_metadata.get("timestamp", ""),
            "gaia_state": state_metadata.get("gaia_state", "unknown"),
            "heartbeat_tick": state_metadata.get("heartbeat_tick", 0),
            "rounds": rounds,
            "round_count": len(rounds),
            "coherence": coherence,
            "duration_ms": duration_ms,
        }

    def _save_transcript(self, transcript: Dict[str, Any]) -> Optional[Path]:
        """Save the interview transcript to disk as JSON."""
        state_id = transcript["state_id"]
        ts_safe = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        filename = f"interview_{state_id}_{ts_safe}.json"
        path = self.transcript_dir / filename

        try:
            path.write_text(
                json.dumps(transcript, indent=2, default=str),
                encoding="utf-8",
            )
            logger.info("Interview transcript saved: %s", path.name)
            return path
        except OSError as exc:
            logger.error("Failed to save transcript: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Timeline Events
    # ------------------------------------------------------------------

    def _emit_interview_event(self, transcript: Dict[str, Any]) -> None:
        if self._timeline is None:
            return
        try:
            self._timeline.append("temporal_interview", {
                "state_id": transcript["state_id"],
                "state_timestamp": transcript["state_timestamp"],
                "round_count": transcript["round_count"],
                "duration_ms": transcript["duration_ms"],
                "overall_coherence": transcript["coherence"].get(
                    "overall_coherence", -1
                ),
                "topic_overlap": transcript["coherence"].get(
                    "topic_overlap", -1
                ),
            })
        except Exception:
            logger.debug("Timeline interview event emit failed", exc_info=True)

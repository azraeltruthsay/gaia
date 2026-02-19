"""
Lite Cognitive Journal — running introspective log written by the Lite model.

Mirrors the PrimeCheckpointManager pattern: regular writes, timestamped entries,
rotation to history directory when the journal grows too long.

Unlike prime.md (written on sleep), Lite.md is written every heartbeat tick
(~20 min) and captures Lite's running operational state — what it's been doing,
patterns it notices, and unresolved threads.

Storage lives on the shared Docker volume for persistence across restarts.
"""

from __future__ import annotations

import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from gaia_core.cognition.temporal_state_manager import _LITE_LOCK

logger = logging.getLogger("GAIA.LiteJournal")

_JOURNAL_SYSTEM_PROMPT = """\
You are GAIA-Lite writing a brief journal entry about your current cognitive state. \
Write in first person. Be specific and concrete — reference actual activities, \
patterns, and observations from the context provided. Keep it under 100 words. \
Do NOT use generic filler or placeholders. Do NOT use markdown headings."""

_JOURNAL_USER_TEMPLATE = """\
Current time: {semantic_time}
GAIA state: {state} for {state_duration}
Active sessions: {session_count}
Heartbeat tick: #{heartbeat_count}
Last activity: {last_activity}
Recent events:
{recent_events}

Write your journal entry."""


class LiteJournal:
    """Manages Lite's introspective journal (Lite.md)."""

    JOURNAL_FILENAME = "Lite.md"
    MAX_ENTRIES = 50
    HISTORY_DIR_NAME = "lite_history"

    def __init__(
        self,
        config,
        model_pool=None,
        timeline_store=None,
        sleep_wake_manager=None,
    ) -> None:
        self.config = config
        self.model_pool = model_pool
        self._timeline = timeline_store
        self._swm = sleep_wake_manager

        shared_dir = getattr(config, "SHARED_DIR", "/shared")
        self.journal_dir = Path(shared_dir) / "lite_journal"
        self.journal_file = self.journal_dir / self.JOURNAL_FILENAME
        self.history_dir = self.journal_dir / self.HISTORY_DIR_NAME

        # Heartbeat tick counter (set externally by heartbeat)
        self.tick_count = 0

        # Ensure directories exist
        try:
            self.journal_dir.mkdir(parents=True, exist_ok=True)
            self.history_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Could not create journal dirs: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_entry(self) -> Optional[str]:
        """Generate and append a journal entry using Lite.

        Returns the entry text, or None if generation failed.
        """
        llm = None
        if self.model_pool is not None:
            try:
                llm = self.model_pool.get_model_for_role("lite")
            except Exception:
                logger.warning("LiteJournal: could not get Lite model", exc_info=True)

        if llm is None:
            return None

        try:
            with _LITE_LOCK:
                entry_text = self._generate_entry(llm)
            if not entry_text:
                return None

            self._append_entry(entry_text)
            logger.info("Lite journal entry written (%d chars)", len(entry_text))
            return entry_text
        except Exception:
            logger.error("LiteJournal: write_entry failed", exc_info=True)
            return None

    def load_latest(self) -> str:
        """Load the current journal file contents."""
        if not self.journal_file.exists():
            return ""
        try:
            return self.journal_file.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not read journal: %s", exc)
            return ""

    def load_recent_entries(self, n: int = 5) -> List[str]:
        """Parse and return the N most recent entry blocks."""
        content = self.load_latest()
        if not content:
            return []

        # Split on ## Entry: headers
        entries = re.split(r"(?=^## Entry: )", content, flags=re.MULTILINE)
        # Filter out non-entry blocks (header, empty)
        entries = [e.strip() for e in entries if e.strip().startswith("## Entry:")]
        # Return most recent N (entries are appended, so last N)
        return entries[-n:] if len(entries) > n else entries

    def rotate(self) -> None:
        """Rotate journal when it exceeds MAX_ENTRIES.

        Moves current Lite.md to lite_history/{timestamp}-lite.md and
        creates a fresh journal with just the header.
        """
        if not self.journal_file.exists():
            return

        try:
            self.history_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
            archive_path = self.history_dir / f"{ts}-lite.md"
            shutil.copy2(self.journal_file, archive_path)
            logger.info("Lite journal rotated to %s", archive_path)

            # Write fresh header
            self.journal_file.write_text(
                "# Lite Cognitive Journal\n\n", encoding="utf-8"
            )
        except OSError as exc:
            logger.error("Lite journal rotation failed: %s", exc, exc_info=True)

    def get_entry_count(self) -> int:
        """Count entry headers in current journal."""
        content = self.load_latest()
        return len(re.findall(r"^## Entry: ", content, flags=re.MULTILINE))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate_entry(self, llm) -> str:
        """Ask Lite to write a journal entry given current context."""
        system_msg, user_msg = self._build_journal_prompt()

        try:
            result = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.4,
                max_tokens=200,
                stream=False,
            )
            return result["choices"][0]["message"]["content"].strip()
        except Exception:
            logger.warning("Lite journal generation failed", exc_info=True)
            return ""

    def _build_journal_prompt(self) -> tuple[str, str]:
        """Build system + user prompts for journal generation."""
        now = datetime.now(timezone.utc)
        semantic_time = now.strftime("%A %Y-%m-%d, %H:%M UTC")

        # State info
        state = "unknown"
        state_duration = "unknown"
        if self._swm is not None:
            try:
                status = self._swm.get_status()
                state = status.get("state", "unknown").upper()
                secs = status.get("seconds_in_state", 0)
                state_duration = self._format_duration(secs)
            except Exception:
                pass

        # Session count
        session_count = "unknown"

        # Recent timeline events
        recent_events = "none available"
        if self._timeline is not None:
            try:
                events = self._timeline.recent_events(limit=5)
                if events:
                    lines = []
                    for e in events:
                        lines.append(f"- [{e.ts[:19]}] {e.event}: {self._summarize_event(e.data)}")
                    recent_events = "\n".join(lines)
            except Exception:
                pass

        # Last activity
        last_activity = "unknown"
        if self._timeline is not None:
            try:
                last_msg = self._timeline.last_event_of_type("message")
                if last_msg and last_msg.timestamp:
                    gap = (now - last_msg.timestamp).total_seconds()
                    last_activity = f"{self._format_duration(gap)} ago"
            except Exception:
                pass

        user_msg = _JOURNAL_USER_TEMPLATE.format(
            semantic_time=semantic_time,
            state=state,
            state_duration=state_duration,
            session_count=session_count,
            heartbeat_count=self.tick_count,
            last_activity=last_activity,
            recent_events=recent_events,
        )

        return _JOURNAL_SYSTEM_PROMPT, user_msg

    def _append_entry(self, entry_text: str) -> None:
        """Append a formatted entry block to Lite.md."""
        now = datetime.now(timezone.utc)
        ts = now.isoformat()

        # Build metadata line
        state = "unknown"
        state_duration = "unknown"
        if self._swm is not None:
            try:
                status = self._swm.get_status()
                state = status.get("state", "unknown").upper()
                secs = status.get("seconds_in_state", 0)
                state_duration = self._format_duration(secs)
            except Exception:
                pass

        entry_block = (
            f"\n## Entry: {ts}\n"
            f"**State:** {state} for {state_duration} | **Heartbeat:** #{self.tick_count}\n"
            f"{entry_text}\n"
        )

        # Create file with header if it doesn't exist
        if not self.journal_file.exists():
            self.journal_file.write_text(
                "# Lite Cognitive Journal\n", encoding="utf-8"
            )

        # Append
        with open(self.journal_file, "a", encoding="utf-8") as f:
            f.write(entry_block)

        # Check rotation
        if self.get_entry_count() > self.MAX_ENTRIES:
            self.rotate()

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format seconds into human-readable: '2h 15m', '45m', '<1m'."""
        if seconds < 60:
            return "<1m"
        minutes = int(seconds / 60)
        if minutes < 60:
            return f"{minutes}m"
        hours = minutes // 60
        remaining = minutes % 60
        if remaining == 0:
            return f"{hours}h"
        return f"{hours}h {remaining}m"

    @staticmethod
    def _summarize_event(data: Dict[str, Any]) -> str:
        """One-line summary of a timeline event's data dict."""
        if not data:
            return ""
        # Pick the most informative fields
        parts = []
        for key in ("from", "to", "session_id", "method", "seeds_found", "success"):
            if key in data:
                parts.append(f"{key}={data[key]}")
        if parts:
            return ", ".join(parts[:3])
        # Fallback: first key-value
        first_key = next(iter(data))
        return f"{first_key}={data[first_key]}"

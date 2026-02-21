"""
Council Notes -- structured handoff notes from Lite to Prime.

When Prime is asleep and Lite fields a prompt that deserves deeper thought,
Lite writes a Council note capturing the user's prompt, her quick take, and
why the topic warrants Prime's attention.  When Prime wakes, she reads these
notes and follows up naturally -- owning Lite's answer as her own quick
instinct before going deeper.

Storage layout (on the shared Docker volume):
    /shared/council/notes/       -- pending notes (one .md file each)
    /shared/council/archive/     -- consumed notes moved here after Prime reads them

Follows the same pattern as LiteJournal (lite_journal.py) and
PrimeCheckpointManager (prime_checkpoint.py).
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from gaia_core.cognition.temporal_state_manager import _LITE_LOCK

logger = logging.getLogger("GAIA.CouncilNotes")


class CouncilNoteManager:
    """Manages Council notes between Lite and Prime."""

    NOTES_DIR_NAME = "notes"
    ARCHIVE_DIR_NAME = "archive"
    NOTE_SUFFIX = "-council.md"
    MAX_PENDING_NOTES = 10
    NOTE_MAX_AGE_HOURS = 24

    def __init__(self, config, timeline_store=None):
        self.config = config
        self._timeline = timeline_store

        constants = getattr(config, "constants", {})
        council_cfg = constants.get("COUNCIL", {})
        self._enabled = council_cfg.get("enabled", True)
        self.MAX_PENDING_NOTES = council_cfg.get("max_pending_notes", 10)
        self.NOTE_MAX_AGE_HOURS = council_cfg.get("note_max_age_hours", 24)
        self._archive_consumed = council_cfg.get("archive_consumed", True)

        shared_dir = getattr(config, "SHARED_DIR", "/shared")
        self.council_dir = Path(shared_dir) / "council"
        self.notes_dir = self.council_dir / self.NOTES_DIR_NAME
        self.archive_dir = self.council_dir / self.ARCHIVE_DIR_NAME

        try:
            self.notes_dir.mkdir(parents=True, exist_ok=True)
            self.archive_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Could not create council dirs: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_note(self, user_prompt, lite_response, escalation_reason,
                   session_id, metadata=None):
        """Write a Council note from Lite to Prime."""
        if not self._enabled:
            logger.debug("Council notes disabled; skipping write")
            return None

        self._enforce_cap()

        now = datetime.now(timezone.utc)
        ts_str = now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond:06d}Z"
        filename = f"{ts_str}{self.NOTE_SUFFIX}"
        note_path = self.notes_dir / filename

        content = self._format_note(
            user_prompt=user_prompt,
            lite_response=lite_response,
            escalation_reason=escalation_reason,
            session_id=session_id,
            timestamp=now,
            metadata=metadata,
        )

        try:
            with _LITE_LOCK:
                note_path.write_text(content, encoding="utf-8")
            logger.info("Council note written: %s", filename)

            if self._timeline:
                self._timeline.append("council_note", {
                    "action": "write",
                    "path": str(note_path),
                    "session_id": session_id,
                    "reason": escalation_reason[:200],
                })

            return note_path
        except OSError as exc:
            logger.error("Failed to write council note: %s", exc)
            return None

    def read_pending_notes(self, since=None):
        """Read all pending notes, optionally filtered to after *since*.

        Returns list of dicts sorted chronologically (oldest first).
        """
        notes = []

        if not self.notes_dir.exists():
            return notes

        for note_path in sorted(self.notes_dir.glob(f"*{self.NOTE_SUFFIX}")):
            ts = self._parse_timestamp_from_filename(note_path.name)
            if since and ts and ts <= since:
                continue
            if ts and self._is_expired(ts):
                self._archive_note(note_path)
                continue
            try:
                content = note_path.read_text(encoding="utf-8")
                parsed = self._parse_note(content)
                parsed["path"] = note_path
                parsed["filename"] = note_path.name
                notes.append(parsed)
            except OSError as exc:
                logger.warning("Could not read council note %s: %s",
                               note_path.name, exc)

        return notes

    def mark_notes_consumed(self, note_paths):
        """Move consumed notes to the archive directory."""
        for path in note_paths:
            if not isinstance(path, Path):
                path = Path(path)
            self._archive_note(path)

    def format_notes_for_prime(self, notes):
        """Format pending notes as review context for Prime's system prompt."""
        if not notes:
            return ""

        lines = [
            "[COUNCIL NOTES -- Messages from your quick-thinking self]",
            "You responded to these while half-asleep. Review them and",
            "follow up where your quick take deserves deeper thought.",
            "",
        ]

        for i, note in enumerate(notes, 1):
            ts = note.get("timestamp", "unknown")
            lines.append(f"### Note {i} ({ts})")
            lines.append(
                f"**User asked:** {note.get('user_prompt', '(unknown)')}")
            lines.append(
                f"**Your quick take:** {note.get('lite_response', '(none)')}")
            lines.append(
                "**Why it needs more:** "
                f"{note.get('escalation_reason', '(unknown)')}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_note(self, user_prompt, lite_response, escalation_reason,
                     session_id, timestamp, metadata=None):
        ts_iso = timestamp.isoformat()
        parts = [
            "# Council Note",
            "**From:** Lite",
            "**To:** Prime",
            f"**Timestamp:** {ts_iso}",
            f"**Session:** {session_id}",
            f"**Escalation:** {escalation_reason}",
            "",
            "## User Prompt",
            user_prompt,
            "",
            "## Lite's Quick Take",
            lite_response,
            "",
            "## Why This Needs Deeper Thought",
            escalation_reason,
        ]
        if metadata:
            parts.extend(["", "## Metadata"])
            for k, v in metadata.items():
                parts.append(f"- **{k}:** {v}")
        parts.append("")
        return "\n".join(parts)

    def _parse_note(self, content):
        """Parse a council note markdown file into a dict."""
        result = {}
        lines = content.split("\n")
        section = None
        section_lines = []

        for line in lines:
            if line.startswith("**Timestamp:**"):
                result["timestamp"] = line.split("**Timestamp:**", 1)[1].strip()
            elif line.startswith("**Session:**"):
                result["session_id"] = line.split("**Session:**", 1)[1].strip()
            elif line.startswith("**Escalation:**"):
                result["escalation_reason"] = line.split("**Escalation:**", 1)[1].strip()
            elif line.startswith("## User Prompt"):
                if section and section_lines:
                    result[section] = "\n".join(section_lines).strip()
                section = "user_prompt"
                section_lines = []
            elif line.startswith("## Lite"):
                if section and section_lines:
                    result[section] = "\n".join(section_lines).strip()
                section = "lite_response"
                section_lines = []
            elif line.startswith("## Why This Needs Deeper Thought"):
                if section and section_lines:
                    result[section] = "\n".join(section_lines).strip()
                section = "deeper_reason"
                section_lines = []
            elif line.startswith("## Metadata"):
                if section and section_lines:
                    result[section] = "\n".join(section_lines).strip()
                section = None
                section_lines = []
            elif section is not None:
                section_lines.append(line)

        if section and section_lines:
            result[section] = "\n".join(section_lines).strip()

        return result

    def _parse_timestamp_from_filename(self, filename):
        """Extract datetime from filename like 20260221T143200123456Z-council.md."""
        ts_part = filename.replace(self.NOTE_SUFFIX, "")
        # Try microsecond-precision format first, then second-precision fallback
        for fmt in ("%Y%m%dT%H%M%S%fZ", "%Y%m%dT%H%M%SZ"):
            try:
                return datetime.strptime(ts_part, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _is_expired(self, ts):
        age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
        return age_hours > self.NOTE_MAX_AGE_HOURS

    def _enforce_cap(self):
        """If pending notes exceed the cap, archive the oldest."""
        if not self.notes_dir.exists():
            return
        pending = sorted(self.notes_dir.glob(f"*{self.NOTE_SUFFIX}"))
        while len(pending) >= self.MAX_PENDING_NOTES:
            oldest = pending.pop(0)
            logger.info("Council note cap reached; archiving %s", oldest.name)
            self._archive_note(oldest)

    def _archive_note(self, path):
        """Move a note to archive (or delete if archiving disabled)."""
        if not path.exists():
            return
        try:
            if self._archive_consumed:
                dest = self.archive_dir / path.name
                shutil.move(str(path), str(dest))
                logger.debug("Archived council note: %s", path.name)
            else:
                path.unlink()
                logger.debug("Deleted council note: %s", path.name)
        except OSError as exc:
            logger.warning("Could not archive/delete %s: %s",
                           path.name, exc)

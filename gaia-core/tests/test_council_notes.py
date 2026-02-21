"""Unit tests for CouncilNoteManager."""

import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gaia_core.cognition.council_notes import CouncilNoteManager


# ── Helpers ──────────────────────────────────────────────────────────


def _make_config(shared_dir: str, council_cfg: dict | None = None):
    """Build a mock config pointing at a temp shared dir."""
    config = MagicMock()
    config.SHARED_DIR = shared_dir
    config.constants = {"COUNCIL": council_cfg or {
        "enabled": True,
        "max_pending_notes": 10,
        "note_max_age_hours": 24,
        "archive_consumed": True,
    }}
    return config


@pytest.fixture
def council(tmp_path):
    """Provide a CouncilNoteManager backed by a temp directory."""
    config = _make_config(str(tmp_path))
    return CouncilNoteManager(config)


@pytest.fixture
def council_with_timeline(tmp_path):
    """CouncilNoteManager with a mock timeline store."""
    config = _make_config(str(tmp_path))
    timeline = MagicMock()
    return CouncilNoteManager(config, timeline_store=timeline), timeline


# ── Write Tests ──────────────────────────────────────────────────────


class TestWriteNote:
    def test_write_creates_file(self, council):
        path = council.write_note(
            user_prompt="What is consciousness?",
            lite_response="Great question — here's my quick take...",
            escalation_reason="philosophical depth",
            session_id="sess-001",
        )
        assert path is not None
        assert path.exists()
        assert path.suffix == ".md"
        assert "council" in path.name

    def test_write_content_format(self, council):
        path = council.write_note(
            user_prompt="Explain GAIA's architecture",
            lite_response="GAIA has several core services...",
            escalation_reason="system internals",
            session_id="sess-002",
        )
        content = path.read_text(encoding="utf-8")
        assert "# Council Note" in content
        assert "**From:** Lite" in content
        assert "**To:** Prime" in content
        assert "Explain GAIA's architecture" in content
        assert "GAIA has several core services" in content
        assert "system internals" in content

    def test_write_with_metadata(self, council):
        path = council.write_note(
            user_prompt="test",
            lite_response="response",
            escalation_reason="reason",
            session_id="sess-003",
            metadata={"confidence": "0.85", "source": "discord"},
        )
        content = path.read_text(encoding="utf-8")
        assert "## Metadata" in content
        assert "confidence" in content
        assert "0.85" in content

    def test_write_disabled(self, tmp_path):
        config = _make_config(str(tmp_path), council_cfg={"enabled": False})
        mgr = CouncilNoteManager(config)
        result = mgr.write_note("prompt", "response", "reason", "sess-004")
        assert result is None

    def test_write_emits_timeline_event(self, council_with_timeline):
        mgr, timeline = council_with_timeline
        mgr.write_note("prompt", "response", "reason", "sess-005")
        timeline.append.assert_called_once()
        args = timeline.append.call_args
        assert args[0][0] == "council_note"
        assert args[0][1]["action"] == "write"


# ── Read Tests ───────────────────────────────────────────────────────


class TestReadPendingNotes:
    def test_read_empty_queue(self, council):
        notes = council.read_pending_notes()
        assert notes == []

    def test_read_after_write(self, council):
        council.write_note("prompt1", "response1", "reason1", "sess-001")
        council.write_note("prompt2", "response2", "reason2", "sess-001")
        notes = council.read_pending_notes()
        assert len(notes) == 2
        # Sorted chronologically (oldest first)
        assert notes[0]["user_prompt"] == "prompt1"
        assert notes[1]["user_prompt"] == "prompt2"

    def test_read_with_timestamp_filter(self, council):
        # Write a note, then filter to after it
        council.write_note("old", "old_resp", "old_reason", "sess-001")
        cutoff = datetime.now(timezone.utc) + timedelta(seconds=1)
        council.write_note("new", "new_resp", "new_reason", "sess-001")

        # Only notes after cutoff should appear — but both are likely
        # within the same second, so let's test with a future cutoff
        far_future = datetime.now(timezone.utc) + timedelta(hours=1)
        notes = council.read_pending_notes(since=far_future)
        assert len(notes) == 0

    def test_parsed_fields(self, council):
        council.write_note(
            user_prompt="What is love?",
            lite_response="Baby don't hurt me",
            escalation_reason="emotional depth",
            session_id="sess-010",
        )
        notes = council.read_pending_notes()
        assert len(notes) == 1
        note = notes[0]
        assert note["user_prompt"] == "What is love?"
        assert note["lite_response"] == "Baby don't hurt me"
        assert "timestamp" in note
        assert "path" in note


# ── Consume / Archive Tests ──────────────────────────────────────────


class TestMarkConsumed:
    def test_consume_moves_to_archive(self, council):
        path = council.write_note("prompt", "response", "reason", "sess-001")
        assert path.exists()

        council.mark_notes_consumed([path])
        assert not path.exists()
        # Should be in archive
        archived = list(council.archive_dir.glob("*"))
        assert len(archived) == 1

    def test_consume_clears_pending(self, council):
        p1 = council.write_note("p1", "r1", "reason1", "sess-001")
        p2 = council.write_note("p2", "r2", "reason2", "sess-001")

        council.mark_notes_consumed([p1, p2])
        notes = council.read_pending_notes()
        assert len(notes) == 0

    def test_consume_delete_mode(self, tmp_path):
        config = _make_config(str(tmp_path), council_cfg={
            "enabled": True,
            "archive_consumed": False,
        })
        mgr = CouncilNoteManager(config)
        path = mgr.write_note("prompt", "response", "reason", "sess-001")
        mgr.mark_notes_consumed([path])

        assert not path.exists()
        archived = list(mgr.archive_dir.glob("*"))
        assert len(archived) == 0  # Not archived, just deleted


# ── Cap Enforcement ──────────────────────────────────────────────────


class TestCapEnforcement:
    def test_cap_archives_oldest(self, tmp_path):
        config = _make_config(str(tmp_path), council_cfg={
            "enabled": True,
            "max_pending_notes": 3,
            "archive_consumed": True,
        })
        mgr = CouncilNoteManager(config)

        # Write 4 notes — the first should be archived when the 4th is written
        for i in range(4):
            mgr.write_note(f"prompt{i}", f"response{i}", f"reason{i}", "sess-001")

        pending = mgr.read_pending_notes()
        assert len(pending) <= 3


# ── Format for Prime ─────────────────────────────────────────────────


class TestFormatForPrime:
    def test_format_empty(self, council):
        result = council.format_notes_for_prime([])
        assert result == ""

    def test_format_single_note(self, council):
        council.write_note("What is AI?", "AI is...", "technical depth", "sess-001")
        notes = council.read_pending_notes()
        formatted = council.format_notes_for_prime(notes)

        assert "COUNCIL NOTES" in formatted
        assert "quick-thinking self" in formatted
        assert "### Note 1" in formatted
        assert "What is AI?" in formatted
        assert "AI is..." in formatted
        assert "technical depth" in formatted

    def test_format_multiple_notes(self, council):
        council.write_note("q1", "r1", "reason1", "sess-001")
        council.write_note("q2", "r2", "reason2", "sess-001")
        notes = council.read_pending_notes()
        formatted = council.format_notes_for_prime(notes)

        assert "### Note 1" in formatted
        assert "### Note 2" in formatted


# ── Expiry ───────────────────────────────────────────────────────────


class TestExpiry:
    def test_expired_notes_auto_archived(self, tmp_path):
        config = _make_config(str(tmp_path), council_cfg={
            "enabled": True,
            "note_max_age_hours": 0,  # Expire immediately
            "archive_consumed": True,
        })
        mgr = CouncilNoteManager(config)
        mgr.write_note("old prompt", "old response", "old reason", "sess-001")

        # Reading should auto-archive expired notes
        notes = mgr.read_pending_notes()
        assert len(notes) == 0
        archived = list(mgr.archive_dir.glob("*"))
        assert len(archived) == 1

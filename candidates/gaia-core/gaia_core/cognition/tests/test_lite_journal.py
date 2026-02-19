"""Tests for LiteJournal — Lite's introspective journal system."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gaia_core.cognition.lite_journal import LiteJournal


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def mock_config(tmp_path):
    config = MagicMock()
    config.SHARED_DIR = str(tmp_path)
    return config


def _mock_llm(response: str = "I've been handling intent detection.") -> MagicMock:
    llm = MagicMock()
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": response}}]
    }
    return llm


def _mock_pool(response: str = "I've been handling intent detection.") -> MagicMock:
    pool = MagicMock()
    pool.get_model_for_role.return_value = _mock_llm(response)
    return pool


def _mock_swm(state: str = "active", seconds: float = 3600.0) -> MagicMock:
    swm = MagicMock()
    swm.get_status.return_value = {"state": state, "seconds_in_state": seconds}
    return swm


# ── TestJournalLifecycle ────────────────────────────────────────────


class TestJournalLifecycle:
    def test_write_creates_journal_file(self, mock_config):
        journal = LiteJournal(config=mock_config, model_pool=_mock_pool())
        entry = journal.write_entry()

        assert entry is not None
        assert journal.journal_file.exists()
        content = journal.journal_file.read_text(encoding="utf-8")
        assert "# Lite Cognitive Journal" in content
        assert "## Entry:" in content

    def test_write_appends_to_existing(self, mock_config):
        journal = LiteJournal(config=mock_config, model_pool=_mock_pool())
        journal.write_entry()
        journal.write_entry()

        content = journal.load_latest()
        assert content.count("## Entry:") == 2

    def test_returns_none_without_llm(self, mock_config):
        pool = MagicMock()
        pool.get_model_for_role.return_value = None
        journal = LiteJournal(config=mock_config, model_pool=pool)

        assert journal.write_entry() is None

    def test_returns_none_without_model_pool(self, mock_config):
        journal = LiteJournal(config=mock_config, model_pool=None)
        assert journal.write_entry() is None

    def test_entry_format_has_timestamp(self, mock_config):
        journal = LiteJournal(config=mock_config, model_pool=_mock_pool())
        journal.write_entry()

        content = journal.load_latest()
        # Should contain ISO8601 timestamp after "## Entry: "
        match = re.search(r"## Entry: (\d{4}-\d{2}-\d{2}T)", content)
        assert match is not None

    def test_entry_includes_state_metadata(self, mock_config):
        swm = _mock_swm(state="active", seconds=7200)
        journal = LiteJournal(
            config=mock_config, model_pool=_mock_pool(), sleep_wake_manager=swm,
        )
        journal.tick_count = 5
        journal.write_entry()

        content = journal.load_latest()
        assert "**State:**" in content
        assert "ACTIVE" in content
        assert "**Heartbeat:** #5" in content


# ── TestRotation ────────────────────────────────────────────────────


class TestRotation:
    def test_rotate_when_max_entries_exceeded(self, mock_config):
        journal = LiteJournal(config=mock_config, model_pool=_mock_pool())
        journal.MAX_ENTRIES = 2

        journal.write_entry()
        journal.write_entry()
        journal.write_entry()  # Triggers rotation after append

        # History dir should have at least one file
        history_files = list(journal.history_dir.glob("*.md"))
        assert len(history_files) >= 1

    def test_history_dir_created_on_rotate(self, mock_config):
        journal = LiteJournal(config=mock_config, model_pool=_mock_pool())
        # history_dir is created in __init__, but verify it exists after rotation
        journal.MAX_ENTRIES = 1
        journal.write_entry()
        journal.write_entry()  # Should trigger rotation

        assert journal.history_dir.exists()

    def test_rotated_file_is_timestamped(self, mock_config):
        journal = LiteJournal(config=mock_config, model_pool=_mock_pool())
        journal.MAX_ENTRIES = 1
        journal.write_entry()
        journal.write_entry()  # Triggers rotation

        history_files = list(journal.history_dir.glob("*-lite.md"))
        assert len(history_files) >= 1
        # Filename should contain date pattern
        name = history_files[0].name
        assert re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-lite\.md", name)


# ── TestLoadEntries ─────────────────────────────────────────────────


class TestLoadEntries:
    def test_load_latest_returns_full_content(self, mock_config):
        journal = LiteJournal(config=mock_config, model_pool=_mock_pool())
        journal.write_entry()

        content = journal.load_latest()
        assert len(content) > 0
        assert "## Entry:" in content

    def test_load_recent_entries_returns_n(self, mock_config):
        journal = LiteJournal(config=mock_config, model_pool=_mock_pool())
        for _ in range(5):
            journal.write_entry()

        entries = journal.load_recent_entries(n=2)
        assert len(entries) == 2
        # Each entry should start with ## Entry:
        for entry in entries:
            assert entry.startswith("## Entry:")

    def test_load_recent_entries_empty_journal(self, mock_config):
        journal = LiteJournal(config=mock_config, model_pool=_mock_pool())
        entries = journal.load_recent_entries(n=5)
        assert entries == []

    def test_get_entry_count(self, mock_config):
        journal = LiteJournal(config=mock_config, model_pool=_mock_pool())
        assert journal.get_entry_count() == 0
        journal.write_entry()
        assert journal.get_entry_count() == 1
        journal.write_entry()
        assert journal.get_entry_count() == 2

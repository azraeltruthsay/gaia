"""Tests for the Temporal Context Builder."""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock
from pathlib import Path

import pytest

from gaia_core.utils.temporal_context import (
    _format_duration,
    _semantic_time,
    _session_summary,
    _state_summary,
    _code_evolution_summary,
    build_temporal_context,
)


class TestSemanticTime:
    def test_includes_day_of_week(self):
        # 2026-02-18 is a Wednesday
        dt = datetime(2026, 2, 18, 14, 30, tzinfo=timezone.utc)
        result = _semantic_time(dt)
        assert "Wednesday" in result

    def test_morning(self):
        dt = datetime(2026, 2, 18, 9, 0, tzinfo=timezone.utc)
        assert "(morning)" in _semantic_time(dt)

    def test_afternoon(self):
        dt = datetime(2026, 2, 18, 14, 0, tzinfo=timezone.utc)
        assert "(afternoon)" in _semantic_time(dt)

    def test_evening(self):
        dt = datetime(2026, 2, 18, 19, 0, tzinfo=timezone.utc)
        assert "(evening)" in _semantic_time(dt)

    def test_night(self):
        dt = datetime(2026, 2, 18, 23, 0, tzinfo=timezone.utc)
        assert "(night)" in _semantic_time(dt)

    def test_early_morning(self):
        dt = datetime(2026, 2, 18, 3, 0, tzinfo=timezone.utc)
        assert "(early morning)" in _semantic_time(dt)


class TestFormatDuration:
    def test_sub_minute(self):
        assert _format_duration(30) == "<1m"

    def test_minutes(self):
        assert _format_duration(300) == "5m"

    def test_hours_and_minutes(self):
        assert _format_duration(8100) == "2h 15m"

    def test_exact_hours(self):
        assert _format_duration(7200) == "2h"


class TestSessionSummary:
    def test_full_session(self):
        now = datetime.now(timezone.utc)
        created = now - timedelta(minutes=45)
        last_msg = now - timedelta(minutes=3)
        result = _session_summary("test", created, 12, last_msg)
        assert "45m old" in result
        assert "12 messages" in result
        assert "3m ago" in result

    def test_no_data_returns_empty(self):
        assert _session_summary(None, None, 0, None) == ""


class TestStateSummary:
    def test_active_state(self):
        result = _state_summary({"state": "active", "seconds_in_state": 3600})
        assert "ACTIVE" in result
        assert "1h" in result

    def test_empty_returns_empty(self):
        assert _state_summary({}) == ""


class TestCodeEvolutionSummary:
    def test_reads_snapshot(self, tmp_path):
        snapshot = tmp_path / "code_evolution.md"
        snapshot.write_text(
            "# Code Evolution Snapshot\n"
            "## Pending Candidate Changes\n"
            "- **gaia-core**: 3 changed\n"
            "- **gaia-web**: 1 added\n"
            "## Recent Commits\n"
        )
        result = _code_evolution_summary(str(snapshot))
        assert "2 services" in result
        assert "gaia-core" in result
        assert "gaia-web" in result

    def test_no_changes(self, tmp_path):
        snapshot = tmp_path / "code_evolution.md"
        snapshot.write_text("All candidates match production.")
        result = _code_evolution_summary(str(snapshot))
        assert "match production" in result

    def test_missing_file(self):
        result = _code_evolution_summary("/nonexistent/path")
        assert result == ""


class TestBuildTemporalContext:
    def test_minimal_output(self):
        result = build_temporal_context()
        assert "[Temporal Context]" in result
        # Should at least have semantic time
        assert "UTC" in result

    def test_with_session_data(self):
        now = datetime.now(timezone.utc)
        result = build_temporal_context(
            session_id="test",
            session_created_at=now - timedelta(minutes=30),
            session_message_count=8,
            last_message_ts=now - timedelta(minutes=2),
        )
        assert "30m old" in result
        assert "8 messages" in result

    def test_graceful_with_broken_timeline(self):
        """Should not crash even if timeline store raises."""
        broken = MagicMock()
        broken.events_by_type.side_effect = RuntimeError("broken")
        broken.events_since.side_effect = RuntimeError("broken")
        result = build_temporal_context(timeline_store=broken)
        # Should still have semantic time at minimum
        assert "[Temporal Context]" in result

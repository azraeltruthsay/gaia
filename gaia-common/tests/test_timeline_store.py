"""Tests for the Timeline Store — GAIA's temporal event log."""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from gaia_common.utils.timeline_store import TimelineEvent, TimelineStore


@pytest.fixture
def tmp_timeline(tmp_path):
    """Create a TimelineStore backed by a temp directory."""
    return TimelineStore(timeline_dir=str(tmp_path))


# ── TimelineEvent serialization ──────────────────────────────────────────


class TestTimelineEventSerialization:
    def test_to_json_line_roundtrip(self):
        event = TimelineEvent(
            ts="2026-02-18T22:00:00+00:00",
            event="state_change",
            data={"from": "active", "to": "drowsy"},
        )
        line = event.to_json_line()
        restored = TimelineEvent.from_json_line(line)
        assert restored is not None
        assert restored.ts == event.ts
        assert restored.event == event.event
        assert restored.data == event.data

    def test_from_json_line_invalid_returns_none(self):
        assert TimelineEvent.from_json_line("not json") is None
        assert TimelineEvent.from_json_line("") is None
        assert TimelineEvent.from_json_line("{}") is not None  # valid but empty

    def test_timestamp_property(self):
        event = TimelineEvent(ts="2026-02-18T22:00:00+00:00", event="test", data={})
        assert event.timestamp is not None
        assert event.timestamp.year == 2026

    def test_timestamp_property_invalid(self):
        event = TimelineEvent(ts="not-a-date", event="test", data={})
        assert event.timestamp is None


# ── TimelineStore append ─────────────────────────────────────────────────


class TestTimelineStoreAppend:
    def test_append_creates_daily_file(self, tmp_timeline, tmp_path):
        tmp_timeline.append("test_event", {"key": "value"})
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        expected = tmp_path / f"gaia_timeline_{date_str}.jsonl"
        assert expected.exists()
        content = expected.read_text()
        assert "test_event" in content

    def test_append_writes_valid_jsonl(self, tmp_timeline, tmp_path):
        tmp_timeline.append("state_change", {"from": "active", "to": "asleep"})
        tmp_timeline.append("message", {"session_id": "test", "role": "user"})

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = tmp_path / f"gaia_timeline_{date_str}.jsonl"
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2

        for line in lines:
            obj = json.loads(line)
            assert "ts" in obj
            assert "event" in obj
            assert "data" in obj

    def test_event_type_preserved(self, tmp_timeline):
        tmp_timeline.append("checkpoint", {"method": "llm"})
        events = tmp_timeline.recent_events(limit=1)
        assert len(events) == 1
        assert events[0].event == "checkpoint"
        assert events[0].data["method"] == "llm"


# ── TimelineStore queries ────────────────────────────────────────────────


class TestTimelineStoreQuery:
    def test_recent_events_returns_newest_first(self, tmp_timeline):
        tmp_timeline.append("event_a", {"order": 1})
        tmp_timeline.append("event_b", {"order": 2})
        tmp_timeline.append("event_c", {"order": 3})

        events = tmp_timeline.recent_events(limit=10)
        assert len(events) == 3
        assert events[0].event == "event_c"
        assert events[2].event == "event_a"

    def test_recent_events_limit(self, tmp_timeline):
        for i in range(10):
            tmp_timeline.append("bulk", {"i": i})
        events = tmp_timeline.recent_events(limit=3)
        assert len(events) == 3

    def test_events_by_type_filters(self, tmp_timeline):
        tmp_timeline.append("state_change", {"to": "active"})
        tmp_timeline.append("message", {"role": "user"})
        tmp_timeline.append("state_change", {"to": "drowsy"})

        states = tmp_timeline.events_by_type("state_change")
        assert len(states) == 2
        assert all(e.event == "state_change" for e in states)

    def test_events_since_datetime_filter(self, tmp_timeline):
        tmp_timeline.append("old", {"note": "before"})
        # All events are "now" so they should all be returned
        since = datetime.now(timezone.utc) - timedelta(minutes=1)
        events = tmp_timeline.events_since(since)
        assert len(events) >= 1

    def test_last_event_of_type(self, tmp_timeline):
        tmp_timeline.append("state_change", {"to": "active"})
        tmp_timeline.append("state_change", {"to": "drowsy"})

        last = tmp_timeline.last_event_of_type("state_change")
        assert last is not None
        assert last.data["to"] == "drowsy"

    def test_last_event_of_type_missing(self, tmp_timeline):
        assert tmp_timeline.last_event_of_type("nonexistent") is None

    def test_empty_store_returns_empty(self, tmp_path):
        store = TimelineStore(timeline_dir=str(tmp_path / "empty"))
        assert store.recent_events() == []
        assert store.events_by_type("state_change") == []
        assert store.last_event_of_type("test") is None
        assert store.state_duration_stats() == {}
        assert store.session_stats("test")["message_count"] == 0

    def test_state_duration_stats(self, tmp_timeline):
        # Manually write events with known timestamps
        now = datetime.now(timezone.utc)
        t1 = (now - timedelta(hours=2)).isoformat()
        t2 = (now - timedelta(hours=1)).isoformat()

        path = tmp_timeline._today_file()
        with open(path, "a") as f:
            f.write(json.dumps({"ts": t1, "event": "state_change", "data": {"from": "offline", "to": "active"}}) + "\n")
            f.write(json.dumps({"ts": t2, "event": "state_change", "data": {"from": "active", "to": "asleep"}}) + "\n")

        stats = tmp_timeline.state_duration_stats(hours=3)
        assert "active" in stats
        assert "asleep" in stats
        # Active was 1 hour = ~3600s
        assert 3500 < stats["active"] < 3700

    def test_session_stats(self, tmp_timeline):
        tmp_timeline.append("message", {"session_id": "sess1", "role": "user"})
        tmp_timeline.append("message", {"session_id": "sess1", "role": "assistant"})
        tmp_timeline.append("message", {"session_id": "sess2", "role": "user"})

        stats = tmp_timeline.session_stats("sess1")
        assert stats["message_count"] == 2
        assert stats["first_message"] is not None
        assert stats["last_message"] is not None

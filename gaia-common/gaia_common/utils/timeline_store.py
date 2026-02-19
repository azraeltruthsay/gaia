"""
Timeline Store — append-only JSONL event log for GAIA's temporal self-awareness.

Events are appended as single JSON lines to daily-rotated files:
    /shared/timeline/gaia_timeline_2026-02-18.jsonl

Each line: {"ts": "ISO8601", "event": "<type>", "data": {...}}

Event types:
    state_change   — sleep/wake state transitions
    session_start  — new conversation session created
    message        — user or assistant message (lightweight: no content)
    task_exec      — sleep task execution start/complete
    checkpoint     — prime.md checkpoint created
    gpu_handoff    — GPU ownership transfer
    code_evolution — code evolution snapshot generated

Query functions:
    recent_events(n)              — last N events across all types
    events_by_type(type, n)       — last N events of a specific type
    events_since(dt)              — all events after a datetime
    last_event_of_type(type)      — most recent event of a type
    state_duration_stats(hours)   — time-in-state statistics
    session_stats(session_id)     — message count + duration for a session
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GAIA.Timeline")


@dataclass
class TimelineEvent:
    """A single temporal event in GAIA's state timeline."""

    ts: str
    event: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_json_line(self) -> str:
        return json.dumps(
            {"ts": self.ts, "event": self.event, "data": self.data},
            ensure_ascii=False,
            default=str,
        )

    @classmethod
    def from_json_line(cls, line: str) -> Optional["TimelineEvent"]:
        try:
            obj = json.loads(line.strip())
            return cls(
                ts=obj.get("ts", ""),
                event=obj.get("event", ""),
                data=obj.get("data", {}),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    @property
    def timestamp(self) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(self.ts)
        except (ValueError, TypeError):
            return None


class TimelineStore:
    """Append-only JSONL event store with daily file rotation."""

    def __init__(self, timeline_dir: str = "/shared/timeline") -> None:
        self._dir = Path(timeline_dir)
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.debug("Cannot create timeline dir %s", self._dir)

    # ── Write API ────────────────────────────────────────────────────────

    def append(self, event_type: str, data: Dict[str, Any]) -> None:
        """Append a single event. Thread-safe via atomic line append."""
        event = TimelineEvent(
            ts=datetime.now(timezone.utc).isoformat(),
            event=event_type,
            data=data,
        )
        try:
            path = self._today_file()
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(event.to_json_line() + "\n")
        except OSError:
            logger.debug("Timeline append failed", exc_info=True)

    # ── Read API ─────────────────────────────────────────────────────────

    def recent_events(self, limit: int = 20) -> List[TimelineEvent]:
        """Last N events across today + yesterday (newest first)."""
        events = self._read_recent_files(max_days=2)
        return events[:limit]

    def events_by_type(
        self, event_type: str, limit: int = 10
    ) -> List[TimelineEvent]:
        """Last N events of a specific type."""
        all_events = self._read_recent_files(max_days=2)
        filtered = [e for e in all_events if e.event == event_type]
        return filtered[:limit]

    def events_since(
        self, since: datetime, limit: int = 100
    ) -> List[TimelineEvent]:
        """All events after a given datetime, up to limit (newest first)."""
        # Read enough days to cover the range
        now = datetime.now(timezone.utc)
        days = max(1, (now - since).days + 1)
        all_events = self._read_recent_files(max_days=min(days, 7))
        filtered = [
            e for e in all_events if e.timestamp and e.timestamp >= since
        ]
        return filtered[:limit]

    def last_event_of_type(self, event_type: str) -> Optional[TimelineEvent]:
        """Most recent event of the given type, or None."""
        results = self.events_by_type(event_type, limit=1)
        return results[0] if results else None

    def state_duration_stats(self, hours: int = 24) -> Dict[str, float]:
        """Seconds spent in each GaiaState over the last N hours.

        Returns e.g. {"active": 7200.0, "asleep": 3600.0, "drowsy": 120.0}
        """
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        changes = self.events_since(since, limit=500)
        # Filter to state_change events and reverse to chronological order
        changes = [e for e in reversed(changes) if e.event == "state_change"]

        stats: Dict[str, float] = {}
        if not changes:
            return stats

        now = datetime.now(timezone.utc)
        for i, event in enumerate(changes):
            state = event.data.get("to", "unknown")
            start = event.timestamp
            if start is None:
                continue
            if i + 1 < len(changes):
                end = changes[i + 1].timestamp
                if end is None:
                    end = now
            else:
                end = now
            duration = (end - start).total_seconds()
            stats[state] = stats.get(state, 0.0) + duration

        return stats

    def session_stats(self, session_id: str) -> Dict[str, Any]:
        """Message count, first/last message time for a session."""
        all_events = self._read_recent_files(max_days=7)
        messages = [
            e
            for e in reversed(all_events)
            if e.event == "message"
            and e.data.get("session_id") == session_id
        ]

        if not messages:
            return {
                "session_id": session_id,
                "message_count": 0,
                "first_message": None,
                "last_message": None,
            }

        return {
            "session_id": session_id,
            "message_count": len(messages),
            "first_message": messages[0].ts,
            "last_message": messages[-1].ts,
        }

    # ── Internal ─────────────────────────────────────────────────────────

    def _today_file(self) -> Path:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self._dir / f"gaia_timeline_{date_str}.jsonl"

    def _file_for_date(self, dt: datetime) -> Path:
        date_str = dt.strftime("%Y-%m-%d")
        return self._dir / f"gaia_timeline_{date_str}.jsonl"

    def _read_recent_files(self, max_days: int = 2) -> List[TimelineEvent]:
        """Read today + recent daily files, return events sorted newest first."""
        events: List[TimelineEvent] = []
        now = datetime.now(timezone.utc)

        for day_offset in range(max_days):
            dt = now - timedelta(days=day_offset)
            path = self._file_for_date(dt)
            if not path.exists():
                continue
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        event = TimelineEvent.from_json_line(line)
                        if event is not None:
                            events.append(event)
            except OSError:
                logger.debug("Cannot read timeline file %s", path)

        # Sort newest first
        events.sort(key=lambda e: e.ts, reverse=True)
        return events

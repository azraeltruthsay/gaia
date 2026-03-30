"""
Event Buffer — rolling episodic memory for GAIA.

A shared, append-only event log that gives GAIA truthful access to
recent system events. Injected into the world state section of her
system prompt so she can answer "what happened recently?" without
hallucinating.

Two views:
  - recent(n): Last N events as formatted text (for prompt injection)
  - full(hours): All events within a time window (for CFR analysis)

Any subsystem can append events: lifecycle transitions, conversations,
training runs, penpal cycles, doctor alerts, etc.

Storage: JSON lines file at /shared/event_buffer.jsonl
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("GAIA.EventBuffer")

BUFFER_PATH = Path(os.environ.get("SHARED_DIR", "/shared")) / "event_buffer.jsonl"
MAX_EVENTS = 500  # Rotate after this many events


class EventBuffer:
    """Thread-safe rolling event buffer with JSONL persistence."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, path: Path = BUFFER_PATH):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()

    @classmethod
    def instance(cls, path: Path = BUFFER_PATH) -> "EventBuffer":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(path)
        return cls._instance

    def append(self, event_type: str, summary: str,
               source: str = "", details: Optional[Dict] = None) -> None:
        """Append an event to the buffer.

        Args:
            event_type: Category (lifecycle, conversation, training, alert, penpal, etc.)
            summary: One-line human-readable summary
            source: Which subsystem generated this (lifecycle_machine, sleep_cycle, etc.)
            details: Optional structured data for CFR analysis
        """
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "summary": summary,
            "source": source,
        }
        if details:
            event["details"] = details

        with self._write_lock:
            try:
                with open(self._path, "a") as f:
                    f.write(json.dumps(event, default=str) + "\n")
            except Exception as e:
                logger.debug("Event buffer write failed: %s", e)

            # Rotate if too large
            self._maybe_rotate()

    def recent(self, n: int = 8) -> List[Dict]:
        """Return the last N events (newest first)."""
        events = self._read_all()
        return list(reversed(events[-n:]))

    def recent_formatted(self, n: int = 8) -> str:
        """Return last N events as formatted text for prompt injection."""
        events = self.recent(n)
        if not events:
            return "No recent events recorded."

        lines = []
        for e in events:
            ts = e.get("ts", "")
            # Parse and format as short time
            try:
                dt = datetime.fromisoformat(ts)
                time_str = dt.strftime("%H:%M")
            except Exception:
                time_str = ts[:16]

            etype = e.get("type", "")
            summary = e.get("summary", "")
            lines.append(f"- {time_str} [{etype}] {summary}")

        return "\n".join(lines)

    def full(self, hours: float = 24.0) -> List[Dict]:
        """Return all events within a time window (for CFR analysis)."""
        cutoff = time.time() - (hours * 3600)
        events = self._read_all()
        result = []
        for e in events:
            try:
                dt = datetime.fromisoformat(e["ts"])
                if dt.timestamp() >= cutoff:
                    result.append(e)
            except Exception:
                result.append(e)  # Include if can't parse timestamp
        return result

    def full_formatted(self, hours: float = 24.0) -> str:
        """Return all events in time window as formatted text (for CFR)."""
        events = self.full(hours)
        if not events:
            return "No events in the last {hours} hours."

        lines = []
        for e in events:
            ts = e.get("ts", "")
            try:
                dt = datetime.fromisoformat(ts)
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                time_str = ts

            etype = e.get("type", "")
            summary = e.get("summary", "")
            source = e.get("source", "")
            details = e.get("details", {})

            line = f"[{time_str}] ({etype}) {summary}"
            if source:
                line += f" [source: {source}]"
            if details:
                line += f"\n  {json.dumps(details, default=str)}"
            lines.append(line)

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear the buffer (for testing)."""
        with self._write_lock:
            try:
                self._path.write_text("")
            except Exception:
                pass

    def _read_all(self) -> List[Dict]:
        """Read all events from disk."""
        if not self._path.exists():
            return []
        try:
            events = []
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            return events
        except Exception:
            return []

    def _maybe_rotate(self) -> None:
        """Keep only the last MAX_EVENTS entries."""
        try:
            events = self._read_all()
            if len(events) > MAX_EVENTS:
                keep = events[-MAX_EVENTS:]
                with open(self._path, "w") as f:
                    for e in keep:
                        f.write(json.dumps(e, default=str) + "\n")
        except Exception:
            pass


# ── Convenience function for quick appends ────────────────────────────────────

def log_event(event_type: str, summary: str,
              source: str = "", details: Optional[Dict] = None) -> None:
    """Append an event to the global buffer. Safe to call from anywhere."""
    try:
        EventBuffer.instance().append(event_type, summary, source, details)
    except Exception:
        pass  # Never crash the caller

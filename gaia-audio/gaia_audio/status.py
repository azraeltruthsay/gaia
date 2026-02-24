"""Real-time status tracking — state machine + event ring buffer.

Provides the transparency layer: every audio operation emits an AudioEvent
that is stored in a ring buffer and pushed to connected WebSocket clients.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("GAIA.Audio.Status")

_MAX_EVENTS = 100


@dataclass
class AudioEvent:
    """A single audio processing event."""

    timestamp: str
    event_type: str  # stt_start, stt_complete, tts_start, tts_complete, gpu_swap, error, mute, unmute, ...
    detail: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "detail": self.detail,
            "latency_ms": self.latency_ms,
        }


class AudioStatusTracker:
    """Central status tracker for the audio service.

    Thread-safe (uses asyncio.Lock). Maintains a ring buffer of events
    and broadcasts to connected WebSocket listeners.
    """

    def __init__(self) -> None:
        self.state: str = "idle"  # idle | listening | transcribing | synthesizing | muted
        self.gpu_mode: str = "idle"  # idle | stt | tts
        self.stt_model: str | None = None
        self.tts_engine: str | None = None
        self.vram_used_mb: float = 0.0
        self.muted: bool = False
        self.last_transcription: str | None = None
        self.last_synthesis_text: str | None = None
        self.queue_depth: int = 0

        self._events: deque[AudioEvent] = deque(maxlen=_MAX_EVENTS)
        self._lock = asyncio.Lock()
        self._ws_clients: set[asyncio.Queue] = set()

        # Latency tracking for sparkline (last 20)
        self._stt_latencies: deque[float] = deque(maxlen=20)
        self._tts_latencies: deque[float] = deque(maxlen=20)

    async def emit(self, event_type: str, detail: str = "", latency_ms: float = 0.0) -> AudioEvent:
        """Record an event and broadcast to WebSocket listeners."""
        event = AudioEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            detail=detail,
            latency_ms=latency_ms,
        )
        async with self._lock:
            self._events.append(event)

            if event_type == "stt_complete" and latency_ms > 0:
                self._stt_latencies.append(latency_ms)
            elif event_type == "tts_complete" and latency_ms > 0:
                self._tts_latencies.append(latency_ms)

        # Broadcast to WebSocket clients (non-blocking)
        payload = event.to_dict()
        dead: set[asyncio.Queue] = set()
        for q in self._ws_clients:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                dead.add(q)
        if dead:
            self._ws_clients -= dead

        logger.debug("Audio event: %s — %s (%.1fms)", event_type, detail, latency_ms)
        return event

    def subscribe(self) -> asyncio.Queue:
        """Create a new WebSocket subscription queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._ws_clients.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove a WebSocket subscription."""
        self._ws_clients.discard(q)

    def snapshot(self) -> dict[str, Any]:
        """Return current status as a dict (for /status endpoint)."""
        return {
            "state": self.state,
            "gpu_mode": self.gpu_mode,
            "stt_model": self.stt_model,
            "tts_engine": self.tts_engine,
            "vram_used_mb": round(self.vram_used_mb, 1),
            "muted": self.muted,
            "last_transcription": self.last_transcription,
            "last_synthesis_text": self.last_synthesis_text,
            "queue_depth": self.queue_depth,
            "stt_latencies": list(self._stt_latencies),
            "tts_latencies": list(self._tts_latencies),
            "events": [e.to_dict() for e in self._events],
        }


# Module-level singleton
status_tracker = AudioStatusTracker()

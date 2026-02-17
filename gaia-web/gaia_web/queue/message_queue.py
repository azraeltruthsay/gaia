"""
Message queue for the sleep/wake cycle.

Messages arrive via Discord (or other sources) while GAIA is sleeping.
They are held here until gaia-core wakes and pulls them.  The first
enqueue triggers a wake signal to gaia-core.

Implementation: lightweight asyncio queue — no heavy broker needed at
this scale.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GAIA.MessageQueue")


@dataclass
class QueuedMessage:
    """A message waiting to be processed."""

    message_id: str
    content: str
    source: str  # "discord", "web", "cli"
    session_id: str
    priority: int = 0  # Higher = more urgent
    queued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


class MessageQueue:
    """Thread-safe async message queue for sleep/wake cycle."""

    def __init__(self, core_url: str = "http://gaia-core:6415") -> None:
        self._queue: List[QueuedMessage] = []
        self._lock = asyncio.Lock()
        self._wake_signal_sent = False
        self._core_url = core_url
        logger.info("MessageQueue initialized (core_url=%s)", core_url)

    async def enqueue(self, message: QueuedMessage) -> bool:
        """Add a message and send wake signal on first enqueue."""
        async with self._lock:
            self._queue.append(message)
            logger.info("Message queued: %s from %s", message.message_id, message.source)

            if not self._wake_signal_sent:
                await self._send_wake_signal()
                self._wake_signal_sent = True

            return True

    async def dequeue(self) -> Optional[QueuedMessage]:
        """Remove and return highest-priority message (FIFO within priority)."""
        async with self._lock:
            if not self._queue:
                return None
            self._queue.sort(key=lambda m: (-m.priority, m.queued_at))
            msg = self._queue.pop(0)
            logger.info("Message dequeued: %s", msg.message_id)
            if not self._queue:
                self._wake_signal_sent = False
            return msg

    async def peek(self) -> Optional[QueuedMessage]:
        async with self._lock:
            if not self._queue:
                return None
            self._queue.sort(key=lambda m: (-m.priority, m.queued_at))
            return self._queue[0]

    async def get_queue_status(self) -> Dict[str, Any]:
        async with self._lock:
            oldest_age = 0.0
            if self._queue:
                oldest_age = (datetime.now(timezone.utc) - self._queue[0].queued_at).total_seconds()
            return {
                "count": len(self._queue),
                "wake_signal_sent": self._wake_signal_sent,
                "oldest_message_age_seconds": oldest_age,
            }

    async def wait_for_active(self, poll_interval: float = 1.5, timeout: float = 120.0) -> bool:
        """Poll gaia-core /sleep/status until state is 'active' or timeout.

        Returns True if core reached active state, False on timeout or
        unresolvable states (offline, dreaming).
        """
        import httpx
        import time

        deadline = time.monotonic() + timeout
        unresolvable = {"offline", "dreaming"}

        while time.monotonic() < deadline:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"{self._core_url}/sleep/status", timeout=5.0
                    )
                    if resp.status_code == 200:
                        state = resp.json().get("state", "")
                        if state == "active":
                            logger.info("Core reached ACTIVE state")
                            return True
                        if state in unresolvable:
                            logger.warning("Core in unresolvable state: %s", state)
                            return False
                        logger.debug("Core state: %s — waiting...", state)
            except Exception:
                logger.debug("Poll failed — retrying", exc_info=True)

            await asyncio.sleep(poll_interval)

        logger.warning("wait_for_active timed out after %.0fs", timeout)
        return False

    async def _send_wake_signal(self) -> None:
        """POST to gaia-core /sleep/wake to trigger wakeup."""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.post(f"{self._core_url}/sleep/wake", timeout=5.0)
                if resp.status_code == 200:
                    logger.info("Wake signal sent to core")
                else:
                    logger.warning("Wake signal returned %d", resp.status_code)
        except Exception:
            logger.error("Wake signal failed", exc_info=True)

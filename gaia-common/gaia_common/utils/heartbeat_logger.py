"""
Heartbeat logger with log compression.

Instead of writing every heartbeat as a separate log line, this module
buffers repeated identical heartbeats and tracks a repeat count. The
heartbeat is only written to the log when:
  1. The heartbeat status/content changes
  2. A non-heartbeat log is written (flush before the other log)
  3. flush() is explicitly called (e.g., on shutdown)

Usage:
    from gaia_common.utils.heartbeat_logger import HeartbeatLogger

    hb_logger = HeartbeatLogger(logger)

    # In your heartbeat loop:
    hb_logger.heartbeat("healthy", {"memory": "1024MB", "cpu": "5%"})

    # When writing other logs, flush first:
    hb_logger.flush()
    logger.info("Some other event happened")

    # Or use the context manager for automatic flushing:
    with hb_logger.pause():
        logger.info("This will auto-flush before and after")

    # On shutdown:
    hb_logger.flush()
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Optional


class HeartbeatLogger:
    """
    Buffers repeated heartbeat logs and collapses them into a single entry
    with a repeat count.
    """

    def __init__(
        self,
        logger: logging.Logger,
        level: int = logging.INFO,
        include_timestamps: bool = True,
    ):
        """
        Initialize the heartbeat logger.

        Args:
            logger: The underlying logger to write to
            level: Log level for heartbeat messages (default: INFO)
            include_timestamps: Whether to include start/end timestamps in output
        """
        self._logger = logger
        self._level = level
        self._include_timestamps = include_timestamps
        self._lock = threading.Lock()

        # Buffered heartbeat state
        self._current_status: Optional[str] = None
        self._current_details: Optional[Dict[str, Any]] = None
        self._repeat_count: int = 0
        self._first_timestamp: Optional[datetime] = None
        self._last_timestamp: Optional[datetime] = None

    def _details_match(self, new_details: Optional[Dict[str, Any]]) -> bool:
        """Check if new details match the buffered details."""
        if self._current_details is None and new_details is None:
            return True
        if self._current_details is None or new_details is None:
            return False
        return self._current_details == new_details

    def _format_message(self, is_final: bool = False) -> str:
        """Format the heartbeat log message."""
        parts = [f"Heartbeat: {self._current_status}"]

        if self._current_details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self._current_details.items())
            parts.append(f"[{detail_str}]")

        if self._repeat_count > 0:
            parts.append(f"(repeated {self._repeat_count}x)")

        if self._include_timestamps and self._first_timestamp:
            if is_final and self._last_timestamp and self._repeat_count > 0:
                parts.append(
                    f"[{self._first_timestamp.isoformat()} - {self._last_timestamp.isoformat()}]"
                )
            elif self._repeat_count == 0:
                parts.append(f"[{self._first_timestamp.isoformat()}]")

        return " ".join(parts)

    def heartbeat(
        self,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a heartbeat. If it matches the previous heartbeat, increment
        the repeat counter. Otherwise, flush the previous heartbeat and
        start a new one.

        Args:
            status: The heartbeat status (e.g., "healthy", "degraded", "error")
            details: Optional dictionary of additional metrics/details
        """
        now = datetime.now(timezone.utc)

        with self._lock:
            # Check if this matches the current buffered heartbeat
            if self._current_status == status and self._details_match(details):
                # Same heartbeat - just increment counter
                self._repeat_count += 1
                self._last_timestamp = now
            else:
                # Different heartbeat - flush the old one first
                self._flush_locked()

                # Start buffering the new heartbeat
                self._current_status = status
                self._current_details = details.copy() if details else None
                self._repeat_count = 0
                self._first_timestamp = now
                self._last_timestamp = now

    def flush(self) -> None:
        """
        Flush any buffered heartbeat to the log.
        Call this before writing other log messages or on shutdown.
        """
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        """Internal flush (must be called with lock held)."""
        if self._current_status is not None:
            message = self._format_message(is_final=True)
            self._logger.log(self._level, message)

            # Clear the buffer
            self._current_status = None
            self._current_details = None
            self._repeat_count = 0
            self._first_timestamp = None
            self._last_timestamp = None

    @contextmanager
    def pause(self) -> Generator[None, None, None]:
        """
        Context manager that flushes before and after the block.
        Use this when you need to write other log messages.

        Example:
            with hb_logger.pause():
                logger.info("This message won't be mixed with heartbeats")
        """
        self.flush()
        try:
            yield
        finally:
            self.flush()

    @property
    def pending_count(self) -> int:
        """Return the current repeat count of the buffered heartbeat."""
        with self._lock:
            return self._repeat_count if self._current_status else 0

    @property
    def current_status(self) -> Optional[str]:
        """Return the current buffered heartbeat status."""
        with self._lock:
            return self._current_status


class HeartbeatLoggerProxy(logging.Handler):
    """
    A logging handler that intercepts heartbeat-pattern logs and routes
    them through a HeartbeatLogger for compression.

    This can be used to retrofit existing code that logs heartbeats directly
    without modifying the logging calls.

    Usage:
        logger = logging.getLogger("MyService")
        hb_proxy = HeartbeatLoggerProxy(
            heartbeat_logger=HeartbeatLogger(logger),
            pattern="Heartbeat:",  # Lines starting with this get compressed
        )
        logger.addHandler(hb_proxy)
    """

    def __init__(
        self,
        heartbeat_logger: HeartbeatLogger,
        pattern: str = "Heartbeat:",
    ):
        super().__init__()
        self._hb_logger = heartbeat_logger
        self._pattern = pattern

    def emit(self, record: logging.LogRecord) -> None:
        """Handle a log record, routing heartbeats to the compressor."""
        msg = self.format(record) if self.formatter else record.getMessage()

        if msg.startswith(self._pattern):
            # Extract status from the message
            status_part = msg[len(self._pattern):].strip()
            self._hb_logger.heartbeat(status_part)
        else:
            # Non-heartbeat log - flush and let it pass through
            self._hb_logger.flush()


__all__ = ["HeartbeatLogger", "HeartbeatLoggerProxy"]

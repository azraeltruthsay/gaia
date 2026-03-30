"""
Centralized logging setup for GAIA services.

This module provides a helper to configure logging with UTC timestamps
and a consistent formatter so all services inherit the same behavior.

Usage:
    from gaia_common.utils import setup_logging, get_logger

    setup_logging(log_dir="/var/log/gaia", level=logging.INFO)
    logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


class UTCFormatter(logging.Formatter):
    """Custom formatter that uses UTC timestamps in ISO format."""

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()


class HealthCheckFilter(logging.Filter):
    """
    Filter that suppresses health check access log spam.

    Uvicorn logs every HTTP request at INFO level, including repeated
    /health endpoint calls. This filter drops those log records to keep
    logs clean while preserving all other access logs.
    """

    def __init__(self, endpoints: Optional[List[str]] = None):
        """
        Args:
            endpoints: List of endpoint paths to filter out.
                       Defaults to ["/health", "/healthz", "/ready", "/readiness", "/live", "/liveness"].
        """
        super().__init__()
        self.endpoints = endpoints or [
            "/health",
            "/healthz",
            "/ready",
            "/readiness",
            "/live",
            "/liveness",
        ]

    def filter(self, record: logging.LogRecord) -> bool:
        # Check if this is a uvicorn access log for a health endpoint
        message = record.getMessage()
        for endpoint in self.endpoints:
            # Uvicorn access log format: '127.0.0.1:port - "GET /health HTTP/1.1" 200 OK'
            if f'"{endpoint} ' in message or f'" {endpoint} ' in message or f'GET {endpoint} ' in message:
                return False
        return True


class LevelRingHandler(logging.Handler):
    """Routes log records to per-level FIFO ring-buffer files.

    Each severity level gets its own file under ``{log_dir}/{service_name}/``.
    When the line count exceeds the configured max for a level, the oldest
    lines are discarded (ring-buffer / FIFO semantics).  Disk writes are
    batched: the buffer is flushed every ``_flush_every`` writes to amortise
    I/O overhead.
    """

    DEFAULT_LIMITS: Dict[str, int] = {
        "DEBUG": 200,
        "INFO": 500,
        "WARNING": 500,
        "ERROR": 1000,
        "CRITICAL": 1000,
    }

    def __init__(
        self,
        log_dir: str,
        service_name: str,
        limits: Optional[Dict[str, int]] = None,
        flush_every: int = 50,
    ):
        super().__init__()
        self.log_dir = Path(log_dir) / service_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.limits: Dict[str, int] = {**self.DEFAULT_LIMITS, **(limits or {})}
        self._buffers: Dict[str, deque] = {}
        self._files: Dict[str, Path] = {}
        self._write_counts: Dict[str, int] = {}
        self._flush_every = flush_every

        for level_name, max_lines in self.limits.items():
            file_path = self.log_dir / f"{level_name.lower()}.log"
            self._files[level_name] = file_path
            # Seed buffer from any existing file content
            existing: List[str] = []
            if file_path.exists():
                try:
                    existing = file_path.read_text(errors="replace").splitlines()
                except OSError:
                    pass
            self._buffers[level_name] = deque(existing, maxlen=max_lines)
            self._write_counts[level_name] = 0

    def emit(self, record: logging.LogRecord) -> None:
        level = record.levelname
        if level not in self._buffers:
            level = "INFO"  # fallback for custom levels
        try:
            line = self.format(record)
            self._buffers[level].append(line)
            self._write_counts[level] += 1
            if self._write_counts[level] >= self._flush_every:
                self._flush_level(level)
                self._write_counts[level] = 0
        except Exception:
            self.handleError(record)

    def _flush_level(self, level: str) -> None:
        """Atomic rewrite of the ring buffer to disk."""
        path = self._files[level]
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text("\n".join(self._buffers[level]) + "\n", errors="replace")
            tmp.rename(path)
        except OSError:
            pass  # best-effort — don't crash the service

    def flush(self) -> None:
        for level in self._buffers:
            self._flush_level(level)

    def close(self) -> None:
        self.flush()
        super().close()


def install_health_check_filter(endpoints: Optional[List[str]] = None) -> None:
    """
    Install the health check filter on uvicorn's access logger.

    Call this at application startup to suppress health check log spam.

    Args:
        endpoints: Optional list of endpoints to filter. Uses defaults if not provided.
    """
    health_filter = HealthCheckFilter(endpoints)

    # Apply to uvicorn's access logger
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.addFilter(health_filter)

    # Also apply to root logger in case access logs go there
    logging.getLogger().addFilter(health_filter)


def setup_logging(
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    handlers: Optional[List[logging.Handler]] = None,
    service_name: Optional[str] = None,
    level_ring: bool = True,
    ring_limits: Optional[Dict[str, int]] = None,
    max_main_log_bytes: int = 10 * 1024 * 1024,
    main_log_backup_count: int = 3,
) -> None:
    """
    Configure logging with UTC timestamps and consistent formatting.

    Args:
        log_dir: Directory for log files. If None, only console logging is used.
        level: Logging level (default: INFO)
        handlers: Additional handlers to add
        service_name: Name of the service for log file naming
        level_ring: Enable per-level FIFO ring-buffer files (default: True)
        ring_limits: Override per-level line limits (e.g. {"DEBUG": 500})
        max_main_log_bytes: Max size of the main rotating log file (default: 10MB)
        main_log_backup_count: Number of rotated backup files to keep (default: 3)
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Build format string with optional service name
    if service_name:
        fmt = f"%(asctime)s [{service_name}] %(levelname)s:%(name)s:%(message)s"
    else:
        fmt = "%(asctime)s %(levelname)s:%(name)s:%(message)s"

    formatter = UTCFormatter(fmt)

    # Make the UTC formatter the default
    try:
        logging._defaultFormatter = formatter  # type: ignore[attr-defined]
    except Exception:
        pass

    # Remove existing handlers to avoid double-logging
    for h in list(root.handlers):
        root.removeHandler(h)

    # Console handler
    stream_h = logging.StreamHandler()
    stream_h.setFormatter(formatter)
    stream_h.setLevel(level)
    root.addHandler(stream_h)

    # File handler (if log_dir specified) — now uses RotatingFileHandler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_filename = f"{service_name}.log" if service_name else "gaia.log"
        file_h = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, log_filename),
            maxBytes=max_main_log_bytes,
            backupCount=main_log_backup_count,
            encoding="utf-8",
        )
        file_h.setFormatter(formatter)
        file_h.setLevel(level)
        root.addHandler(file_h)

        # Per-level ring-buffer handler
        if level_ring and service_name:
            # Allow env-var overrides for ring limits (e.g. LOG_RING_DEBUG=500)
            resolved_limits = dict(ring_limits or {})
            for level_name in LevelRingHandler.DEFAULT_LIMITS:
                env_key = f"LOG_RING_{level_name}"
                env_val = os.getenv(env_key)
                if env_val and level_name not in resolved_limits:
                    try:
                        resolved_limits[level_name] = int(env_val)
                    except ValueError:
                        pass

            ring_h = LevelRingHandler(
                log_dir=log_dir,
                service_name=service_name,
                limits=resolved_limits if resolved_limits else None,
            )
            ring_h.setFormatter(formatter)
            # Ring handler accepts ALL levels — it routes internally
            ring_h.setLevel(logging.DEBUG)
            root.addHandler(ring_h)

    # Additional handlers
    if handlers:
        for h in handlers:
            h.setFormatter(formatter)
            root.addHandler(h)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    This is a convenience wrapper around logging.getLogger that ensures
    the logging system is configured (at least with defaults).

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


__all__ = [
    "setup_logging",
    "get_logger",
    "UTCFormatter",
    "HealthCheckFilter",
    "LevelRingHandler",
    "install_health_check_filter",
]

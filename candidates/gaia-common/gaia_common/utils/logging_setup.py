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
import os
from datetime import datetime, timezone
from typing import List, Optional


class UTCFormatter(logging.Formatter):
    """Custom formatter that uses UTC timestamps in ISO format."""

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()


def setup_logging(
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    handlers: Optional[List[logging.Handler]] = None,
    service_name: Optional[str] = None,
) -> None:
    """
    Configure logging with UTC timestamps and consistent formatting.

    Args:
        log_dir: Directory for log files. If None, only console logging is used.
        level: Logging level (default: INFO)
        handlers: Additional handlers to add
        service_name: Name of the service for log file naming
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

    # File handler (if log_dir specified)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_filename = f"{service_name}.log" if service_name else "gaia.log"
        file_h = logging.FileHandler(
            os.path.join(log_dir, log_filename),
            encoding="utf-8"
        )
        file_h.setFormatter(formatter)
        file_h.setLevel(level)
        root.addHandler(file_h)

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


__all__ = ["setup_logging", "get_logger", "UTCFormatter"]

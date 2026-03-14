"""Structured JSON log formatter for GAIA services.

Outputs one JSON object per log line for Logstash/ELK ingestion.

Usage:
    import logging
    from gaia_common.utils.json_formatter import setup_json_logging

    setup_json_logging(service="core", level=logging.INFO)
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for ELK ingestion."""

    def __init__(self, service: str = "unknown"):
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Include extra fields if attached to the record
        for key in (
            "request_id", "user_id", "channel_id", "session_id", "duration_ms",
            "error_code", "error_hint", "error_category",
        ):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val

        return json.dumps(entry, default=str, ensure_ascii=False)


def setup_json_logging(
    service: str | None = None,
    level: int | None = None,
) -> None:
    """Configure root logger to output structured JSON.

    Args:
        service: Service name (defaults to GAIA_SERVICE env var).
        level: Log level (defaults to LOG_LEVEL env var or INFO).
    """
    service = service or os.environ.get("GAIA_SERVICE", "unknown")
    if level is None:
        level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter(service=service))

    root = logging.getLogger()
    root.setLevel(level)
    # Replace existing handlers
    root.handlers = [handler]

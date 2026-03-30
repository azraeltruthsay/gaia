"""Structured error logging helper for the GAIA SOA.

Provides ``log_gaia_error()`` — a single call that looks up the error
definition, logs at the registered level, and attaches ``error_code``,
``error_hint``, and ``error_category`` as log-record extras so that
:class:`~gaia_common.utils.json_formatter.JSONFormatter` can include them
in the JSON output automatically.

Stdlib-only — safe for gaia-doctor.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from gaia_common.errors import lookup


def log_gaia_error(
    logger: logging.Logger,
    error_code: str,
    detail: str = "",
    *,
    exc_info: Any = None,
    level_override: Optional[int] = None,
    **extra: Any,
) -> None:
    """Log a structured GAIA error.

    Parameters
    ----------
    logger:
        The logger instance to write to.
    error_code:
        A ``GAIA-{SERVICE}-{NNN}`` code from the error registry.
    detail:
        Free-text context about this specific occurrence.
    exc_info:
        Exception info tuple, ``True``, or ``None`` (passed to ``logger.log``).
    level_override:
        Override the level from the registry (rarely needed).
    **extra:
        Additional key/value pairs attached to the log record.
    """
    defn = lookup(error_code)

    if defn is not None:
        level = level_override if level_override is not None else defn.level
        message = f"[{error_code}] {defn.message}"
        if detail:
            message += f" — {detail}"
        hint = defn.hint
        category = defn.category.value
    else:
        # Unknown code — still log, just without enrichment
        level = level_override if level_override is not None else logging.ERROR
        message = f"[{error_code}] (unregistered)"
        if detail:
            message += f" — {detail}"
        hint = ""
        category = "unknown"

    # Build extra dict for the log record
    log_extra = {
        "error_code": error_code,
        "error_hint": hint,
        "error_category": category,
        **extra,
    }

    logger.log(level, message, exc_info=exc_info, extra=log_extra)

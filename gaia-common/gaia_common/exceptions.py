"""Exception hierarchy for the GAIA SOA.

Every exception carries an ``error_code`` that maps to the error registry
(:mod:`gaia_common.errors`), a free-text ``detail`` string, and an optional
``context`` dict for structured metadata.

Hierarchy::

    GaiaError
    ├── GaiaConfigError    — config/env issues
    ├── GaiaModelError     — model load/inference
    ├── GaiaSafetyError    — guardian/sentinel/identity
    ├── GaiaToolError      — tool execution
    └── GaiaNetworkError   — inter-service comms

Stdlib-only — safe for gaia-doctor.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from gaia_common.errors import lookup


class GaiaError(Exception):
    """Base exception for all GAIA structured errors."""

    def __init__(
        self,
        error_code: str,
        detail: str = "",
        context: Optional[Dict[str, Any]] = None,
    ):
        self.error_code = error_code
        self.detail = detail
        self.context = context or {}

        # Auto-lookup hint from registry
        defn = lookup(error_code)
        self.hint = defn.hint if defn else ""
        self.message = defn.message if defn else error_code

        # Build human-readable message
        parts = [f"[{error_code}]"]
        if defn:
            parts.append(defn.message)
        if detail:
            parts.append(f"— {detail}")
        super().__init__(" ".join(parts))

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for JSON responses."""
        d: Dict[str, Any] = {
            "error_code": self.error_code,
            "message": self.message,
            "detail": self.detail,
        }
        if self.hint:
            d["hint"] = self.hint
        if self.context:
            d["context"] = self.context
        return d


class GaiaConfigError(GaiaError):
    """Configuration or environment issue."""


class GaiaModelError(GaiaError):
    """Model loading, inference, or routing failure."""


class GaiaToolError(GaiaError):
    """Tool execution failure."""


class GaiaSafetyError(GaiaError):
    """Safety, guardian, or identity violation."""


class GaiaNetworkError(GaiaError):
    """Inter-service communication failure."""

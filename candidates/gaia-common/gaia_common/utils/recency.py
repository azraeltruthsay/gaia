"""Recency-decay kernel for the World Model (GAIA_Project-lw4).

Applies time-based decay to a stored confidence value. Each fact_type
has its own half-life (see `fact_types.HALFLIFE`); the decay function
is selectable (exponential by default, linear or step as alternatives).

The kernel is intentionally pure — no DB, no logging side effects. It
takes (confidence, age_seconds, fact_type) and returns a float in [0, 1].

Math:

  - exponential: relevance = conf × 0.5 ^ (age / halflife)
       smooth, half-life feel. Default. Never reaches 0.

  - linear: relevance = conf × max(0, 1 - age / (2 × halflife))
       at age=halflife → 0.5, at age=2×halflife → 0. Cuts to 0 hard.

  - step: relevance = conf if age < halflife else conf × 0.5 ^ ((age-hl)/hl)
       a grace period then exponential. Useful when "fresh" should not
       penalize at all.

If fact_type has halflife=None (biographical, etc.), relevance == conf
regardless of age. The age argument is clamped to ≥ 0.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional, Union

from gaia_common.utils.fact_types import halflife_seconds, BIOGRAPHICAL


DecayKind = str  # "exponential" | "linear" | "step"


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 string into a tz-aware datetime. None on failure."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def age_seconds(
    valid_from: Union[str, datetime, None],
    *,
    now: Optional[datetime] = None,
) -> Optional[float]:
    """Compute age in seconds from valid_from to now.

    `valid_from` may be an ISO-8601 string or a datetime. Returns None
    if valid_from is missing or unparseable, else max(0, now - valid_from).
    """
    if valid_from is None:
        return None
    if isinstance(valid_from, str):
        vf = _parse_iso(valid_from)
    else:
        vf = valid_from if valid_from.tzinfo else valid_from.replace(tzinfo=timezone.utc)
    if vf is None:
        return None
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return max(0.0, (now - vf).total_seconds())


def decay(
    age: float,
    fact_type: Optional[str],
    *,
    kind: DecayKind = "exponential",
) -> float:
    """Return decay multiplier in [0, 1] for a given age and fact_type."""
    hl = halflife_seconds(fact_type)
    if hl is None or fact_type == BIOGRAPHICAL:
        # No-decay class — always 1.0.
        return 1.0
    if hl <= 0.0:
        return 0.0
    a = max(0.0, float(age))
    if kind == "exponential":
        return math.pow(0.5, a / hl)
    if kind == "linear":
        return max(0.0, 1.0 - a / (2.0 * hl))
    if kind == "step":
        if a < hl:
            return 1.0
        return math.pow(0.5, (a - hl) / hl)
    raise ValueError(f"Unknown decay kind: {kind!r}")


def decayed_relevance(
    confidence: float,
    valid_from: Union[str, datetime, None],
    fact_type: Optional[str],
    *,
    now: Optional[datetime] = None,
    kind: DecayKind = "exponential",
) -> float:
    """Compute confidence × decay(age, fact_type) for a stored triple.

    If valid_from is missing/unparseable, the confidence is returned
    unmodified (no age → no decay applied). If fact_type's halflife is
    None (e.g. BIOGRAPHICAL), the confidence is also returned unmodified.

    Output is clamped to [0, 1] for safety.
    """
    conf = max(0.0, min(1.0, float(confidence)))
    a = age_seconds(valid_from, now=now)
    if a is None:
        return conf
    d = decay(a, fact_type, kind=kind)
    return max(0.0, min(1.0, conf * d))

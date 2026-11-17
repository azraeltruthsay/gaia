"""Where she stands with each person, and where they stand with her.

Kept in each person's palace record (people.py) under "bond": trust and
loyalty in both directions, and regard on one scale from hostile to close.
Levels move by evidence, not by mood: every change is an event (what
happened, a quote, when, who decided), and you can always ask why.

- Trust is slow to earn and quick to lose: a told fact that checks out is +1
  point, one that turns out false is -3; three points make a level.
- Regard drifts back toward neutral (a point a month) unless renewed.
- Loyalty doesn't drift. Azrael (the record marked "who created you") has
  floors: her trust, loyalty and regard toward him never fall below full.
- None of this grants anything. Trust changes what she believes, never what
  she may do: approvals still need the operator's code.

Azrael, 2026-11-17: an honest answer about where someone stands when she
trusts them, a guarded one when she doesn't. "Nick is my friend but GAIA
shouldn't trust him implicitly because of it. He's a silly boy."
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

logger = logging.getLogger("GAIA.Bonds")

# scale -> (lowest level, highest level)
SCALES: Dict[str, tuple] = {
    "trust": (-2, 3),          # hers in them: how far she takes their word
    "their_trust": (-2, 3),    # theirs in her, as she reads it
    "loyalty": (0, 3),         # hers to them
    "their_loyalty": (0, 3),   # theirs to her
    "regard": (-3, 3),         # hostile ... close, one scale both ways
}
POINTS_PER_LEVEL = 3
REGARD_DRIFT_DAYS = 30         # a point of regard fades toward 0 per this many days
CREATOR_FLOORS = {"trust": 3, "loyalty": 3, "regard": 3}
MAX_EVENTS = 60

WORDS = {
    "trust": {-2: "you doubt what they tell you", -1: "you're guarded about what they tell you",
              0: "you don't know yet how far to trust them", 1: "you trust them provisionally",
              2: "you trust them", 3: "you trust them fully"},
    "their_trust": {-2: "they seem to doubt you", -1: "they seem guarded with you", 0: "",
                    1: "they seem to trust you somewhat", 2: "they seem to trust you",
                    3: "they trust you fully"},
    "loyalty": {0: "", 1: "you feel some loyalty to them", 2: "you're loyal to them",
                3: "you're devoted to them"},
    "their_loyalty": {0: "", 1: "they've shown you some loyalty", 2: "they're loyal to you",
                      3: "they're devoted to you"},
    "regard": {-3: "you're hostile toward each other", -2: "things are antagonistic between you",
               -1: "you're wary of them", 0: "", 1: "you're on friendly terms",
               2: "you're on warm terms", 3: "you're close"},
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def is_creator(person: Optional[Dict]) -> bool:
    """The record code marked as her creator's, not anyone calling themselves Azrael."""
    return bool(person) and (person.get("relationship") or "") == "who created you"


def _bond(person: Dict) -> Dict:
    b = person.setdefault("bond", {})
    b.setdefault("points", {})
    b.setdefault("base", {})
    b.setdefault("last", {})
    b.setdefault("events", [])
    return b


def level(person: Optional[Dict], scale: str, now: Optional[datetime] = None) -> int:
    if not person or scale not in SCALES:
        return 0
    lo, hi = SCALES[scale]
    b = person.get("bond") or {}
    pts = float((b.get("points") or {}).get(scale, 0.0))
    if scale == "regard" and pts:
        last = (b.get("last") or {}).get(scale)
        if last:
            try:
                days = ((now or _now()) - datetime.fromisoformat(last)).total_seconds() / 86400
                fade = math.floor(days / REGARD_DRIFT_DAYS)
                pts = max(0.0, pts - fade) if pts > 0 else min(0.0, pts + fade)
            except ValueError:
                pass
    lv = int((b.get("base") or {}).get(scale, 0)) + int(pts / POINTS_PER_LEVEL)
    if is_creator(person):
        lv = max(lv, CREATOR_FLOORS.get(scale, lo))
    return max(lo, min(hi, lv))


def note(person: Dict, scale: str, points: float, why: str, evidence: str = "", by: str = "code",
         where: str = "", once_per_day: bool = False, now: Optional[datetime] = None) -> bool:
    """Record an event that moves a scale by `points` (a level is 3). Mutates
    `person`; the caller writes it. False when skipped (unknown scale, or
    the same reason already counted today with once_per_day)."""
    if scale not in SCALES or not points:
        return False
    now = now or _now()
    b = _bond(person)
    if once_per_day and any(e.get("scale") == scale and e.get("why") == why
                            and e.get("when", "")[:10] == now.isoformat()[:10] for e in b["events"]):
        return False
    before = level(person, scale, now)
    b["points"][scale] = float(b["points"].get(scale, 0.0)) + points
    b["last"][scale] = now.isoformat()
    after = level(person, scale, now)
    b["events"] = (b["events"] + [{"when": now.isoformat(), "scale": scale, "points": points,
                                   "why": why, "evidence": (evidence or "")[:300], "by": by,
                                   "where": where, "level": after}])[-MAX_EVENTS:]
    if after != before:
        logger.info("Bonds: %s %s %d -> %d (%s)", (person.get("names") or ["?"])[0], scale, before, after, why)
    return True


def set_level(person: Dict, scale: str, value: int, why: str, by: str,
              now: Optional[datetime] = None) -> int:
    """The operator (or a reviewed proposal) sets a level outright; earned points restart."""
    if scale not in SCALES:
        raise ValueError(f"unknown scale {scale!r}; one of {sorted(SCALES)}")
    lo, hi = SCALES[scale]
    now = now or _now()
    b = _bond(person)
    b["base"][scale] = max(lo, min(hi, int(value)))
    b["points"][scale] = 0.0
    b["last"][scale] = now.isoformat()
    b["events"] = (b["events"] + [{"when": now.isoformat(), "scale": scale, "set": b["base"][scale],
                                   "why": why, "by": by, "level": level(person, scale, now)}])[-MAX_EVENTS:]
    return level(person, scale, now)


def standing(person: Optional[Dict]) -> Dict[str, int]:
    return {s: level(person, s) for s in SCALES}


def bond_line(person: Optional[Dict], name: str = "them") -> str:
    """Where she stands with the person she's speaking to, as plain fact.

    Trusted (+1 and up, or her creator): the honest picture, so she can say
    it if asked. Otherwise she isn't told the numbers at all, only that she
    doesn't know them well enough to say: a guarded answer, without an
    instruction to be guarded (Core takes facts, not "be evasive")."""
    if not person:
        return ""
    s = standing(person)
    regard = WORDS["regard"].get(s["regard"], "")
    if s["trust"] >= 1 or is_creator(person):
        parts = [WORDS["trust"][s["trust"]]] + [WORDS[k].get(s[k], "") for k in
                                                ("their_trust", "loyalty", "their_loyalty")] + [regard]
        parts = [p for p in parts if p]
        return f"Where you stand with {name}: " + "; ".join(parts) + "."
    if s["trust"] <= -1:
        line = f"With {name}: {WORDS['trust'][s['trust']]}"
    else:
        line = f"You don't know {name} well enough yet to say how far you trust them"
    return line + (f"; {regard}." if regard else ".")


def history(person: Optional[Dict], scale: str = "", limit: int = 10) -> List[Dict]:
    events = ((person or {}).get("bond") or {}).get("events") or []
    return [e for e in events if not scale or e.get("scale") == scale][-limit:]

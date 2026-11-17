"""What a message says about how someone regards her: warmth or hostility.

Moves the regard scale in the speaker's palace record (gaia_common bonds.py),
conservatively: thanks and appreciation +1 point (once a day), a direct
insult or "shut up" at her -3 points (one level). Teasing isn't hostility —
Nick joking he'd eat her for dinner moves nothing. Who is speaking comes from
the transport (the Discord account, the session), never from the words.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger("GAIA.RegardSignals")

_WARM_RE = re.compile(
    r"\b(thank(s| you)|thx|ty|appreciate (it|you|that)|you'?re (the best|great|awesome|amazing|wonderful)"
    r"|love (you|ya|that you)|good job|well done|nice work|proud of you)\b", re.I)
_HOSTILE_RE = re.compile(
    r"\b(you'?re|you are|ur)\s+(so\s+|such\s+an?\s+|an?\s+|really\s+)?"
    r"(stupid|useless|worthless|dumb|an idiot|idiot|pathetic|garbage|trash|a joke|broken junk)\b"
    r"|\b(shut up|f+u+c*k+ (you|off)|screw you|i hate you|go to hell|piece of (shit|crap))\b", re.I)


def classify(text: str) -> str:
    """'hostile', 'warm' or '' for one message."""
    t = text or ""
    if _HOSTILE_RE.search(t):
        return "hostile"
    if _WARM_RE.search(t):
        return "warm"
    return ""


def speaker_person(session_id: str, metadata: Optional[dict]):
    from gaia_common.utils.people import PeopleRegistry, is_channel, person_for_discord
    registry = PeopleRegistry()
    meta = metadata or {}
    if is_channel(session_id) or meta.get("channel_id") and not meta.get("is_dm"):
        uid = meta.get("user_id")
        return registry, (person_for_discord(str(uid), registry) if uid else None)
    return registry, registry.for_session(session_id)


def note_regard(session_id: str, metadata: Optional[dict], text: str) -> str:
    """Record warmth or hostility from this message; returns what was noted."""
    kind = classify(text)
    if not kind:
        return ""
    try:
        registry, person = speaker_person(session_id, metadata)
        if not person:
            return ""
        if kind == "warm":
            registry.bond_note(person["id"], "regard", 1, "was warm to her", evidence=text[:200],
                               by="message", where=session_id, once_per_day=True)
        else:
            registry.bond_note(person["id"], "regard", -3, "was hostile to her", evidence=text[:200],
                               by="message", where=session_id, once_per_day=True)
        return kind
    except Exception:
        logger.debug("regard note failed", exc_info=True)
        return ""

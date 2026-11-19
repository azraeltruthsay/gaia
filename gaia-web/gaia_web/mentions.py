"""Her Discord replies @mention the people she speaks to by name.

Only a name used to address someone becomes a mention: "Nick, ...",
"... what do you think, Andy?", "@Nick". A name in passing ("Azrael's friend")
stays text, so talking about someone never pings them. The people are those the
registry has seen in this channel (gaia_common.utils.people), so "Nick" finds
the account Nick writes from even when its display name is schwiftyboi.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional


def channel_people(people: Iterable[Dict], session_id: str) -> Dict[str, str]:
    """name -> Discord account id, for people known in this channel. A name two
    people answer to is left out: a wrong ping is worse than none."""
    found: Dict[str, set] = {}
    for p in people:
        ids = [a.get("id") for a in p.get("accounts", [])
               if a.get("kind") == "discord" and str(a.get("id") or "").isdigit()]
        if len(ids) != 1:
            continue
        here = any(session_id in str(a.get("source", "")) for a in p.get("accounts", [])) or \
            any(session_id in str(v) for v in (p.get("name_sources") or {}).values())
        if not here:
            continue
        for name in p.get("names", []):
            if len(name.strip()) >= 2:
                found.setdefault(name.strip(), set()).add(ids[0])
    return {n: next(iter(i)) for n, i in found.items() if len(i) == 1}


def _patterns(name: str) -> List[re.Pattern]:
    n = re.escape(name)
    return [
        re.compile(rf"@{n}\b"),                                   # @Nick
        re.compile(rf"(?:^|(?<=[.!?]\s)|(?<=\n)){n}(?=,)"),      # Nick, ...
        re.compile(rf"(?<=,\s){n}(?=\s*(?:[.!?]|$))", re.M),      # ..., Nick?
    ]


def mention_names(text: str, names: Dict[str, str], skip_ids: Optional[Iterable[str]] = None) -> str:
    """Each addressed name becomes <@id>, the first time only."""
    skip = {str(s) for s in (skip_ids or [])}
    for name in sorted(names, key=len, reverse=True):
        uid = names[name]
        if uid in skip or f"<@{uid}>" in text:
            continue
        for pat in _patterns(name):
            text, n = pat.subn(f"<@{uid}>", text, count=1)
            if n:
                break
    return text

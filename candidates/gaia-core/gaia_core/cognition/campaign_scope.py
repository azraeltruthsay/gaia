"""Who may see the DM's side of the campaign (the Kanka worldbuild tool).

Azrael, 2026-11-17: she can read and propose changes to the campaign wiki,
every change approved with his authenticator code. Most of it is DM-only
(87 of 90 entries) and players are in the Grotto, so DM-only entries are
open only in a conversation that is his alone: a Discord DM or a web session
linked to him (people.py). Never in a channel, even when he is the one
asking: everyone there reads the reply. Decided here from the turn's
session, never from what the model wrote: any _dm_view in a tool call is
replaced.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger("GAIA.CampaignScope")

SCOPED_METHODS = ("worldbuild",)


def is_scoped(method: str) -> bool:
    return method in SCOPED_METHODS or (method or "").startswith("kanka_")


def dm_view(session_id: str) -> bool:
    if not session_id:
        return False
    try:
        from gaia_common.utils.bonds import is_creator
        from gaia_common.utils.people import PeopleRegistry, is_channel
        if is_channel(session_id):
            return False
        return is_creator(PeopleRegistry().for_session(session_id))
    except Exception:
        logger.debug("campaign scope: lookup failed; DM view off", exc_info=True)
        return False


def scope(method: str, params: Optional[Dict], session_id: Optional[str] = None) -> Dict:
    """params with _dm_view set by code for campaign calls (others untouched)."""
    out = dict(params or {})
    if not is_scoped(method):
        return out
    out.pop("_dm_view", None)
    if session_id is None:
        from gaia_core.models.turn_context import turn_ids
        session_id = turn_ids().get("session_id", "")
    out["_dm_view"] = dm_view(session_id)
    return out


# ── The campaign's own names stay off the web ────────────────────────────
# 2026-11-17: asked "look up Leshen in the Kanka campaign", grounding searched
# the web first and found the Witcher's leshen; with no campaign entry she
# could see, she called it "an antlered forest guardian". A name that is an
# entry in the campaign is theirs, like a name they introduce (own_names).
# gaia-mcp sends hashes of the names, never the names: DM-only names are
# spoilers too, and a match needs the name already in the message.

import hashlib
import re
import threading
import time

NAME_TTL_S = 1800
_names_lock = threading.Lock()
_names_cache: Dict = {"at": 0.0, "hashes": frozenset(), "max_words": 1}
_CAMPAIGN_WORDS_RE = re.compile(r"\bkanka\b", re.I)


def _name_hash(name: str) -> str:
    return hashlib.sha256(" ".join((name or "").lower().split()).encode()).hexdigest()


def _hashes() -> Dict:
    with _names_lock:
        if time.time() - _names_cache["at"] < NAME_TTL_S:
            return _names_cache
    try:
        from gaia_core.utils import mcp_client
        r = mcp_client.call_jsonrpc("kanka_name_hashes", {}, timeout=20)
        res = (r.get("response") or {}).get("result", {}) if r.get("ok") else {}
        hashes = frozenset(res.get("hashes") or ())
        if hashes:
            with _names_lock:
                _names_cache.update(at=time.time(), hashes=hashes, max_words=int(res.get("max_words") or 1))
        else:
            with _names_lock:   # try again in a minute, not every turn
                _names_cache["at"] = time.time() - NAME_TTL_S + 60
    except Exception:
        logger.debug("campaign names unavailable", exc_info=True)
    return _names_cache


def campaign_entries_in(text: str) -> List[str]:
    """Campaign entry names in ``text`` as written there, longest first and
    not inside a longer one ("Dusk in Leshen", not also "Leshen")."""
    spans = re.finditer(r"[A-Za-z][A-Za-z'\-]*", text or "")
    toks = [(m.group(0), re.sub(r"'s$", "", m.group(0))) for m in spans]
    if not toks:
        return []
    cache = _hashes()
    if not cache["hashes"]:
        return []
    found, used = [], set()
    for n in range(min(cache["max_words"], len(toks)), 0, -1):
        for i in range(len(toks) - n + 1):
            idx = set(range(i, i + n))
            if idx & used or not toks[i][1][:1].isupper():
                continue
            gram = " ".join(t[1] for t in toks[i:i + n])
            if _name_hash(gram) in cache["hashes"]:
                found.append(gram)
                used |= idx
    return found


_CHANGE_RE = re.compile(r"\b(add|update|change|edit|create|write|record|append|rename|put|set|"
                        r"make (a|an) (entry|note|page))\b", re.I)


def asks_for_change(text: str) -> bool:
    return bool(_CHANGE_RE.search(text or ""))


def campaign_names_in(text: str) -> List[str]:
    """The capitalized words of every campaign entry named in ``text`` (and
    "Kanka" itself), for own_names; [] if the campaign can't be reached."""
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", text or "")
    out: List[str] = ["Kanka"] if _CAMPAIGN_WORDS_RE.search(text or "") else []
    if not words:
        return out
    cache = _hashes()
    if not cache["hashes"]:
        return out
    bare = [re.sub(r"'s$", "", w) for w in words]
    for n in range(min(cache["max_words"], len(bare)), 0, -1):
        for i in range(len(bare) - n + 1):
            gram = bare[i:i + n]
            if not gram[0][:1].isupper():
                continue
            if _name_hash(" ".join(gram)) in cache["hashes"]:
                for w in gram:
                    if w[:1].isupper() and w not in out:
                        out.append(w)
    return out


# ── A change to an entry: which entry, and what text ─────────────────────
# Asked to "add to Leshen's entry that the Lords of Llasak meet at dusk", the
# general tool selector proposed {"campaign": "the Dusk in Leshen",
# "entity_id": "Lords of Llasak"} with no text, and the reviewer rightly
# refused it (2026-11-17). Asked only this, the model has two blanks to fill.

_CHANGE_PROMPT = (
    "Someone asked for a change to an entry in their D&D campaign wiki.\n"
    "Request: {request}\n"
    "Entries it names: {entries}\n\n"
    "Which entry is to be changed, and what text is to be added to it? Write the text "
    "as a sentence for the entry, in the request's own words where you can.\n"
    'Answer with JSON only: {{"entity": "<one of the entries named>", "append_entry": "<text>"}}'
)


def change_params(model, request: str, entries: List[str]) -> Optional[Dict]:
    """worldbuild(update) params for a change request, or None if the model's
    answer doesn't name one of the entries or gives no text."""
    if not entries or model is None:
        return None
    import json as _json
    msgs = [{"role": "user", "content": _CHANGE_PROMPT.format(request=request.strip(),
                                                                entries=", ".join(entries))}]
    try:
        out = model.create_chat_completion(messages=msgs, temperature=0.1, max_tokens=200, stream=False)
        from gaia_core.cognition.tool_selector import _extract_content
        text = _extract_content(out)
        m = re.search(r"\{.*\}", text or "", re.S)
        got = _json.loads(m.group(0)) if m else {}
    except Exception:
        logger.debug("campaign change extraction failed", exc_info=True)
        return None
    entity = next((e for e in entries if e.lower() == str(got.get("entity") or "").strip().lower()), None)
    append = str(got.get("append_entry") or "").strip()
    if not entity or not append:
        return None
    return {"action": "update", "entity": entity, "append_entry": append}

"""Facts she is told go in the palace, and come back out of it (1zbx).

Both directions are processes, not her choice. Asked to save a fact she said
"I'll update my knowledge base" and called nothing 5/6 times; asked about one
on plain chat she invented an answer 4/5 ("the Hall of Wonders"), because that
path never looks in memory. She still words the reply, from the real result.
"""
import os
import re
from datetime import datetime, timezone
from typing import Optional

from gaia_common.utils.lookup_memory import content_terms, provenance_note, trust_level
from gaia_core.cognition.reply_scrub import claims_success

_STORE = r"(?:my|your|the)\s+(?:knowledge\s*base|memory\s+palace|memory|palace)"
_WHAT = r"(?:(?:this|that|these|the\s+following)\s*)?(?:(?:fact|facts|information|info|note)\s*)?"
_TOLD_RES = (
    # "Update my knowledge base with this fact: ..."
    re.compile(rf"^\W*(?:please\s+|can\s+you\s+|could\s+you\s+)*(?:update|add\s+to)\s+{_STORE}"
               rf"(?:\s+with)?\s*{_WHAT}[:,\-—]?\s*(?P<fact>\S.*)$", re.IGNORECASE | re.DOTALL),
    # "Please save the following to my knowledge base: ..."
    re.compile(rf"^\W*(?:please\s+|can\s+you\s+|could\s+you\s+)*(?:save|store|add|put|record|write)\s+{_WHAT}"
               rf"(?:to|in|into)\s+{_STORE}\s*[:,\-—]\s*(?P<fact>\S.*)$", re.IGNORECASE | re.DOTALL),
)
# "Remember that the lantern is in the attic", not "remember that time we...".
_REMEMBER_RE = re.compile(r"^\W*(?:please\s+)?remember\s+(?:this\s*[:,\-\u2014]?|that\s*[:,\-\u2014]\s*|that\s+(?!time\b|day\b|one\b|when\b))\s*(?P<fact>\S.*)$",
                          re.IGNORECASE | re.DOTALL)
_QUESTION_RE = re.compile(r"\?\s*$|^\W*(?:who|what|when|where|which|why|how|do\s+you\s+know|tell\s+me)\b", re.IGNORECASE)
_KNOW_ABOUT_RE = re.compile(
    r"^\W*(?:what\s+do\s+you\s+know\s+about|tell\s+me\s+(?:what\s+you\s+know\s+)?about)\s+(?P<topic>.+?)"
    r"(?:\s+from\s+(?:your|the|my)\s+.+?)?\W*$", re.IGNORECASE)
# Of the question's content words, how many the fact must hold to answer it.
MIN_MATCH = 0.5
_SAME_SUBJECT = 0.5
_CALLS_ALL_TRUE_RE = re.compile(r"\bboth\b|\ball\s+(?:right|correct|true)\b|\bplausible\b", re.IGNORECASE)
_ASKS_IF_CHECKED_RE = re.compile(r"\b(?:verified|validated|confirmed|checked|proven|corroborated)\b", re.IGNORECASE)


def speaker(source: str, session_id: str, metadata: Optional[dict]) -> dict:
    """Who is talking, as far as the transport knows, and where. Never from the
    model or the words of the message: a claim to be someone isn't identity.
    Without a real id the session stands in, so what is about "me" stays with
    that conversation."""
    meta = metadata or {}
    user_id = meta.get("user_id")
    if source == "discord":
        where = "in a Discord DM" if meta.get("is_dm") else f"in Discord channel {meta.get('channel_id') or '?'}"
    elif source == "web":
        where = "on the web dashboard"
    else:
        where = f"via {source}"
    who = {
        "who": meta.get("author_name") or (f"Discord user {user_id}" if user_id else "someone"),
        "id": f"{source}:{user_id}" if user_id else f"session:{session_id}",
        "where": where,
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    operator_id = os.environ.get("GAIA_OPERATOR_DISCORD_ID", "").strip()
    operator_name = os.environ.get("GAIA_OPERATOR_NAME", "Azrael")
    if source == "discord" and operator_id and str(user_id) == operator_id:
        who["who"] = operator_name
        who["operator"] = True
    elif who["who"].strip().casefold() == operator_name.casefold():
        # A display name anyone can set.
        who["who"] = f"someone calling themselves {operator_name}"
    return who


def told_fact(text: str) -> Optional[str]:
    """The fact in a request to keep one, or None."""
    for pattern in (*_TOLD_RES, _REMEMBER_RE):
        m = pattern.match(text or "")
        if m:
            return m.group("fact").strip()
    return None


def is_question(text: str) -> bool:
    return bool(_QUESTION_RE.search(text or ""))


def recall_query(question: str) -> str:
    """What to look for: the topic, for "What do you know about X?"."""
    m = _KNOW_ABOUT_RE.match(question or "")
    return m.group("topic") if m else (question or "")


def told_facts_in(hits: list, question: str = "") -> list:
    """Which remembered lookups are facts she was told that answer `question`.
    One shared word isn't enough unless the question is only about that topic:
    "What is The Raven about?" isn't answered by who published it."""
    told = [h for h in hits or [] if isinstance(h, dict) and h.get("kind") == "told" and h.get("snippet")]
    if _KNOW_ABOUT_RE.match(question or ""):
        return told
    terms = content_terms(question)
    return [h for h in told if len(terms & content_terms(h["snippet"])) >= 2]


def disagreeing(hits: list) -> list:
    """Told facts about the same thing that say different things: most of their
    content words shared, the rest not. Nobody wins by rank, the architect
    included; she reports who said what."""
    facts = [(h, content_terms(h["snippet"])) for h in hits]
    out = []
    for i, (hit, terms) in enumerate(facts):
        for j, (other, other_terms) in enumerate(facts):
            if i != j and terms != other_terms and terms and other_terms \
                    and len(terms & other_terms) / len(terms | other_terms) >= _SAME_SUBJECT:
                out.append(hit)
                break
    return out


def covers_each_side(reply: str, sides: list) -> bool:
    """Whether a reply gives every side, without declaring them all true."""
    if _CALLS_ALL_TRUE_RE.search(reply):
        return False
    said = content_terms(reply)
    for side in sides:
        own = content_terms(side["snippet"]) - set().union(*(content_terms(o["snippet"]) for o in sides if o is not side))
        if own and not said & own:
            return False
    return True


def credits_rightly(reply: str, hits: list) -> bool:
    """A reply can't hand the architect's name to a fact he didn't give: she
    shortened "someone calling themselves Azrael" to "Azrael"."""
    name = os.environ.get("GAIA_OPERATOR_NAME", "Azrael")
    if not re.search(rf"\b{re.escape(name)}\b", reply or "", re.IGNORECASE):
        return True
    if re.search(rf"calling\s+themselves\s+{re.escape(name)}\b", reply, re.IGNORECASE):
        return True
    return any(t.get("operator") for h in hits for t in h.get("told_by") or [])


def asks_if_checked(question: str) -> bool:
    return bool(_ASKS_IF_CHECKED_RE.search(question or ""))


def checked_answer(hit: dict) -> str:
    """Whether a fact has been checked, from the record. Not left to her: with
    "not independently checked" in front of her she still answered "Yes"."""
    opening = "Yes." if trust_level(hit) == "validated" else "No, not independently."
    return f"{opening} {hit['snippet']} ({provenance_note(hit)})"


def speaks_to(reply: str, question: str, fact: str) -> bool:
    """Whether a reply uses what the fact adds to the question."""
    added = content_terms(fact) - content_terms(question)
    return bool(content_terms(reply) & (added or content_terms(fact)))

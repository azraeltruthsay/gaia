"""Facts she is told go in the palace, and come back out of it (1zbx).

Both directions are processes, not her choice. Asked to save a fact she said
"I'll update my knowledge base" and called nothing 5/6 times; asked about one
on plain chat she invented an answer 4/5 ("the Hall of Wonders"), because that
path never looks in memory. She still words the reply, from the real result.
"""
import re
from typing import Optional

from gaia_common.utils.lookup_memory import content_terms
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
_QUESTION_RE = re.compile(r"\?\s*$|^\W*(?:who|what|when|where|which|why|how|do\s+you\s+know|tell\s+me)\b", re.IGNORECASE)
_KNOW_ABOUT_RE = re.compile(
    r"^\W*(?:what\s+do\s+you\s+know\s+about|tell\s+me\s+(?:what\s+you\s+know\s+)?about)\s+(?P<topic>.+?)"
    r"(?:\s+from\s+(?:your|the|my)\s+.+?)?\W*$", re.IGNORECASE)
TOLD_SOURCE = "Told in conversation"
# Of the question's content words, how many the fact must hold to answer it.
MIN_MATCH = 0.5


def told_fact(text: str) -> Optional[str]:
    """The fact in a request to keep one, or None."""
    for pattern in _TOLD_RES:
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


def speaks_to(reply: str, question: str, fact: str) -> bool:
    """Whether a reply uses what the fact adds to the question."""
    added = content_terms(fact) - content_terms(question)
    return bool(content_terms(reply) & (added or content_terms(fact)))

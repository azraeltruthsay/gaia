"""'Recite that poem': which work 'that' is (2026-11-11).

A recite request names its work outright or it never reaches the recitation
pipeline (nlu/intent_detection._is_recite_request). Asked "Can you recite that
poem for me?" right after saying "that's The Tyger by William Blake", Core
had no title to fetch, so it wrote a poem from memory: the wrong one ("Do Not
Go Gentle"), with half its lines invented. Here the request is rewritten to
name the work the conversation was just about, and the pipeline then fetches
it like any other.
"""
import re
from typing import Iterable, Optional

_ANAPHOR_RE = re.compile(
    r"(?i)^\s*(?:(?:hey|hi|ok|okay|gaia|please|so|now|can you|could you|would you|will you)[,\s]+)*"
    r"recite\s+(?:(?:it|that|this|the\s+same)(?:\s+(?:poem|one|speech|song|piece|work|text|verse|passage))?"
    r"|the\s+(?:poem|speech|song|piece|passage)(?=\s*(?:for\s+me|to\s+me|please|again)?\s*[?.!]*\s*$))"
    r"\b"
)

# "The Tyger" by William Blake  /  **The Tyger** by William Blake  /  The Tyger by William Blake
_NAME = r"[A-Z](?:[a-z'-]+|\.)"   # "Blake", "T." (an initial); never "Blake." at a sentence end
_BY = r"(?:,?\s+by\s+(" + _NAME + r"(?:\s+(?:" + _NAME + r"|of|de|van|von)){0,4}))"
_QUOTED_RE = re.compile(r'(?:\*\*|["“])\s*([^"”*\n]{2,80}?)\s*(?:\*\*|["”])' + _BY + "?")
_BARE_BY_RE = re.compile(r"\b((?:The|A|An|On|To|Ode)\s+[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,5})" + _BY)


def is_anaphoric_recite(text: str) -> bool:
    return bool(_ANAPHOR_RE.match(text or ""))


def _work_in(text: str) -> Optional[str]:
    """The last work a message names, as "Title by Author" when it says who."""
    found = []
    for m in _QUOTED_RE.finditer(text or ""):
        title, author = m.group(1).strip(), m.group(2)
        if title.lower().startswith(("here is", "here's")) or len(title.split()) > 10:
            continue
        # A quoted first line ("Tyger, Tyger burning bright") isn't a title.
        if "," in title and not author:
            continue
        found.append((m.start(), f"{title} by {author}" if author else title))
    for m in _BARE_BY_RE.finditer(text or ""):
        found.append((m.start(), f"{m.group(1).strip()} by {m.group(2)}"))
    if not found:
        return None
    return max(found)[1]


def resolve(text: str, history: Iterable[dict], lookback: int = 6) -> Optional[str]:
    """A recite request naming the work, when `text` points back at one; else None.

    `history` is the session's messages, oldest first; the current message may
    or may not be its last entry.
    """
    if not is_anaphoric_recite(text):
        return None
    recent = [m for m in list(history or [])[-(lookback + 1):]
              if isinstance(m, dict) and (m.get("content") or "").strip() != (text or "").strip()]
    for msg in reversed(recent):
        work = _work_in(str(msg.get("content") or ""))
        if work:
            return f"Recite {work}"
    return None

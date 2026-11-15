"""Does a self-generated reflection name anything it wasn't given? (tt78)

The idle heartbeat and the Lite journal ask her to write about herself from a
thin context. 2026-11-15 an idle reflection read: "I was in the middle of a
refactor on the knowledge_graph module ... things with Alex have been steady".
There is no Alex and no such refactor, and these entries are stored as her
own memory and read back later.

Telling a small model in the prompt "don't invent" backfires; this is a check
after generation instead. A reflection is kept only if every person-like name
(a capitalised word mid-sentence) and every code identifier (snake_case, a
dotted module, a file name) in it appears in the context she was given, or is
part of her standing vocabulary. Vague mood ("a loose thread", "a gap I
noticed") passes; a specific invented name or module does not.
"""
from __future__ import annotations

import re
from typing import List

# Names she can always use: herself, her minds, her people, her substrate, time.
KNOWN_NAMES = frozenset({
    "gaia", "azrael", "claude", "gemini", "core", "prime", "lite", "nano",
    "gemma", "qwen", "groq", "discord", "notebooklm", "grotto",
    "samvega", "cognitive", "journal", "observer", "doctor",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    # ordinary words capitalised mid-line (after an ellipsis, in a quote)
    "the", "a", "an", "no", "yes", "true", "false", "it", "this", "that", "and", "but",
})

# A capitalised word that is not all caps (acronyms like GPU pass).
_TITLE_WORD = re.compile(r"\b[A-Z][a-z]+(?:'[a-z]+)?\b")
# snake_case, dotted.module, or a file name.
_IDENTIFIER = re.compile(
    r"\b[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)+\b"
    r"|\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"
)
# What may come right before the first word of a sentence.
_SENTENCE_START = re.compile(r"(?:^|[.!?:;\n—–-]|[\"'“‘(])\s*$")


def _is_sentence_start(text: str, pos: int) -> bool:
    return bool(_SENTENCE_START.search(text[max(0, pos - 3):pos]))


def _in(word: str, ctx: str) -> bool:
    w = word.lower()
    return w in ctx or (w.endswith("s") and w[:-1] in ctx)


def ungrounded_specifics(reflection: str, context: str) -> List[str]:
    """Names and identifiers in ``reflection`` found neither in ``context`` nor
    in her standing vocabulary."""
    ctx = (context or "").lower()
    found: List[str] = []
    for m in _TITLE_WORD.finditer(reflection or ""):
        word = m.group(0)
        base = word.split("'")[0].lower()
        if base in KNOWN_NAMES or _in(base, ctx):
            continue
        if _is_sentence_start(reflection, m.start()):
            continue
        found.append(word)
    for m in _IDENTIFIER.finditer(reflection or ""):
        ident = m.group(0)
        if _in(ident, ctx):
            continue
        if re.fullmatch(r"(?:e\.g|i\.e|a\.m|p\.m)", ident):
            continue
        found.append(ident)
    return found


# A file path: absolute, ./relative or ../relative, or a bare name with a
# script/file extension ("run_incidents_list.sh").
_PATH = re.compile(
    r"(?<![\w/.])(?:\.{1,2})?/[\w.\-]+(?:/[\w.\-]+)*"
    r"|(?<![\w/.])[\w\-]+\.(?:sh|py|md|json|ya?ml|txt|toml|cfg|ini|log)\b"
)


def ungrounded_paths(text: str, context: str, exists=None) -> List[str]:
    """File paths in ``text`` that appear neither in ``context`` nor on disk
    (duzp: a seed expansion ran 'cd /gaia/maintenance && python3
    /gaia/scripts/docs_maintenance.py'; neither exists)."""
    import os
    exists = exists or os.path.exists
    ctx = (context or "")
    found: List[str] = []
    for m in _PATH.finditer(text or ""):
        path = m.group(0).rstrip(".,;:")
        if len(path) < 3 or path in ctx:
            continue
        if path.startswith("/") and exists(path):
            continue
        found.append(path)
    return found

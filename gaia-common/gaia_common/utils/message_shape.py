"""The part of a message that is the person's own ask.

Keyword and pattern detectors (persona, intent, stakes, host facts, told facts,
the semantic probe) were written for a message that *is* the ask. When someone
pastes a document, words inside it fire them: 2026-11-15 a 17 KB dev journal
became a D&D persona turn ("spell" in "spelling"), a read_file intent ("reads"
+ a path), a hardware answer ("the GPU") and a "real-world or game-state?"
question ("broke", "re-rolled"), all at once, and she never answered it (5qv4).

``asking_text`` gives detectors the person's own framing; the model still reads
the whole message. A stopgap until Core pre-reads the message (d3px, etyn).
"""
from __future__ import annotations

import re

# A message this short is all ask.
ASK_MAX_CHARS = 600
# How much of the first and last paragraph speaks for a long message with no separator.
EDGE_CHARS = 300

# A line that sets pasted material apart: "-----", "----- JOURNAL -----",
# "=====", "***", a code fence.
_SEPARATOR_RE = re.compile(r"^\s*(?:-{3,}.*|={3,}.*|\*{3,}\s*|_{3,}\s*|```.*|~~~.*)$")


def _edges(text: str) -> str:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        return ""
    first = paras[0][:EDGE_CHARS]
    if len(paras) == 1:
        return first
    return f"{first}\n\n{paras[-1][-EDGE_CHARS:]}"


def asking_text(text: str) -> str:
    """The person's own words in a message: all of a short one; for a long one,
    what comes before pasted material (and a short note after it, if the
    material is closed off), else its first and last paragraphs."""
    text = (text or "").strip()
    if len(text) <= ASK_MAX_CHARS:
        return text
    lines = text.splitlines()
    seps = [i for i, line in enumerate(lines) if _SEPARATOR_RE.match(line)]
    if seps:
        before = "\n".join(lines[:seps[0]]).strip()
        after = "\n".join(lines[seps[-1] + 1:]).strip() if len(seps) > 1 else ""
        if len(after) > ASK_MAX_CHARS:  # the material runs on past the last separator
            after = ""
        ask = "\n\n".join(p for p in (before, after) if p)
        if ask:
            return ask if len(ask) <= 2 * ASK_MAX_CHARS else _edges(ask)
    return _edges(text)


def is_long_message(text: str) -> bool:
    """Whether a message is longer than an ask (it carries pasted material)."""
    return len((text or "").strip()) > ASK_MAX_CHARS

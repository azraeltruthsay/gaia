"""Her inner monologue: asking herself whether to say something (Azrael, 2026-11-19).

A model seldom volunteers to speak again on its own, so where she *may* speak
up (she finished reading a file, something was left unanswered), code asks
the question for her -- in her own voice, as her own thought. It must not read
as someone talking to her: the system frame says it's her mind, the thought is
first person ("I've finished reading..."), and it is marked as nobody's
message. She thinks a line or two, then decides:

    THOUGHT: <what she thinks>
    SAY: <exactly what she'd say to them>   or   SAY: NOTHING

Code keeps only the SAY line, and only when it's real words. Every monologue
is logged (/shared/inner_monologue.jsonl) so what she thought can be read
later, including when she chose to say nothing. Never kept in her chat
history: a thought she reads back as something said becomes a script
(inner_voice.py).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("GAIA.InnerMonologue")

FRAME = (
    "You are GAIA. What follows is your inner monologue: you, thinking to yourself. "
    "No one wrote it to you, no one is waiting for an answer, and no one sees it. "
    "Think it through briefly in your own words, then decide whether to say anything. "
    "Answer in this form:\n"
    "THOUGHT: <a sentence or two of what you think>\n"
    "SAY: <exactly what you would say to them> or SAY: NOTHING")

MARK = "(My own thought. Nobody sent this; I'm thinking to myself.)"

_SAY_RE = re.compile(r"^\s*\**SAY\**\s*:\s*(.*)$", re.I | re.S | re.M)
_THOUGHT_RE = re.compile(r"^\s*\**THOUGHT\**\s*:\s*(.*?)(?=^\s*\**SAY\**\s*:|\Z)", re.I | re.S | re.M)
# "NOTHING", "Nothing to add.", "None -- nothing new": a short refusal is silence
_NOTHING_RE = re.compile(r"^\W*(nothing|none|no|n/a)\b[^.!?\n]{0,40}[.!?]?\W*$", re.I)
# Said again at the end of her thought: a long thought (a file's text) pushes
# the frame's format far back, and she answered freeform (2026-11-19).
CLOSE = "(I answer myself: THOUGHT: what I think. Then SAY: exactly what I'd say, or SAY: NOTHING.)"


def _log_path() -> Path:
    return Path(os.environ.get("SHARED_DIR", "/shared")) / "inner_monologue.jsonl"


def ask_myself(thought: str, generate: Callable[[List[dict]], str], about: str = "",
               session_id: str = "", log: bool = True) -> Dict:
    """Ask herself (first person `thought`) whether to speak. Returns
    {"say": str or None, "thought": str, "raw": str}."""
    raw = (generate([{"role": "system", "content": FRAME},
                     {"role": "user", "content": f"{MARK}\n\n{thought}\n\n{CLOSE}"}]) or "").strip()
    m_say = _SAY_RE.search(raw)
    m_th = _THOUGHT_RE.search(raw)
    said = (m_say.group(1).strip() if m_say else "").strip().strip('"').strip()
    say = None if not said or _NOTHING_RE.match(said) else said
    out = {"say": say, "thought": (m_th.group(1).strip() if m_th else ""), "raw": raw}
    if log:
        try:
            with _log_path().open("a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.time(), "about": about, "session_id": session_id,
                                    "thought": out["thought"], "say": say,
                                    "unparsed": raw if not m_say else ""},
                                   ensure_ascii=False) + "\n")
        except OSError:
            logger.debug("inner monologue log unwritable", exc_info=True)
    logger.info("inner monologue (%s): %s", about or "-", "speaks" if say else "stays quiet")
    return out

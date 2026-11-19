"""Missed messages: after a quiet spell, she looks back at what was said since
her last reply and answers the one she should have, if any (2026-11-19: Andy's
link arrived while she was restarting; a reply to her question went unseen).

She asks herself, in her inner voice, whether one needs her answer and names it
by number, or NOTHING; code keeps the
pick only if it is a real message from someone else, after her last reply, not
looked at before, and recent. Each message is looked at once.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

MAX_AGE_S = 1800
MAX_MESSAGES = 12
_STATE_FILE = "missed_sweep.json"
_SPEAKER_RE = re.compile(r"^([^:\n]{1,64}):\s")
# Answers to a pending approval act on their own; never answered late.
_APPROVAL_RE = re.compile(r"^\s*(?:\d{6}|cancel|approve|deny)\s*[.!]?\s*$", re.I)

def _epoch(ts) -> Optional[float]:
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _dir(base: Optional[Path]) -> Path:
    return Path(base) if base else Path(os.environ.get("SHARED_DIR", "/shared")) / "discord"


def _load(base) -> dict:
    try:
        return json.loads((_dir(base) / _STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(state: dict, base) -> None:
    d = _dir(base)
    d.mkdir(parents=True, exist_ok=True)
    (d / _STATE_FILE).write_text(json.dumps(state), encoding="utf-8")


def unanswered(history: List[Dict], session_id: str, now: Optional[float] = None,
               base: Optional[Path] = None):
    """Her last reply and the messages after it not looked at before."""
    now = now if now is not None else time.time()
    seen = _load(base).get(session_id, 0)
    last_i = max((i for i, m in enumerate(history) if m.get("role") == "assistant"), default=-1)
    after = []
    for m in history[last_i + 1:]:
        when = _epoch(m.get("timestamp"))
        text = _SPEAKER_RE.sub("", m.get("content") or "", count=1)
        if m.get("role") == "user" and when and when > seen and now - when <= MAX_AGE_S \
                and not _APPROVAL_RE.match(text):
            after.append(m)
    mine = history[last_i].get("content", "") if last_i >= 0 else ""
    return mine, after[-MAX_MESSAGES:]


def pick_missed(history: List[Dict], session_id: str, generate: Callable[[List[dict]], str],
                now: Optional[float] = None, base: Optional[Path] = None) -> Optional[Dict]:
    """The message she should answer, as kept in her history, or None."""
    mine, after = unanswered(history, session_id, now, base)
    if not after:
        return None
    state = _load(base)
    state[session_id] = max(_epoch(m.get("timestamp")) or 0 for m in after)
    _save(state, base)          # looked at once, whatever she decides
    # Her own question, in her inner voice (inner_monologue, 2026-11-19): not a
    # system instruction she might take for a person's request.
    from gaia_core.cognition.inner_monologue import ask_myself
    listed = "\n".join(f"{i}. {m.get('content', '')[:400]}" for i, m in enumerate(after, 1))
    thought = ((f"My last message in the channel was: \"{mine[:400]}\"\n\n" if mine else "")
               + f"Since then, people said:\n{listed}\n\nDid anyone say something to me, or ask "
               "me something, that I haven't answered? If one of these needs an answer from me, "
               "which one? If they were talking among themselves, I'll leave it.")
    answer = ask_myself(thought, generate, about="missed messages", session_id=session_id,
                        label="ANSWER", form="the number of the message you should answer",
                        my_form="the number of the message I should answer")["say"] or ""
    m = re.match(r"\D*?(\d{1,2})\b", answer)
    if not m:
        return None
    n = int(m.group(1))
    if not 1 <= n <= len(after):
        return None
    picked = after[n - 1]
    content = picked.get("content", "")
    sp = _SPEAKER_RE.match(content)
    return {"content": content, "speaker": sp.group(1) if sp else "",
            "text": content[sp.end():] if sp else content, "timestamp": picked.get("timestamp")}

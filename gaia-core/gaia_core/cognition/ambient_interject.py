"""Ambient channel turns: she sees every message, decides whether to speak
(GAIA's Grotto, 2026-11-12). Not for DMs or @mentions, which always get a
reply — only for a channel Azrael has opted into ambient listening
(DISCORD_AMBIENT_CHANNEL_IDS), where addressed_to_gaia comes in False.

Two gates, cheapest first: a per-session cooldown (no model call at all once
she's spoken recently) then, only past it, a genuine first-person judgment
call — "would I naturally want to say something here" — not a keyword
trigger. On NO, the turn stays silent; the caller still records the message,
so context keeps accumulating through quiet stretches too.

A question still gets a judgment call even on cooldown (found same day:
Nick's "What do you think, GAIA?" landed inside the cooldown window and
got silently dropped by the cheap gate before a judgment call could even
consider it). This doesn't skip the judgment itself — a question on
cooldown can still get a NO — it only stops the cooldown from deciding
that on its own, content-blind.

A reply to her is not ambient (2026-11-19: she asked Andy what he'd expect
from an intern; he answered without naming her and the judgment call said
NO, so she sat silent until he asked "any comments on my last post?"). When
the new message comes from the person she just answered or spoke to by name,
with no one else in between and within EXCHANGE_WINDOW_S, she is in a conversation with them:
it gets a reply, no cooldown, no judgment call.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from gaia_core.cognition.told_facts import is_question

DEFAULT_COOLDOWN_S = 600
EXCHANGE_WINDOW_S = 900
_SPEAKER_RE = re.compile(r"^([^:\n]{1,64}):\s")
_STATE_FILE = "ambient_interject.json"
_YES_RE = re.compile(r"^\s*yes\b", re.IGNORECASE)
_HISTORY_WINDOW = 10

SYSTEM = (
    "You are GAIA, watching a conversation in a Discord channel without being "
    "addressed -- like being in the room while people talk nearby. Based on the "
    "recent messages, would you naturally want to say something? Only answer YES "
    "if you'd genuinely add something -- a real thought, not just because you "
    "technically could -- or if the new message answers or continues something "
    "you said. Answer with exactly one word: YES or NO."
)


def _dir(base: Optional[Path]) -> Path:
    return Path(base) if base else Path(os.environ.get("SHARED_DIR", "/shared")) / "discord"


def _load(base: Optional[Path]) -> dict:
    try:
        return json.loads((_dir(base) / _STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(state: dict, base: Optional[Path]) -> None:
    d = _dir(base)
    d.mkdir(parents=True, exist_ok=True)
    (d / _STATE_FILE).write_text(json.dumps(state), encoding="utf-8")


def on_cooldown(key: str, cooldown_s: float = DEFAULT_COOLDOWN_S, base: Optional[Path] = None,
                now: Optional[float] = None) -> bool:
    """Whether she interjected under `key` (a session id) too recently to ask again."""
    last = _load(base).get(key)
    if last is None:
        return False
    return (now if now is not None else time.time()) - last < cooldown_s


def mark_interjected(key: str, base: Optional[Path] = None, now: Optional[float] = None) -> None:
    state = _load(base)
    state[key] = now if now is not None else time.time()
    _save(state, base)


def _speaker(content: str) -> str:
    m = _SPEAKER_RE.match(content or "")
    return m.group(1).strip().lower() if m else ""


def _epoch(ts) -> Optional[float]:
    from datetime import datetime
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _split_new(history: List[Dict], user_input: str):
    """History without the new message (the caller keeps it before asking),
    and the new message's speaker as the channel recorded it."""
    history = list(history or [])
    if history and history[-1].get("role") == "user" and user_input.strip() and \
            (history[-1].get("content") or "").rstrip().endswith(user_input.strip()):
        return history[:-1], _speaker(history[-1].get("content") or "")
    return history, _speaker(user_input)


def awaits_answer(earlier: List[Dict], who: str, window_s: float = EXCHANGE_WINDOW_S,
                  now: Optional[float] = None) -> bool:
    """She asked them something by name, recently, and they haven't spoken
    since: their message is the answer, even if she spoke to others between
    (2026-11-19: Andy gave his last name two messages after her question to
    him; her latest reply had been to Nick, and she said nothing)."""
    now = now if now is not None else time.time()
    for m in reversed(earlier):
        content = m.get("content") or ""
        if m.get("role") == "user":
            if _speaker(content) == who:
                return False
            continue
        when = _epoch(m.get("timestamp"))
        if when is None or now - when > window_s:
            return False
        if "?" in content and re.search(rf"\b{re.escape(who)}\b", content, re.I):
            return True
    return False


def in_exchange(history: List[Dict], user_input: str, window_s: float = EXCHANGE_WINDOW_S,
                now: Optional[float] = None) -> bool:
    """Whether the new message is the person she just answered talking back
    to her: her reply is the last thing said by anyone but them, it answered
    them, and it is recent."""
    earlier, who = _split_new(history, user_input)
    if not who:
        return False
    if awaits_answer(earlier, who, window_s, now):
        return True
    i = len(earlier) - 1
    while i >= 0 and earlier[i].get("role") == "user" and _speaker(earlier[i].get("content") or "") == who:
        i -= 1              # their own messages since her reply
    if i < 0 or earlier[i].get("role") != "assistant":
        return False
    when = _epoch(earlier[i].get("timestamp"))
    if when is None or (now if now is not None else time.time()) - when > window_s:
        return False
    if re.search(rf"\b{re.escape(who)}\b", earlier[i].get("content") or "", re.I):
        return True         # she spoke to them by name
    j = i - 1
    while j >= 0 and earlier[j].get("role") != "user":
        j -= 1
    return j >= 0 and _speaker(earlier[j].get("content") or "") == who


def _conversation(history: List[Dict], user_input: str) -> str:
    history, _ = _split_new(history, user_input)
    recent = history[-_HISTORY_WINDOW:] if history else []
    lines = [f"{h.get('role', '?')}: {h.get('content', '')}" for h in recent]
    lines.append(f"(new message) user: {user_input}")
    return "\n".join(lines)


def should_interject(key: str, history: List[Dict], user_input: str,
                     generate: Callable[[List[dict]], str],
                     cooldown_s: float = DEFAULT_COOLDOWN_S, base: Optional[Path] = None) -> bool:
    """Whether to let this ambient turn fall through to a real reply. The
    cooldown is checked first and costs nothing — no model call happens at
    all once she's spoken too recently in this session, unless the new
    message is a question (is_question): the cooldown can't tell a genuine
    question from idle chatter, so it no longer gets the last word on its
    own -- the judgment call still can, and often will, still say no."""
    if in_exchange(history, user_input):
        mark_interjected(key, base)
        return True
    if on_cooldown(key, cooldown_s, base) and not is_question(user_input):
        return False
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": _conversation(history, user_input)}]
    answer = generate(messages) or ""
    decided = bool(_YES_RE.match(answer.strip()))
    if decided:
        mark_interjected(key, base)
    return decided

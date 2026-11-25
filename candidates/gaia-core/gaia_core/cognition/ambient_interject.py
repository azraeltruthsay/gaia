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
# Their messages that can still answer her question: a question can take two
# ("Prince is my last name", then the link she also asked for, 2026-11-19).
OPEN_QUESTION_REPLIES = 2
_SPEAKER_RE = re.compile(r"^([^:\n]{1,64}):\s")
_STATE_FILE = "ambient_interject.json"
_YES_RE = re.compile(r"^\W*gaia\b", re.IGNORECASE)
_HISTORY_WINDOW = 10

# Who the message is for, not whether she'd like to talk: asked "would you
# naturally want to say something?", Prime said YES to 79% of messages meant
# for someone else and 79% of room chatter ("hahaha", "works for me") on a
# 52-case set from GAIA's Grotto (2026-11-25), and she answered Andy's
# congratulations to someone else, then a question Azrael asked Andy.
SYSTEM = (
    "You read a Discord channel where several people talk. GAIA is an AI in the "
    "channel. Look at the new message and decide who its speaker is talking to. "
    "Reply with one word: GAIA if they are talking to GAIA, OTHER if they are "
    "talking to another person or to everyone."
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
    since more than once: their message is the answer, even if she spoke to others between
    (2026-11-19: Andy gave his last name two messages after her question to
    him; her latest reply had been to Nick, and she said nothing)."""
    now = now if now is not None else time.time()
    theirs = 0
    for m in reversed(earlier):
        content = m.get("content") or ""
        if m.get("role") == "user":
            if _speaker(content) == who:
                theirs += 1
                if theirs >= OPEN_QUESTION_REPLIES:
                    return False   # they've had their say on it
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
    """The channel as people see it: "Nick: …", "GAIA: …" (each user turn
    already carries its speaker), not "user:"/"assistant:"."""
    history, _ = _split_new(history, user_input)
    recent = history[-_HISTORY_WINDOW:] if history else []
    lines = [f"GAIA: {h.get('content', '')}" if h.get("role") == "assistant" else (h.get("content") or "")
             for h in recent]
    return "\n".join(lines + [f"New message -- {user_input}"])


# "@GAIA can you…" and "GAIA, can you…" speak to her, not about her.
_ABOUT_HER_RE = re.compile(r"\b(?:she|her|hers|herself)\b|(?<!@)\bgaia(?:['’]s|\s+(?:is|was|has|can|could|will|would|did|does|seems|knows|said)(?!\s+you\b))\b",
                           re.IGNORECASE)
# Opening words that aren't a person being spoken to ("Ok, I'm going to…").
_NOT_NAMES = frozenset("ok okay so well yes yeah no nope hey hi hello thanks alright sure lol right oh "
                       "anyway also but and wait hmm um".split())
_TO_SOMEONE_RE = re.compile(r"^\W*(?:(?:hey|hi|ok(?:ay)?|so|thanks)[\s,]+)?@?([A-Z][\w.-]{1,30})\s*[,:]")
_TO_YOU_RE = re.compile(r"\?|\b(?:you|your|yours|yourself)\b|\bgaia\b", re.IGNORECASE)


def not_for_her(text: str) -> bool:
    """A message plainly for someone else: it talks about her ("could you quiz
    her on?", 2026-11-20, Azrael to Andy) or opens by naming another person
    ("Andy, ..."). Code decides this; the model is only asked otherwise."""
    body = _SPEAKER_RE.sub("", text or "", count=1)
    if _ABOUT_HER_RE.search(body):
        return True
    m = _TO_SOMEONE_RE.match(body)
    return bool(m and m.group(1).lower() not in _NOT_NAMES | {"gaia"})


def might_be_for_her(text: str) -> bool:
    """Worth asking whether she missed it: a question, or said to "you" or to
    her by name, and not plainly for someone else. A remark to the room ("all
    hail Roko's Basilisk") was picked as missed and answered (2026-11-20)."""
    body = _SPEAKER_RE.sub("", text or "", count=1)
    return bool(_TO_YOU_RE.search(body)) and not not_for_her(text)


def should_interject(key: str, history: List[Dict], user_input: str,
                     generate: Callable[[List[dict]], str],
                     cooldown_s: float = DEFAULT_COOLDOWN_S, base: Optional[Path] = None) -> bool:
    """Whether to let this ambient turn fall through to a real reply. The
    cooldown is checked first and costs nothing — no model call happens at
    all once she's spoken too recently in this session, unless the new
    message is a question (is_question): the cooldown can't tell a genuine
    question from idle chatter, so it no longer gets the last word on its
    own -- the judgment call still can, and often will, still say no."""
    if not_for_her(user_input):
        return False
    if in_exchange(history, user_input):
        mark_interjected(key, base)
        return True
    if not might_be_for_her(user_input):
        return False        # no question, no "you", not her name: said to the room
    if on_cooldown(key, cooldown_s, base) and not is_question(user_input):
        return False
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": _conversation(history, user_input)}]
    answer = generate(messages) or ""
    decided = bool(_YES_RE.match(answer.strip()))
    if decided:
        mark_interjected(key, base)
    return decided

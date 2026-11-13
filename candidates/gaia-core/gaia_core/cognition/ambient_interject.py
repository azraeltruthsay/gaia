"""Ambient channel turns: she sees every message, decides whether to speak
(GAIA's Grotto, 2026-11-12). Not for DMs or @mentions, which always get a
reply — only for a channel Azrael has opted into ambient listening
(DISCORD_AMBIENT_CHANNEL_IDS), where addressed_to_gaia comes in False.

Two gates, cheapest first: a per-session cooldown (no model call at all once
she's spoken recently) then, only past it, a genuine first-person judgment
call — "would I naturally want to say something here" — not a keyword
trigger. On NO, the turn stays silent; the caller still records the message,
so context keeps accumulating through quiet stretches too.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

DEFAULT_COOLDOWN_S = 600
_STATE_FILE = "ambient_interject.json"
_YES_RE = re.compile(r"^\s*yes\b", re.IGNORECASE)
_HISTORY_WINDOW = 10

SYSTEM = (
    "You are GAIA, watching a conversation in a Discord channel without being "
    "addressed -- like being in the room while people talk nearby. Based on the "
    "recent messages, would you naturally want to say something? Only answer YES "
    "if you'd genuinely add something -- a real thought, not just because you "
    "technically could. Answer with exactly one word: YES or NO."
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


def _conversation(history: List[Dict], user_input: str) -> str:
    recent = history[-_HISTORY_WINDOW:] if history else []
    lines = [f"{h.get('role', '?')}: {h.get('content', '')}" for h in recent]
    lines.append(f"(new message) user: {user_input}")
    return "\n".join(lines)


def should_interject(key: str, history: List[Dict], user_input: str,
                     generate: Callable[[List[dict]], str],
                     cooldown_s: float = DEFAULT_COOLDOWN_S, base: Optional[Path] = None) -> bool:
    """Whether to let this ambient turn fall through to a real reply. The
    cooldown is checked first and costs nothing — no model call happens at
    all once she's spoken too recently in this session."""
    if on_cooldown(key, cooldown_s, base):
        return False
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": _conversation(history, user_input)}]
    answer = generate(messages) or ""
    decided = bool(_YES_RE.match(answer.strip()))
    if decided:
        mark_interjected(key, base)
    return decided

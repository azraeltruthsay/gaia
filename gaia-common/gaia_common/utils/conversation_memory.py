"""Conversation memory — the palace's read-only record of what was said.

GAIA_Project-6qzb. "What did we talk about regarding the memory palace?" had
nowhere to look: past exchanges lived only in sessions.json (until archived,
then in /shared/history/conversations/, which nothing searched), so the turn
was skipped as chat and Core promised a lookup that never ran.

This is the conversations wing of the palace: one JSONL file per session
under <palace root>/conversations/, one line per exchange (what the person
said, what she replied, when). It is READ-ONLY to her by construction:
  - written only by code — gaia-core's session manager appends each finished
    exchange, and backfill_from_archives() imports archived sessions;
  - there is no MCP write method and no model-facing tool that writes it, so
    no tool call (inline or routed) can add to or rewrite the record.

A record of what was said is not a record of what is true (8xr4.19: her own
transcripts fed back as grounding turned a confabulated answer into
"evidence"). So recall is used only for questions about the conversation
itself, never as general grounding, and is rendered as a past exchange.

Scope is the session id: discord_dm_<user> and web_<id> are one person on one
channel across days. Only the session ids real people arrive on are recorded
(PERSON_SESSION_RE); test batteries, probes, her own loops and the rest of the
synthetic sessions (hundreds of them, under any name) never are.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from gaia_common.utils.lookup_memory import content_terms

logger = logging.getLogger("GAIA.ConversationMemory")

DIRNAME = "conversations"
# The session ids gaia-web gives the people who talk to her: Discord DMs and
# channels (snowflake ids, 17-20 digits — fake test ids like 999999000001 are
# shorter), web chat (web_ + 8 hex) and voice (voice_nano_ + 12 hex). An
# allowlist, not a denylist: synthetic sessions have arbitrary names
# (smoke-test-*, cogtest-*, wsl_push*, rabbit2, her own gaia_*_session loops),
# and their exchanges were never a conversation with anyone.
PERSON_SESSION_RE = re.compile(
    r"^(discord_(dm|channel)_\d{17,20}|web_[0-9a-f]{8}|voice_nano_[0-9a-f]{12})$")
MAX_SIDE_CHARS = 2000      # per side of an exchange; long replies keep their opening
MIN_SCORE = 0.5            # share of the topic's words an exchange must contain
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]")

_lock = threading.Lock()


def default_root() -> Path:
    palace = os.environ.get("MEMPAL_PALACE_PATH", "/knowledge/mempalace")
    return Path(palace) / DIRNAME


def recorded(session_id: str) -> bool:
    """Whether this session's exchanges belong in the record."""
    return bool(PERSON_SESSION_RE.match((session_id or "").strip()))


def _clean(text: str) -> str:
    text = _THINK_RE.sub("", text or "").strip()
    return text if len(text) <= MAX_SIDE_CHARS else text[:MAX_SIDE_CHARS].rstrip() + "…"


def _exchange_id(session_id: str, user: str, reply: str, ts: str) -> str:
    # The same exchange imported twice (live append, then the archive backfill)
    # has the same id; the minute is enough to keep repeated "hi"s apart.
    key = "\x1f".join((session_id, user, reply, (ts or "")[:16]))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


class ConversationMemory:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else default_root()

    def _path(self, session_id: str) -> Path:
        return self.root / f"{_SAFE_RE.sub('_', session_id)[:120]}.jsonl"

    def _ids(self, path: Path) -> set:
        return {r.get("id") for r in self._read(path)}

    @staticmethod
    def _read(path: Path) -> Iterable[Dict]:
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue  # a crash mid-write leaves at most one bad line
        except FileNotFoundError:
            return

    def record_exchange(self, session_id: str, user: str, reply: str,
                        ts: Optional[str] = None, source: str = "live",
                        _known: Optional[set] = None) -> bool:
        """Append one exchange. Never raises; returns whether it was written."""
        try:
            if not recorded(session_id):
                return False
            user, reply = _clean(user), _clean(reply)
            if not user or not reply:
                return False
            ts = ts or datetime.now(timezone.utc).isoformat()
            rec = {"id": _exchange_id(session_id, user, reply, ts), "ts": ts,
                   "session_id": session_id, "user": user, "reply": reply, "source": source}
            path = self._path(session_id)
            with _lock:
                if _known is not None and rec["id"] in _known:
                    return False
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if _known is not None:
                _known.add(rec["id"])
            return True
        except Exception:
            logger.debug("ConversationMemory: record_exchange failed (non-fatal)", exc_info=True)
            return False

    def count(self, session_id: str) -> int:
        return sum(1 for _ in self._read(self._path(session_id)))

    def latest(self, session_id: str, n: int = 3) -> List[Dict]:
        """The session's newest exchanges, newest first."""
        return list(reversed(list(self._read(self._path(session_id)))))[:n]

    def recall(self, session_id: str, topic: str, top_k: int = 3,
               min_score: float = MIN_SCORE) -> List[Dict]:
        """Exchanges in this session about `topic`, best first.

        Score is the share of the topic's content words the exchange contains
        ("memory palace" → both words: 1.0; one: 0.5). Ties go to the newer
        exchange. An empty topic matches nothing (no "show me everything")."""
        want = content_terms(topic)
        if not want:
            return []
        scored = []
        for i, rec in enumerate(self._read(self._path(session_id))):
            have = content_terms(f"{rec.get('user', '')} {rec.get('reply', '')}")
            score = len(want & have) / len(want)
            if score >= min_score:
                scored.append((score, i, rec))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [{**rec, "score": round(score, 3)} for score, _, rec in scored[:top_k]]

    def backfill_from_archives(self, history_dir: str = "/shared/history") -> Dict[str, int]:
        """Import archived sessions (ConversationArchiver output) into the record.
        Idempotent: exchanges already on record are skipped by id."""
        written: Dict[str, int] = {}
        base = Path(history_dir) / "conversations"
        for f in sorted(base.glob("*/*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                logger.debug("ConversationMemory: unreadable archive %s", f, exc_info=True)
                continue
            sid = data.get("session_id") or f.stem
            if not recorded(sid):
                continue
            known = self._ids(self._path(sid))
            n = 0
            for user, reply, ts in exchanges(data.get("messages") or [], data.get("timestamp", "")):
                n += self.record_exchange(sid, user, reply, ts=ts, source="archive", _known=known)
            if n:
                written[sid] = n
        return written


def exchanges(messages: List[Dict], default_ts: str = "") -> Iterable[tuple]:
    """(user, reply, ts) for each user message answered by an assistant message.
    Tool-result and system turns between them are skipped."""
    pending = None
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "user":
            pending = m
        elif role == "assistant" and pending is not None:
            yield (pending.get("content") or "", m.get("content") or "",
                   m.get("timestamp") or pending.get("timestamp") or default_ts)
            pending = None

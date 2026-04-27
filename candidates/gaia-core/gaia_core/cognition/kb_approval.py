"""KB write approval flow — per-session pending writes and trust grants.

State lives in Session.meta so it round-trips through the existing
sessions.json persistence with no new infrastructure. A pending write
is a deferred file.write that's waiting for the user to type
'confirm', 'cancel', or 'trust <kb>' on the next turn.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, Optional


PENDING_WRITE_TTL_SEC = 300

_CONFIRM_RE = re.compile(
    r'^\s*(?:confirm|yes|yep|yeah|do it|ok|okay|sure|please|go ahead)\s*\.?\s*!?\s*$',
    re.IGNORECASE,
)
_CANCEL_RE = re.compile(
    r"^\s*(?:cancel|no|nope|don'?t|abort|stop|never\s*mind|forget\s+it|skip)\s*\.?\s*!?\s*$",
    re.IGNORECASE,
)
_TRUST_RE = re.compile(
    r'^\s*(?:always\s+)?trust\s+([\w\-]+)\s*\.?\s*!?\s*$',
    re.IGNORECASE,
)


def resolve_kb_name(path: str) -> Optional[str]:
    """Extract the KB name from a /knowledge/<kb>/... path.

    Returns None for non-/knowledge paths and for /knowledge/vector_store
    (internal, never user-facing).
    """
    if not path:
        return None
    try:
        rel = Path(path).resolve().relative_to("/knowledge")
    except (ValueError, OSError):
        return None
    if not rel.parts or rel.parts[0] in ("vector_store",):
        return None
    return rel.parts[0]


def is_kb_trusted(session, kb_name: str) -> bool:
    if not kb_name or session is None:
        return False
    return kb_name in (session.meta.get("trusted_kbs") or [])


def add_trusted_kb(session, kb_name: str) -> None:
    if not kb_name or session is None:
        return
    trusted = session.meta.setdefault("trusted_kbs", [])
    if kb_name not in trusted:
        trusted.append(kb_name)


def set_pending_write(
    session, tool_name: str, params: Dict[str, Any], kb_name: str
) -> None:
    if session is None:
        return
    now = time.time()
    session.meta["pending_write"] = {
        "tool_name": tool_name,
        "params": dict(params or {}),
        "kb_name": kb_name,
        "created_at": now,
        "expires_at": now + PENDING_WRITE_TTL_SEC,
    }


def get_pending_write(session) -> Optional[Dict[str, Any]]:
    if session is None:
        return None
    pending = session.meta.get("pending_write")
    if not pending:
        return None
    if pending.get("expires_at", 0) < time.time():
        session.meta.pop("pending_write", None)
        return None
    return pending


def clear_pending_write(session) -> None:
    if session is None:
        return
    session.meta.pop("pending_write", None)


def classify_user_response(user_input: str) -> Dict[str, Any]:
    """Match a user message against confirm/cancel/trust patterns.

    Returns one of:
      {"action": "confirm"}
      {"action": "cancel"}
      {"action": "trust", "kb_name": "<name>"}
      {"action": "none"}
    """
    text = (user_input or "").strip()
    if _CONFIRM_RE.match(text):
        return {"action": "confirm"}
    if _CANCEL_RE.match(text):
        return {"action": "cancel"}
    m = _TRUST_RE.match(text)
    if m:
        return {"action": "trust", "kb_name": m.group(1).lower()}
    return {"action": "none"}


def build_approval_prompt(kb_name: str, path: str, content: str) -> str:
    byte_count = len((content or "").encode("utf-8"))
    return (
        f"I'd like to write {byte_count} bytes to `{path}` "
        f"(in the `{kb_name}` knowledge base). "
        f"Reply **confirm** to proceed, **cancel** to discard, "
        f"or **trust {kb_name}** to allow further writes to this knowledge base for the rest of this session."
    )

"""She revises her own earlier message (Azrael, 2026-11-19).

When a second look (her other mind, a review pass, a person's correction)
finds something wrong in a message she already sent, she can edit it rather
than leave it standing or pile on a follow-up:

  - minor: a name, a typo, a wrong word that changes nothing material. The
    message is edited in place; Discord itself marks it "(edited)".
  - major: what she said was wrong in substance (a fact, an answer someone
    may already have acted on). Quietly rewriting that would be deceptive, so
    the edited message ends with a note saying what changed.

Every revision keeps the old and new text and the reason in
/shared/revisions.jsonl, and her session history gets the new text, so what
she remembers saying matches what people can read. Discord channel sessions
only, for now (gaia-web /api/conversations/{sid}/edit).
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger("GAIA.Revise")

WEB_URL = os.environ.get("GAIA_WEB_URL", "http://gaia-web:6414")


def _log_path() -> Path:
    return Path(os.environ.get("SHARED_DIR", "/shared")) / "revisions.jsonl"


def with_note(text: str, note: str) -> str:
    """The corrected message with what changed, for a major correction."""
    return f"{text.rstrip()}\n\n*(Edited: {note.strip().rstrip('.')}.)*"


def edit_on_discord(session_id: str, containing: str, text: str) -> Optional[str]:
    data = json.dumps({"containing": containing, "text": text}).encode()
    try:
        req = urllib.request.Request(f"{WEB_URL}/api/conversations/{session_id}/edit", data=data,
                                     method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=40) as r:
            out = json.loads(r.read() or b"{}")
        return out.get("old") if out.get("edited") else None
    except Exception as e:
        logger.warning("revise: edit on Discord failed for %s: %s", session_id, e)
        return None


def revise(session_manager, session_id: str, containing: str, new_text: str, reason: str,
           major: bool = False, note: str = "", edit=edit_on_discord) -> Optional[str]:
    """Edit her message containing `containing` to `new_text` (with `note` shown
    when `major`). Returns the text now in the message, or None if nothing was
    edited (not found, not hers, not a channel)."""
    if major and not note.strip():
        raise ValueError("a major correction carries a note saying what changed")
    final = with_note(new_text, note) if major else new_text.strip()
    old = edit(session_id, containing, final)
    if old is None:
        logger.info("revise: no message of hers containing %r in %s", containing[:40], session_id)
        return None
    replaced = False
    try:
        history = session_manager.get_history(session_id)
        for m in reversed(history):
            if m.get("role") == "assistant" and containing in (m.get("content") or ""):
                m["content"] = final
                m.setdefault("revisions", []).append({"at": time.time(), "old": old, "reason": reason,
                                                      "major": major})
                replaced = True
                break
        if replaced:
            session_manager._save_state()
    except Exception:
        logger.warning("revise: history update failed for %s", session_id, exc_info=True)
    try:
        with _log_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "session_id": session_id, "old": old, "new": final,
                                "reason": reason, "major": major, "history_updated": replaced},
                               ensure_ascii=False) + "\n")
    except OSError:
        logger.debug("revisions log unwritable", exc_info=True)
    logger.info("revise: %s edit in %s (%s)", "major" if major else "minor", session_id, reason)
    return final

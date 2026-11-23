"""She corrects a mistaken message while it still matters (aidw.6,
GROUNDED_SPEECH.md part C3).

Azrael, 2026-11-21: a Samvega finding is timestamped, and if it's about a
recent message, "that could be the impetus for her editing her message to
improve accuracy, then review the finding and run a fix later during her
sleep tasks."

When a finding is confirmed against evidence (C1) and the message is fresh:
  1. she answers the question again from the evidence that showed the
     mistake — the same answer-from-a-tool-result shape a lookup turn uses,
     i.e. what she would have said had she looked;
  2. code checks the new answer against that evidence (finding_grounding);
  3. the message is edited through revise.py as a major correction, which
     carries a plain note saying what changed (rewriting a wrong answer
     silently would be deceptive) and updates her session history;
  4. the finding moves to `amended`, with the new text.

Fixing the cause (why she didn't look) is left to sleep (C4); the finding
stays active until then. Discord channels and DMs, and web chat (aidw.10).
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("GAIA.SamvegaAmend")

MAX_AGE_S = 1800  # a correction after half an hour is a new message, not an edit
CORE_URL = os.environ.get("GAIA_CORE_SELF_URL", "http://localhost:6415")

_SOURCES = {
    "palace": "my memory palace", "study": "my adapters", "issue": "my issue ledger",
    "introspect": "my own status", "web": "the web", "file": "the files",
    "worldbuild": "the campaign notes", "samvega": "my ledger of mistakes", "manage": "my own status",
}


def source_phrase(call: str) -> str:
    """'palace(action=navigate, wing=operational)' -> 'my memory palace'."""
    tool = (call or "").split("(", 1)[0].strip().lower()
    return _SOURCES.get(tool, "the source")


def _age_s(sent_at: str) -> Optional[float]:
    try:
        t = datetime.fromisoformat((sent_at or "").replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - t).total_seconds()
    except ValueError:
        return None


def _anchor(said: str) -> str:
    """Words to find the message by: its opening, without a '[Prime]' label."""
    text = re.sub(r"^\s*\[[^\]\n]{1,40}\]\s*", "", said or "").strip()
    first = text.splitlines()[0] if text else ""
    return first[:80].strip()


def _revise_http(session_id: str, containing: str, text: str, reason: str, note: str) -> Optional[str]:
    data = json.dumps({"containing": containing, "text": text, "reason": reason,
                       "major": True, "note": note}).encode("utf-8")
    req = urllib.request.Request(f"{CORE_URL}/api/sessions/{session_id}/revise", data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read() or b"{}")
    return out.get("text") if out.get("revised") else None


def amendable(session_id: str, message_ref: Dict[str, Any], max_age_s: float = MAX_AGE_S) -> str:
    """'' if the message can be corrected now, else why not."""
    if not (session_id or "").startswith(("discord_channel_", "discord_dm_", "web_")):
        return "only Discord and web-chat messages can be edited"
    age = _age_s((message_ref or {}).get("sent_at", ""))
    if age is None:
        return "no time recorded for the message"
    if age > max_age_s:
        return f"the message is {int(age // 60)} minutes old"
    return ""


def maybe_amend(finding_path, *, question: str, said: str, evidence: str, grounding_call: str,
                session_id: str, message_ref: Dict[str, Any], generate: Callable[[list], str],
                revise: Callable[..., Optional[str]] = None, max_age_s: float = MAX_AGE_S) -> Optional[str]:
    """Correct the message a confirmed finding is about. Returns the text now
    in the message, or None (with the reason logged and noted on the finding)."""
    from gaia_common.utils import samvega_ledger as ledger
    from gaia_core.cognition import finding_grounding as fg
    from pathlib import Path
    ref = Path(finding_path).stem

    def _skip(why: str) -> None:
        logger.info("Samvega amend %s: not amended (%s)", ledger.short_id(ref), why)
        try:
            ledger.note(ref, f"Not amended: {why}.", by="samvega_amend")
        except Exception:
            pass

    why = amendable(session_id, message_ref, max_age_s)
    if why:
        _skip(why)
        return None
    if not fg.has_lookup(evidence):
        # The world-state block alone isn't a source: the note would say "I
        # checked the source" of a rewrite from nothing (2026-11-23 DM).
        _skip("nothing was looked up to answer from")
        return None
    containing = _anchor(said)
    if len(containing) < 12:
        _skip("too little of the message to find it by")
        return None
    # aidw.8: rewriting from a source that doesn't answer the question swaps
    # one unsupported answer for another (her tool list, asked which tools
    # she uses most).
    from gaia_core.cognition.pre_send_check import source_answers
    if not source_answers(question, evidence, generate):
        _skip("the evidence doesn't answer the question, so there is nothing to correct it from")
        return None
    try:
        from gaia_common.utils.tool_call_parser import tool_answer_messages
        new = (generate(tool_answer_messages(question, grounding_call or "the lookup", evidence)) or "").strip()
    except Exception:
        logger.debug("Samvega amend: rewrite failed", exc_info=True)
        new = ""
    new = re.sub(r"^\s*\[[^\]\n]{1,40}\]\s*", "", new).strip()
    if not new or new == said.strip():
        _skip("no different answer came")
        return None
    # The correction must hold up against the same evidence.
    if fg.confirm(new, new, evidence, question, generate) == fg.CONFIRMED:
        _skip("the rewrite didn't hold up against the evidence either")
        return None
    note = f"I checked {source_phrase(grounding_call)} and corrected this"
    try:
        final = (revise or _revise_http)(session_id, containing, new,
                                         f"samvega {ledger.short_id(ref)}: confirmed against "
                                         f"{grounding_call or 'the turn’s lookup'}", note)
    except Exception as e:
        logger.warning("Samvega amend: revise failed: %s", e)
        final = None
    if not final:
        _skip("the message couldn't be found to edit")
        return None
    ledger.set_status(ref, "amended", reason=f"Message corrected from {grounding_call}.",
                      by="samvega_amend", as_code=True, amended_text=final)
    logger.info("Samvega amend %s: message corrected in %s", ledger.short_id(ref), session_id)
    return final

"""The person's answer to a pending approval, in any chat (2026-11-10; TOTP
since GAIA_Project-96xm).

A sensitive tool call (shell, writes outside the sandbox) parks in gaia-mcp
until a human replies with their current authenticator (TOTP) code. Discord
had a waiter for that in gaia-web, but the pending action never reached it,
and the web chat had nothing: the code went to her as a message, and she
answered by inventing a curl command to run. Here the parked action is kept
on the session; the next message that looks like a 6-digit code is forwarded
to gaia-mcp (only it can say whether the code is actually valid -- there is
nothing left to check locally), "cancel" drops it, anything else is an
ordinary message.
"""
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

KEY = "pending_mcp_approval"
TTL_S = 15 * 60  # gaia-mcp's own expiry
_CANCEL = {"cancel", "deny", "no", "abort", "stop"}
_CODE_RE = re.compile(r"\A\d{6}\Z")
# Parked by something other than her turn in this process (a sleep task),
# for a session: a file any process can write. Session meta lives in each
# SessionManager's own copy and the live one would save over it.
APPROVALS_DIR = Path(os.environ.get("SHARED_DIR", "/shared")) / "approvals"


def _file(session_id: str) -> Path:
    return APPROVALS_DIR / (re.sub(r"[^A-Za-z0-9_-]", "_", session_id) + ".json")


def park_for(session_id: str, action_id: str, label: str) -> None:
    """Hold a parked action for `session_id`'s next reply, from any process."""
    if not (session_id and action_id):
        return
    APPROVALS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _file(session_id).with_suffix(".tmp")
    tmp.write_text(json.dumps({"action_id": action_id, "label": label,
                               "at": time.time()}))
    os.replace(tmp, _file(session_id))


def _parked(session_id: str) -> Optional[dict]:
    try:
        return json.loads(_file(session_id).read_text())
    except (OSError, ValueError):
        return None


def remember(session_manager, session_id: str, action_id: str, label: str) -> None:
    if not (session_id and action_id):
        return
    session_manager.set_session_meta(session_id, KEY, {
        "action_id": action_id, "label": label, "at": time.time()})


def _forget(session_manager, session_id: str) -> None:
    if session_manager.get_session_meta(session_id, KEY):
        session_manager.set_session_meta(session_id, KEY, None)
    try:
        _file(session_id).unlink()
    except OSError:
        pass


def _show(result) -> str:
    """The approved action's result, short enough for a chat."""
    if isinstance(result, dict):
        inner = result.get("result", result)
        if isinstance(inner, dict):
            for k in ("stdout", "output", "content", "text"):
                if inner.get(k):
                    out = str(inner[k]).rstrip()
                    err = str(inner.get("stderr") or "").rstrip()
                    body = out + (f"\n[stderr]\n{err}" if err else "")
                    return f"```\n{body[:1800]}\n```"
        return f"```\n{json.dumps(inner, default=str)[:1800]}\n```"
    return str(result)[:1800]


def handle(session_manager, session_id: str, text: str, approve=None) -> Optional[str]:
    """A reply if `text` answers this session's pending approval, else None."""
    try:
        pending = session_manager.get_session_meta(session_id, KEY) or _parked(session_id)
    except Exception:
        return None
    if not pending:
        return None
    if time.time() - pending.get("at", 0) > TTL_S:
        _forget(session_manager, session_id)
        return None
    said = (text or "").strip().rstrip(".!")
    label = pending.get("label") or "that"
    if said.lower() in _CANCEL:
        _forget(session_manager, session_id)
        return f"Cancelled `{label}`; it won't run."
    # GAIA_Project-96xm: no local check is possible anymore -- only
    # gaia-mcp (holding the TOTP secret) can say whether a code is valid.
    # Only forward something that looks like a 6-digit code; anything else
    # isn't an approval reply at all (falls through as ordinary chat).
    if not _CODE_RE.match(said):
        return None
    if approve is None:
        from gaia_core.utils.mcp_client import approve_action_via_mcp as approve
    r = approve(pending["action_id"], said)
    if not r.get("ok"):
        # Wrong or stale code -- leave it parked so they can just retry.
        return f"That code didn't approve `{label}`: {r.get('error') or 'unknown error'}. Try again, or say cancel."
    _forget(session_manager, session_id)
    return f"Approved. `{label}` ran:\n{_show(r.get('result'))}"

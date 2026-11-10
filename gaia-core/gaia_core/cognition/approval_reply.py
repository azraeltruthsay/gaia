"""The person's answer to an approval challenge, in any chat (2026-11-10).

A sensitive tool call (shell, writes outside the sandbox) parks in gaia-mcp
until a human replies with the challenge reversed. Discord had a waiter for
that in gaia-web, but the pending action never reached it, and the web chat
had nothing: the reversed code went to her as a message, and she answered
"HNMFY" by inventing a curl command to run. Here the parked action is kept on
the session; the next message that is the reversed code approves it through
MCP, "cancel" drops it, anything else is an ordinary message.
"""
import json
import re
import time
from typing import Optional

KEY = "pending_mcp_approval"
TTL_S = 15 * 60  # gaia-mcp's own expiry
_CANCEL = {"cancel", "deny", "no", "abort", "stop"}


def remember(session_manager, session_id: str, action_id: str, challenge: str, label: str) -> None:
    if not (session_id and action_id and challenge):
        return
    session_manager.set_session_meta(session_id, KEY, {
        "action_id": action_id, "challenge": challenge, "label": label, "at": time.time()})


def _forget(session_manager, session_id: str) -> None:
    session_manager.set_session_meta(session_id, KEY, None)


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
        pending = session_manager.get_session_meta(session_id, KEY)
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
    expected = pending["challenge"][::-1].upper()
    if said.upper() == expected:
        _forget(session_manager, session_id)
        if approve is None:
            from gaia_core.utils.mcp_client import approve_action_via_mcp as approve
        r = approve(pending["action_id"], expected)
        if not r.get("ok"):
            return f"Approved, but `{label}` didn't run: {r.get('error') or 'unknown error'}"
        return f"Approved. `{label}` ran:\n{_show(r.get('result'))}"
    if re.fullmatch(r"[A-Za-z]{5}", said):
        return (f"That isn't the code for `{label}`: the challenge is `{pending['challenge']}`, "
                f"reply with it reversed (`{expected}`), or say cancel.")
    return None

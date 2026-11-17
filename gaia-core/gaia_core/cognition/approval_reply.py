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
from datetime import datetime
from pathlib import Path
from typing import Optional

KEY = "pending_mcp_approval"
TTL_S = 15 * 60  # gaia-mcp's own expiry
_CANCEL = {"cancel", "deny", "no", "abort", "stop"}
_CODE_RE = re.compile(r"\A\d{6}\Z")
# The card she shows when an action parks (main.py, agent_core). Kept in the
# history in short form only (GAIA_Project-0ait): given the whole card she
# copied it into a later reply with no new action behind it, and a code
# typed then approved the older action.
CARD_RE = re.compile(
    r"That needs your approval first: `(?P<label>[^`\n]{1,80})`.*?"
    r"authenticator code(?: to approve)?,? or say cancel\.",
    re.DOTALL)
# Any other wording that asks for a code: hers, never the code's.
_ASK_RE = re.compile(
    r"[^.\n]*\b(?:send|reply|enter|provide|give|type|share)\b[^.\n]{0,60}"
    r"\bauthenticator code\b[^.\n]*[.\n]?", re.IGNORECASE)

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


def remember(session_manager, session_id: str, action_id: str, label: str,
             shown: Optional[dict] = None) -> None:
    if not (session_id and action_id):
        return
    session_manager.set_session_meta(session_id, KEY, {
        "action_id": action_id, "label": label, "shown": shown, "at": time.time()})


def compact(text: str) -> str:
    """`text` as the history keeps it: an approval card cut to one line that
    names the action, and any other request for a code dropped."""
    if not text or "authenticator code" not in text.lower():
        return text
    out = CARD_RE.sub(lambda m: f"(I asked for approval to run `{m.group('label')}`.)", text)
    out = _ASK_RE.sub("", out)
    return out if out.strip() else "(I asked for approval.)"


def _replies_since(session_manager, session_id: str, since: float) -> int:
    """Her replies in this session after `since` (epoch seconds)."""
    try:
        history = session_manager.get_history(session_id) or []
    except Exception:
        return 0
    n = 0
    for turn in history:
        if turn.get("role") != "assistant":
            continue
        try:
            at = datetime.fromisoformat(str(turn.get("timestamp"))).timestamp()
        except (TypeError, ValueError):
            continue
        if at > since:
            n += 1
    return n


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


def handle(session_manager, session_id: str, text: str, approve=None,
           cancel=None) -> Optional[str]:
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
        # twxs: drop it in gaia-mcp too, or a code could still approve it.
        if cancel is None:
            from gaia_core.utils.mcp_client import cancel_action_via_mcp as cancel
        try:
            r = cancel(pending["action_id"]) or {}
        except Exception as e:
            r = {"ok": False, "error": str(e)}
        if not r.get("ok"):
            return (f"Cancelled `{label}` here, but gaia-mcp didn't confirm it "
                    f"({r.get('error') or 'no answer'}); it expires on its own within 15 minutes.")
        return f"Cancelled `{label}`; it won't run."
    # GAIA_Project-96xm: no local check is possible anymore -- only
    # gaia-mcp (holding the TOTP secret) can say whether a code is valid.
    # Only forward something that looks like a 6-digit code; anything else
    # isn't an approval reply at all (falls through as ordinary chat).
    if not _CODE_RE.match(said):
        return None
    # 0ait: the card is her last word before a code. If she has said
    # something since, the person may be answering something else (she once
    # copied an old card into a later reply): show what is actually waiting,
    # and take the next code for it.
    shown_at = max(pending.get("at", 0), pending.get("reshown_at", 0))
    if _replies_since(session_manager, session_id, shown_at) > 1:
        pending = dict(pending, reshown_at=time.time())
        if session_manager.get_session_meta(session_id, KEY):
            session_manager.set_session_meta(session_id, KEY, pending)
        else:
            _file(session_id).write_text(json.dumps(pending))
        what = f"`{label}`" + (f" {json.dumps(pending['shown'])}" if pending.get("shown") else "")
        mins = max(1, round((time.time() - pending.get("at", time.time())) / 60))
        return (f"I didn't use that code. The one thing waiting for approval is {what}, "
                f"from {mins} min ago. To approve it, send a fresh code; or say cancel.")
    if approve is None:
        from gaia_core.utils.mcp_client import approve_action_via_mcp as approve
    r = approve(pending["action_id"], said)
    if not r.get("ok"):
        # Wrong or stale code -- leave it parked so they can just retry.
        return f"That code didn't approve `{label}`: {r.get('error') or 'unknown error'}. Try again, or say cancel."
    _forget(session_manager, session_id)
    res = r.get("result")
    if isinstance(res, dict) and res.get("approved") and res.get("ok") is False:
        # The code was right; the tool itself failed. Retrying the code
        # can't help: the approval is spent (96xm, 2026-11-16).
        return f"Approved, but `{label}` failed: {res.get('error') or 'unknown error'}"
    return f"Approved. `{label}` ran:\n{_show(res)}"

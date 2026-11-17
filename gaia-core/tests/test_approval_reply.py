"""The answer to a pending approval in chat (2026-11-10; TOTP since
GAIA_Project-96xm): a 6-digit authenticator code is forwarded to gaia-mcp
(only it can say whether the code is valid), cancel drops it, a wrong code
leaves the action parked for a retry, anything else is an ordinary
message."""

import time

from gaia_core.cognition import approval_reply as ar


class Sessions:
    def __init__(self):
        self.meta = {}

    def get_session_meta(self, sid, key, default=None):
        return self.meta.get((sid, key), default)

    def set_session_meta(self, sid, key, value):
        self.meta[(sid, key)] = value


def _parked():
    s = Sessions()
    ar.remember(s, "dm1", "a-1", "shell(run)")
    return s


def test_correct_code_approves_and_shows_output():
    s, seen = _parked(), []

    def approve(action_id, code):
        seen.append((action_id, code))
        return {"ok": True, "result": {"ok": True, "result": {"stdout": "GPU 0: RTX 5080", "stderr": ""}}}

    reply = ar.handle(s, "dm1", "482913", approve=approve)
    assert seen == [("a-1", "482913")]
    assert reply.startswith("Approved. `shell(run)` ran:") and "RTX 5080" in reply
    assert ar.handle(s, "dm1", "482913", approve=approve) is None      # used up


def test_wrong_code_leaves_it_parked_for_retry():
    s, seen = _parked(), []

    def approve(action_id, code):
        seen.append((action_id, code))
        return {"ok": False, "error": "invalid or expired authenticator code"}

    reply = ar.handle(s, "dm1", "000000", approve=approve)
    assert seen == [("a-1", "000000")]
    assert "didn't approve" in reply
    # Still parked -- a wrong code shouldn't lose the pending action.
    assert s.get_session_meta("dm1", ar.KEY)["action_id"] == "a-1"


def test_cancel_drops_the_pending_action():
    s = _parked()
    assert ar.handle(s, "dm1", "cancel").startswith("Cancelled")
    assert ar.handle(s, "dm1", "482913", approve=lambda *a: 1 / 0) is None


def test_ordinary_message_passes_and_keeps_pending():
    s = _parked()
    assert ar.handle(s, "dm1", "Can you explain why the sky is blue?") is None
    assert s.get_session_meta("dm1", ar.KEY)["action_id"] == "a-1"


def test_non_six_digit_text_is_not_treated_as_an_approval_attempt():
    s = _parked()
    # Old reversed-challenge-shaped replies (5 letters) are ordinary
    # messages now -- there's nothing for gaia-core to check locally.
    assert ar.handle(s, "dm1", "CYKOE", approve=lambda *a: 1 / 0) is None
    assert s.get_session_meta("dm1", ar.KEY)["action_id"] == "a-1"


def test_expired_or_other_session():
    s = _parked()
    assert ar.handle(s, "other", "482913") is None
    s.meta[("dm1", ar.KEY)]["at"] = time.time() - ar.TTL_S - 1
    assert ar.handle(s, "dm1", "482913", approve=lambda *a: 1 / 0) is None


def test_parked_call_is_pending_not_success(monkeypatch):
    """call_jsonrpc: /request_approval's ticket (ok + action_id, no result)
    means the call is parked, not that it ran. No challenge comes back
    anymore (GAIA_Project-96xm) -- there's nothing server-generated to
    reveal."""
    from gaia_core.utils import mcp_client as mc

    class R:
        status_code = 403
        def json(self):
            return {"error": "requires approval"}
        def raise_for_status(self):
            raise AssertionError("not reached")

    monkeypatch.setattr(mc.requests, "post", lambda *a, **k: R())
    monkeypatch.setattr(mc, "request_approval_via_mcp", lambda **k: {
        "ok": True, "action_id": "a-9", "proposal": None})
    r = mc.call_jsonrpc(method="shell", params={"action": "run", "command": "nvidia-smi"})
    assert r["ok"] is False and r["pending_approval"] is True
    assert r["action_id"] == "a-9"
    assert "challenge" not in r


def test_a_parked_file_from_another_process_is_answered(tmp_path, monkeypatch):
    """The penpal sleep task parks its letter for Azrael's DM from a file."""
    monkeypatch.setattr(ar, "APPROVALS_DIR", tmp_path)
    ar.park_for("discord_dm_1", "a-9", "send my letter")
    s, seen = Sessions(), []
    reply = ar.handle(s, "discord_dm_1", "482913",
                      approve=lambda a, c: seen.append((a, c)) or {"ok": True, "result": {"ok": True}})
    assert seen == [("a-9", "482913")] and reply.startswith("Approved.")
    assert not list(tmp_path.iterdir())                       # used up
    assert ar.handle(s, "discord_dm_2", "482913") is None       # only for its session


def test_approved_but_the_tool_failed_says_so_and_is_spent():
    # 2026-11-16: the code was right, the write was refused by the path
    # allowlist; that used to read as "That code didn't approve".
    s = _parked()

    def approve(action_id, code):
        return {"ok": True, "result": {"ok": False, "approved": True, "method": "file",
                                       "error": "Path not allowed"}}

    reply = ar.handle(s, "dm1", "482913", approve=approve)
    assert reply.startswith("Approved, but `shell(run)` failed: Path not allowed")
    assert ar.handle(s, "dm1", "482913", approve=approve) is None


# --- 0ait: a card copied from history must not approve an older action ---

from datetime import datetime, timezone  # noqa: E402


class SessionsWithHistory(Sessions):
    def __init__(self):
        super().__init__()
        self.history = {}

    def say(self, sid, role, text, at=None):
        ts = datetime.fromtimestamp(at or time.time(), timezone.utc).isoformat()
        self.history.setdefault(sid, []).append({"role": role, "content": text, "timestamp": ts})

    def get_history(self, sid):
        return self.history.get(sid, [])


CARD = ("That needs your approval first: `worldbuild(update)` "
        '{"action": "update", "entity": "Leshen", "append_entry": "x"}. '
        "Reply with your current 6-digit authenticator code to approve, or say cancel.")


def test_card_is_kept_as_one_line():
    out = ar.compact("Sure. " + CARD)
    assert out == "Sure. (I asked for approval to run `worldbuild(update)`.)"
    assert "authenticator" not in out and "Leshen" not in out


def test_other_requests_for_a_code_are_dropped():
    out = ar.compact("Done thinking. Please send your authenticator code to proceed.\nAnything else?")
    assert "authenticator" not in out and "Anything else?" in out
    assert ar.compact("Hello there.") == "Hello there."


def test_code_right_after_the_card_approves():
    s = SessionsWithHistory()
    ar.remember(s, "dm1", "a-1", "worldbuild(update)", {"entity": "Leshen"})
    s.say("dm1", "assistant", CARD)
    seen = []
    reply = ar.handle(s, "dm1", "123456",
                      approve=lambda a, c: seen.append(a) or {"ok": True, "result": {"ok": True}})
    assert seen == ["a-1"] and reply.startswith("Approved.")


def test_code_after_she_spoke_again_reconfirms_first():
    s = SessionsWithHistory()
    ar.remember(s, "dm1", "a-1", "worldbuild(update)", {"entity": "Leshen"})
    s.say("dm1", "assistant", CARD)
    s.say("dm1", "user", "add it again")
    s.say("dm1", "assistant", CARD)          # copied, no new action behind it
    seen = []

    def approve(a, c):
        seen.append(a)
        return {"ok": True, "result": {"ok": True}}

    reply = ar.handle(s, "dm1", "123456", approve=approve)
    assert seen == [] and "didn't use that code" in reply
    assert "worldbuild(update)" in reply and "Leshen" in reply
    s.say("dm1", "assistant", reply)         # agent_core stores the reconfirmation
    reply = ar.handle(s, "dm1", "654321", approve=approve)
    assert seen == ["a-1"] and reply.startswith("Approved.")


def test_parked_from_a_file_reconfirms_too(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "APPROVALS_DIR", tmp_path)
    s = SessionsWithHistory()
    ar.park_for("dm2", "p-1", "send my letter")
    s.say("dm2", "assistant", "one")
    s.say("dm2", "assistant", "two")
    reply = ar.handle(s, "dm2", "123456", approve=lambda *a: 1 / 0)
    assert "didn't use that code" in reply
    assert ar._parked("dm2")["reshown_at"]


def test_a_rejected_code_report_is_kept():
    msg = "That code didn't approve `x`: invalid or expired authenticator code. Try again, or say cancel."
    assert ar.compact(msg) == msg

"""The answer to an approval challenge in chat (2026-11-10): the reversed
code approves the parked action through MCP, cancel drops it, a wrong
five-letter code is caught, anything else is an ordinary message."""

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
    ar.remember(s, "dm1", "a-1", "EOKYC", "shell(run)")
    return s


def test_reversed_code_approves_and_shows_output():
    s, seen = _parked(), []

    def approve(action_id, code):
        seen.append((action_id, code))
        return {"ok": True, "result": {"ok": True, "result": {"stdout": "GPU 0: RTX 5080", "stderr": ""}}}

    reply = ar.handle(s, "dm1", "cykoe", approve=approve)
    assert seen == [("a-1", "CYKOE")]
    assert reply.startswith("Approved. `shell(run)` ran:") and "RTX 5080" in reply
    assert ar.handle(s, "dm1", "CYKOE", approve=approve) is None      # used up


def test_cancel_and_wrong_code():
    s = _parked()
    assert "isn't the code" in ar.handle(s, "dm1", "HNMFY", approve=lambda *a: 1 / 0)
    assert ar.handle(s, "dm1", "cancel").startswith("Cancelled")
    assert ar.handle(s, "dm1", "CYKOE", approve=lambda *a: 1 / 0) is None


def test_ordinary_message_passes_and_keeps_pending():
    s = _parked()
    assert ar.handle(s, "dm1", "Can you explain why the sky is blue?") is None
    assert s.get_session_meta("dm1", ar.KEY)["action_id"] == "a-1"


def test_expired_or_other_session():
    s = _parked()
    assert ar.handle(s, "other", "CYKOE") is None
    s.meta[("dm1", ar.KEY)]["at"] = time.time() - ar.TTL_S - 1
    assert ar.handle(s, "dm1", "CYKOE", approve=lambda *a: 1 / 0) is None


def test_approval_error_is_reported():
    s = _parked()
    reply = ar.handle(s, "dm1", "CYKOE", approve=lambda *a: {"ok": False, "error": "expired"})
    assert "didn't run: expired" in reply


def test_parked_call_is_pending_not_success(monkeypatch):
    """call_jsonrpc: /request_approval's ticket (ok + action_id, no result)
    means the call is parked, not that it ran."""
    from gaia_core.utils import mcp_client as mc

    class R:
        status_code = 403
        def json(self):
            return {"error": "requires approval"}
        def raise_for_status(self):
            raise AssertionError("not reached")

    monkeypatch.setattr(mc.requests, "post", lambda *a, **k: R())
    monkeypatch.setattr(mc, "request_approval_via_mcp", lambda **k: {
        "ok": True, "action_id": "a-9", "challenge": "NXEDO", "proposal": None})
    r = mc.call_jsonrpc(method="shell", params={"action": "run", "command": "nvidia-smi"})
    assert r["ok"] is False and r["pending_approval"] is True
    assert r["action_id"] == "a-9" and r["challenge"] == "NXEDO"

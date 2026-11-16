"""A tool-routed file write asks for the human's code like any other (diq7).

2026-11-16: tool routing wrote files locally in gaia-core with no approval and
its own path rules, while her inline tool calls went to gaia-mcp and needed a
TOTP code. The same request was refused in one path and approved in the other.
"""
from types import SimpleNamespace

from gaia_core.cognition import agent_core as ac
from gaia_core.cognition import approval_reply


class _Sessions:
    def __init__(self):
        self.meta = {}

    def get_session_meta(self, sid, key, default=None):
        return self.meta.get((sid, key), default)

    def set_session_meta(self, sid, key, value):
        self.meta[(sid, key)] = value


def test_routed_write_parks_in_mcp_and_never_writes_locally(monkeypatch):
    calls = []
    monkeypatch.setattr(ac.mcp_client, "call_jsonrpc",
                        lambda method, params: calls.append((method, dict(params))) or
                        {"ok": False, "pending_approval": True, "action_id": "a-9"})
    monkeypatch.setattr(ac.mcp_client, "ai_write",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("wrote locally")))
    core = object.__new__(ac.AgentCore)
    core.config = SimpleNamespace(constants={"TOOL_ROUTING": {"ALLOW_WRITE_TOOLS": True}})
    core.session_manager = _Sessions()
    tool = ac.SelectedTool(tool_name="file", params={"action": "write", "path": "/approval_test.txt",
                                                     "content": "hello"}, selection_confidence=0.9)
    out = core._execute_mcp_tool(tool, session_id="dm1")
    assert calls == [("file", {"action": "write", "path": "/approval_test.txt", "content": "hello"})]
    assert out.output["pending_approval"] and "authenticator code" in out.output["message"]
    assert core.session_manager.get_session_meta("dm1", approval_reply.KEY)["action_id"] == "a-9"

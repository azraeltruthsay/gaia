"""qgws: a tool's input error (bad path, missing parameter) is its own failed
result, not a server crash. file(list) on a guessed path raised ValueError
('… is not a directory'), MCP logged GAIA-MCP-020 'Tool execution crashed'
and gaia-core got '500 Server Error' instead of the message."""

import pytest
from fastapi.testclient import TestClient

from gaia_mcp import server


@pytest.fixture
def client(monkeypatch):
    async def fake_limb(method, params, approval_store=None, pre_approved=False):
        if method == "list_dir":
            raise ValueError("/knowledge/dev_logs is not a directory")
        if method == "read_file":
            raise FileNotFoundError("/knowledge/nope.md")
        raise RuntimeError("boom")
    monkeypatch.setattr(server, "execute_limb", fake_limb)
    return TestClient(server.app)


_PARAMS = {"list_dir": {"path": "/knowledge/dev_logs"}, "read_file": {"path": "/knowledge/nope.md"}}


def _call(client, method):
    return client.post("/jsonrpc", json={"jsonrpc": "2.0", "method": method,
                                         "params": _PARAMS.get(method, {}), "id": 1})


@pytest.mark.parametrize("method,msg", [("list_dir", "is not a directory"), ("read_file", "nope.md")])
def test_input_errors_come_back_as_the_tools_failed_result(client, method, msg):
    r = _call(client, method)
    assert r.status_code == 200
    result = r.json()["result"]
    assert result["ok"] is False and msg in result["error"]
    assert result["gaia_code"] == "GAIA-MCP-021"


def test_a_real_crash_is_still_a_500(client):
    r = _call(client, "anything_else")
    assert r.status_code == 500
    assert r.json()["error"]["data"]["gaia_code"] == "GAIA-MCP-020"

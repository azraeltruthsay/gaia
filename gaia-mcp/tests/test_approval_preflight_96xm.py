"""A human is only asked to approve a call that could run (96xm, 2026-11-16).

A write to /approval_test.txt was approved with a real authenticator code,
then refused by the path allowlist; the approval was spent and the chat said
the code was wrong.
"""
import asyncio

from gaia_mcp import server
from gaia_mcp.tools import preflight

from .conftest import totp_code


def test_write_outside_the_volumes_fails_preflight(approval_store):
    problem = preflight("file", {"action": "write", "path": "/approval_test.txt", "content": "hi"},
                        approval_store)
    assert problem and "restricted to" in problem


def test_write_in_the_sandbox_passes_preflight(approval_store):
    assert preflight("file", {"action": "write", "path": "approval_test.txt", "content": "hi"},
                     approval_store) is None
    assert preflight("write_file", {"path": "/sandbox/x.txt"}, approval_store) is None


def test_blast_shield_runs_in_preflight(approval_store):
    assert preflight("shell", {"action": "run", "command": "sudo reboot"}, approval_store)


def test_preflight_does_not_change_the_callers_params(approval_store):
    params = {"action": "write", "path": "/sandbox/x.txt"}
    preflight("file", params, approval_store)
    assert params["action"] == "write"


class _Req:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


def test_request_for_a_call_that_cannot_run_asks_nobody(approval_store, monkeypatch):
    monkeypatch.setattr(server, "approval_store", approval_store)
    resp = asyncio.run(server.request_approval_endpoint(_Req(
        {"method": "file", "params": {"action": "write", "path": "/etc/x", "content": ""}})))
    assert resp.status_code == 422
    assert approval_store.list_pending() == []


def test_a_failure_after_a_right_code_is_reported_as_approved(approval_store, totp_secret, monkeypatch):
    monkeypatch.setattr(server, "approval_store", approval_store)
    action_id, _, _ = approval_store.create_pending("file", {"action": "write", "path": "/nope/x"},
                                                    allow_pending=True)

    async def boom(**kw):
        raise ValueError("Path not allowed")
    monkeypatch.setattr(server, "execute_limb", boom)
    out = asyncio.run(server.approve_action_endpoint(_Req(
        {"action_id": action_id, "approval": totp_code(totp_secret)})))
    assert out == {"ok": False, "approved": True, "method": "file", "error": "Path not allowed"}

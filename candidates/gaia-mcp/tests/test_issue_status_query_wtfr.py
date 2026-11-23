"""wtfr: issue(status) with a status and no issue named asks which issues
have that status; with an id it still sets one."""
import asyncio

from gaia_mcp import tools


def _call(params):
    return asyncio.run(tools.execute_limb("issue", dict(params), None, pre_approved=False))


def test_status_without_id_lists(tmp_path, monkeypatch):
    monkeypatch.setenv("GAIA_ISSUES_DIR", str(tmp_path))
    from gaia_common.utils import issue_ledger
    filed = issue_ledger.file_issue("Mid-reply tool calls", "a call leaked into chat", ["line 1"])
    out = _call({"action": "status", "status": "open"})
    assert out.get("ok", True) is not False and "no issue" not in str(out)
    assert filed.get("id", "") in str(out) or "Mid-reply" in str(out)


def test_status_with_id_still_sets(tmp_path, monkeypatch):
    monkeypatch.setenv("GAIA_ISSUES_DIR", str(tmp_path))
    out = _call({"action": "status", "id": "nope", "status": "investigating"})
    assert out.get("ok") is False and "no issue" in out.get("error", "")

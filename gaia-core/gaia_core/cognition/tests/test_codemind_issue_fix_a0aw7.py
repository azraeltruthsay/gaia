"""a0aw.7: an issue she filed, naming a function, reaches a CodeMind fix; a fix
to a vital organ waits for Azrael instead of self-deploying."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from gaia_common.utils import issue_ledger as L
from gaia_core.cognition.sleep_task_scheduler import SleepTaskScheduler

FUNC = '''    @staticmethod
    def _should_use_tool_routing(text):
        exec_indicators = ["run ", "cat "]
        return any(x in text for x in exec_indicators)
'''
PATCH = '''<<<<<<< SEARCH
        exec_indicators = ["run ", "cat "]
=======
        exec_indicators = ["run "]
>>>>>>> REPLACE'''


def _setup(tmp_path, monkeypatch, filename):
    repo, shared = tmp_path / "repo", tmp_path / "shared"
    target = repo / "candidates" / "gaia-core" / "gaia_core" / "cognition" / filename
    target.parent.mkdir(parents=True)
    filler = "".join(f"def filler_{i}():\n    return {i}\n\n\n" for i in range(1500))  # > 30000 chars
    target.write_text(f"class AgentCore:\n{FUNC}\n\n{filler}")
    monkeypatch.setenv("GAIA_REPO_ROOT", str(repo))
    monkeypatch.setenv("GAIA_ISSUES_DIR", str(shared / "issues"))
    monkeypatch.setenv("SHARED_DIR", str(shared))
    monkeypatch.setenv("CODEMIND_CHANGELOG_PATH", str(tmp_path / "changelog.jsonl"))
    L.record_shown(["tool routing fired (intent brainstorming)"])
    iid = L.file_issue("Haiku routed to a tool", "a haiku about a cat went through tool routing",
                       ["tool routing fired (intent brainstorming)"],
                       suspected_cause="'cat ' in exec_indicators in AgentCore._should_use_tool_routing")["id"]
    from gaia_common.utils import codemind_detector, codemind_tests
    monkeypatch.setattr(codemind_detector, "consume_detections", lambda limit=3: [])
    monkeypatch.setattr(codemind_tests, "run_test_step", lambda path, **kw: {
        "ok": True, "service": "gaia-core", "paths": ["tests/test_x.py"], "summary": "18 passed"})
    import gaia_core.cognition.thought_seed as ts
    monkeypatch.setattr(ts, "list_unreviewed_seeds", lambda: [])
    s = SleepTaskScheduler.__new__(SleepTaskScheduler)
    s.config = SimpleNamespace(constants={"CODEMIND": {
        "enabled": True, "dry_run": False, "patch_mode": True, "self_deploy": True,
        "validation": {"py_compile": True, "ruff": False, "ast_parse": True, "pytest": True}}})
    s.check_interrupted = lambda: None
    prompts = []
    s._codemind_propose = MagicMock(side_effect=lambda p: prompts.append(p) or PATCH)
    return s, target, iid, shared, prompts


def test_issue_to_fix_on_a_large_vital_file(tmp_path, monkeypatch):
    s, target, iid, shared, prompts = _setup(tmp_path, monkeypatch, "agent_core.py")
    s._run_codemind_cycle()

    text = target.read_text()
    assert 'exec_indicators = ["run "]' in text and "filler_1499" in text  # patched, rest intact
    assert len(prompts) == 1 and "filler_" not in prompts[0]               # shown only her function
    assert "tool routing fired" in prompts[0]                              # her evidence in the prompt
    issue = L.get_issue(iid)["issue"]
    assert issue["status"] == "proposed"
    assert "18 passed" in issue["notes"][-1]["text"] and "vital organ" in issue["notes"][-1]["text"]
    # a vital organ is never self-deployed
    assert not list((shared / "doctor" / "restart_requests").glob("*.json"))
    [queued] = list((shared / "doctor" / "architect_queue").glob("*.json"))
    m = json.loads(queued.read_text())
    assert m["status"] == "awaiting_architect" and m["vital_organs"] == ["agent_core.py"]
    assert m["promote_files"] == ["candidates/gaia-core/gaia_core/cognition/agent_core.py"]


def test_non_vital_fix_self_deploys(tmp_path, monkeypatch):
    s, target, iid, shared, prompts = _setup(tmp_path, monkeypatch, "routing_rules.py")
    L.note_issue(iid, "it is in routing_rules", file="gaia_core/cognition/routing_rules.py")
    # no function named: a large file can't be shown, so nothing is tried
    s._run_codemind_cycle()
    issue = L.get_issue(iid)["issue"]
    assert "name the function" in issue["notes"][-1]["text"]
    L.note_issue(iid, "the indicator list", file="AgentCore._should_use_tool_routing")
    s._run_codemind_cycle()
    assert L.get_issue(iid)["issue"]["status"] == "proposed"
    assert len(list((shared / "doctor" / "restart_requests").glob("*.json"))) == 1

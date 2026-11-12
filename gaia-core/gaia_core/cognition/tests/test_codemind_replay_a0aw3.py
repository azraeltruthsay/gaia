"""a0aw.3: replay the offending turn on gaia-core-candidate before trusting a
fix for one of her issues — a non-vital fix self-deploys once it's
confirmed; a vital organ still always waits for Azrael regardless."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from gaia_common.utils import issue_ledger as L
from gaia_core.cognition.sleep_task_scheduler import SleepTaskScheduler

FIX = "def f():\n    return 2\n"
ORIGINAL = "def f():\n    return 1\n"


def _setup(tmp_path, monkeypatch, filename):
    repo, shared = tmp_path / "repo", tmp_path / "shared"
    target = repo / "candidates" / "gaia-core" / "gaia_core" / filename
    target.parent.mkdir(parents=True)
    target.write_text(ORIGINAL)
    monkeypatch.setenv("GAIA_REPO_ROOT", str(repo))
    monkeypatch.setenv("GAIA_ISSUES_DIR", str(shared / "issues"))
    monkeypatch.setenv("SHARED_DIR", str(shared))
    monkeypatch.setenv("CODEMIND_CHANGELOG_PATH", str(tmp_path / "changelog.jsonl"))
    L.record_shown(["tool routing fired (intent brainstorming)"])
    iid = L.file_issue("Haiku routed to a tool", "a haiku about a cat went through tool routing",
                       ["tool routing fired (intent brainstorming)"],
                       suspected_cause=f"f() in {filename}", playbook="ts-misrouted-request",
                       original_prompt="Write a haiku about a cat watching snow.")["id"]
    L.note_issue(iid, "it's f()", file=f"gaia_core/{filename}")
    from gaia_common.utils import codemind_detector, codemind_tests
    monkeypatch.setattr(codemind_detector, "consume_detections", lambda limit=3: [])
    monkeypatch.setattr(codemind_tests, "run_test_step", lambda path, **kw: {
        "ok": True, "service": "gaia-core", "paths": ["tests/test_f.py"], "summary": "1 passed"})
    import gaia_core.cognition.thought_seed as ts
    monkeypatch.setattr(ts, "list_unreviewed_seeds", lambda: [])
    s = SleepTaskScheduler.__new__(SleepTaskScheduler)
    s.config = SimpleNamespace(constants={"CODEMIND": {
        "enabled": True, "dry_run": False, "patch_mode": False, "self_deploy": True,
        "write_regression_test": False, "replay_check": True,
        "validation": {"py_compile": True, "ruff": False, "ast_parse": True, "pytest": True}}})
    s.check_interrupted = lambda: None
    s._codemind_propose = MagicMock(return_value=FIX)
    return s, target, iid, shared


def test_non_vital_self_deploys_once_replay_confirms(tmp_path, monkeypatch):
    s, target, iid, shared = _setup(tmp_path, monkeypatch, "routing_rules.py")
    from gaia_common.utils import codemind_replay
    monkeypatch.setattr(codemind_replay, "replay_turn", lambda prompt, **kw: {
        "ok": True, "symptom_seen": False, "detail": "no disallowed line seen within the poll window"})

    s._run_codemind_cycle()

    assert target.read_text() == FIX
    issue = L.get_issue(iid)["issue"]
    assert issue["status"] == "proposed"
    assert "replay confirmed" in issue["notes"][-1]["text"]
    assert "self-deploying" in issue["notes"][-1]["text"]
    assert "waiting for Azrael" not in issue["notes"][-1]["text"]
    assert not list((shared / "doctor" / "architect_queue").glob("*.json"))
    [deployed] = list((shared / "doctor" / "restart_requests").glob("*.json"))
    m = json.loads(deployed.read_text())
    assert "awaiting_architect" != m.get("status")


def test_non_vital_replay_still_shows_symptom_is_restored(tmp_path, monkeypatch):
    s, target, iid, shared = _setup(tmp_path, monkeypatch, "routing_rules.py")
    from gaia_common.utils import codemind_replay
    monkeypatch.setattr(codemind_replay, "replay_turn", lambda prompt, **kw: {
        "ok": True, "symptom_seen": True, "detail": "saw: Tool routing triggered for intent: brainstorming"})

    s._run_codemind_cycle()

    assert target.read_text() == ORIGINAL  # restored — the symptom was still there on replay
    issue = L.get_issue(iid)["issue"]
    assert issue["status"] != "proposed"
    assert "kept nothing" in issue["notes"][-1]["text"]
    assert not list((shared / "doctor" / "architect_queue").glob("*.json"))
    assert not list((shared / "doctor" / "restart_requests").glob("*.json"))


def test_vital_organ_still_waits_for_azrael_even_when_replay_confirms(tmp_path, monkeypatch):
    s, target, iid, shared = _setup(tmp_path, monkeypatch, "agent_core.py")
    from gaia_common.utils import codemind_replay
    monkeypatch.setattr(codemind_replay, "replay_turn", lambda prompt, **kw: {
        "ok": True, "symptom_seen": False, "detail": "no disallowed line seen within the poll window"})

    s._run_codemind_cycle()

    assert target.read_text() == FIX
    issue = L.get_issue(iid)["issue"]
    assert "replay confirmed" in issue["notes"][-1]["text"]
    assert "waiting for Azrael" in issue["notes"][-1]["text"] and "vital organ" in issue["notes"][-1]["text"]
    assert not list((shared / "doctor" / "restart_requests").glob("*.json"))
    [queued] = list((shared / "doctor" / "architect_queue").glob("*.json"))
    m = json.loads(queued.read_text())
    assert m["status"] == "awaiting_architect" and m["vital_organs"] == ["agent_core.py"]
    assert "replay" in json.dumps(m["tests_run"])  # the evidence passed to build_manifest_for_files

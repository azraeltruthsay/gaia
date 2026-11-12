"""a0aw.8: CodeMind writes a regression test alongside the fix, and keeps it
only if it fails on the untouched original — proof, not theater."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gaia_core.cognition.sleep_task_scheduler import SleepTaskScheduler

ORIGINAL = "def f():\n    return 1\n"
FIX = "def f():\n    return 2\n"
GOOD_TEST = "import gaia_core.mod as mod\n\ndef test_f_returns_2():\n    assert mod.f() == 2\n"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    target = repo / "candidates" / "gaia-core" / "gaia_core" / "mod.py"
    target.parent.mkdir(parents=True)
    target.write_text(ORIGINAL)
    monkeypatch.setenv("GAIA_REPO_ROOT", str(repo))
    monkeypatch.setenv("CODEMIND_CHANGELOG_PATH", str(tmp_path / "changelog.jsonl"))
    from gaia_common.utils import codemind_detector, codemind_tests
    monkeypatch.setattr(codemind_detector, "consume_detections", lambda limit=3: [
        {"issue_type": "import_error", "file_path": str(target), "description": "f returns the wrong value"}])
    import gaia_core.cognition.thought_seed as ts
    monkeypatch.setattr(ts, "list_unreviewed_seeds", lambda: [])
    s = SleepTaskScheduler.__new__(SleepTaskScheduler)
    s.config = SimpleNamespace(constants={"CODEMIND": {
        "enabled": True, "dry_run": False, "patch_mode": False, "self_deploy": False,
        "write_regression_test": True, "replay_check": False,
        "validation": {"py_compile": True, "ruff": False, "ast_parse": True, "pytest": True}}})
    s.check_interrupted = lambda: None
    return s, target, codemind_tests, monkeypatch


def _propose(fix_response, test_response):
    """Tell apart the fix-prompt call from the regression-test-prompt call by
    their distinct instruction text (build_fix_prompt vs
    build_regression_test_prompt), the same way the real prompts differ."""
    def _side_effect(prompt):
        if "regression test" in prompt:
            return test_response
        return fix_response
    return MagicMock(side_effect=_side_effect)


def test_kept_when_it_fails_on_the_original(setup):
    s, target, codemind_tests, monkeypatch = setup
    s._codemind_propose = _propose(FIX, GOOD_TEST)
    run_tests_calls = []

    def fake_run_tests(service, paths, **kw):
        run_tests_calls.append((service, paths))
        return {"ok": False, "summary": "1 failed"}  # fails against the still-buggy original

    verify_calls = []

    def fake_run_test_step(path, extra=None, **kw):
        verify_calls.append(extra)
        return {"ok": True, "service": "gaia-core", "paths": ["tests/test_mod.py"], "summary": "2 passed"}

    monkeypatch.setattr(codemind_tests, "run_tests", fake_run_tests)
    monkeypatch.setattr(codemind_tests, "run_test_step", fake_run_test_step)

    s._run_codemind_cycle()

    assert target.read_text() == FIX
    assert len(run_tests_calls) == 1  # the fails-on-original check ran once
    assert verify_calls and verify_calls[0] is not None  # VERIFY got the regression test as `extra`
    written = target.parent.parent / "tests" / verify_calls[0][0].rsplit("/", 1)[-1]
    assert written.exists() and written.read_text() == GOOD_TEST


def test_discarded_when_it_passes_on_the_original(setup):
    s, target, codemind_tests, monkeypatch = setup
    s._codemind_propose = _propose(FIX, GOOD_TEST)
    monkeypatch.setattr(codemind_tests, "run_tests",
                        lambda service, paths, **kw: {"ok": True, "summary": "2 passed"})
    verify_calls = []

    def fake_run_test_step(path, extra=None, **kw):
        verify_calls.append(extra)
        return {"ok": True, "service": "gaia-core", "paths": [], "summary": "0 passed"}
    monkeypatch.setattr(codemind_tests, "run_test_step", fake_run_test_step)

    s._run_codemind_cycle()

    assert target.read_text() == FIX  # the fix itself is unaffected
    assert verify_calls == [None]     # nothing untrustworthy was handed to VERIFY
    assert not list((target.parent.parent / "tests").glob("test_codemind_*.py"))  # no orphan file


def test_declined_test_never_blocks_the_fix(setup):
    s, target, codemind_tests, monkeypatch = setup
    s._codemind_propose = _propose(FIX, "NO_TEST: can't isolate this reliably")
    run_tests_called = []
    monkeypatch.setattr(codemind_tests, "run_tests", lambda *a, **kw: run_tests_called.append(1))
    verify_calls = []

    def fake_run_test_step(path, extra=None, **kw):
        verify_calls.append(extra)
        return {"ok": True, "service": "gaia-core", "paths": [], "summary": "0 passed"}
    monkeypatch.setattr(codemind_tests, "run_test_step", fake_run_test_step)

    s._run_codemind_cycle()

    assert target.read_text() == FIX
    assert not run_tests_called  # a decline never even reaches the fails-on-original check
    assert verify_calls == [None]

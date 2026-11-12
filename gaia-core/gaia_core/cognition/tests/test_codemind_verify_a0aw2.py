"""a0aw.2: a CodeMind fix is kept only when the module's tests pass in the
candidate; failed tests restore the file and send it back to PROPOSE once."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gaia_core.cognition.sleep_task_scheduler import SleepTaskScheduler

ORIGINAL = "def f():\n    return 1\n"
FIRST = "def f():\n    return 2\n"
SECOND = "def f():\n    return 3\n"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    target = tmp_path / "candidates" / "gaia-core" / "gaia_core" / "mod.py"
    target.parent.mkdir(parents=True)
    target.write_text(ORIGINAL)
    monkeypatch.setenv("CODEMIND_CHANGELOG_PATH", str(tmp_path / "changelog.jsonl"))
    from gaia_common.utils import codemind_detector, codemind_tests
    monkeypatch.setattr(codemind_detector, "consume_detections", lambda limit=3: [
        {"issue_type": "import_error", "file_path": str(target), "description": "f returns the wrong value"}])
    import gaia_core.cognition.thought_seed as ts
    monkeypatch.setattr(ts, "list_unreviewed_seeds", lambda: [])
    s = SleepTaskScheduler.__new__(SleepTaskScheduler)
    s.config = SimpleNamespace(constants={"CODEMIND": {
        "enabled": True, "dry_run": False, "patch_mode": False, "self_deploy": False, "test_retries": 1,
        # a0aw.8/a0aw.3 are exercised by their own dedicated tests; this
        # fixture's call-counting mocks don't expect the extra propose() call
        # the regression-test step would make.
        "write_regression_test": False, "replay_check": False,
        "validation": {"py_compile": True, "ruff": False, "ast_parse": True, "pytest": True}}})
    s.check_interrupted = lambda: None
    return s, target, codemind_tests, monkeypatch


def test_failed_tests_restore_and_retry_then_keep(setup):
    s, target, codemind_tests, monkeypatch = setup
    prompts = []
    s._codemind_propose = MagicMock(side_effect=lambda p: prompts.append(p) or (FIRST if len(prompts) == 1 else SECOND))
    seen = []

    def fake_step(path, **kw):
        seen.append(open(path).read())
        return ({"ok": False, "service": "gaia-core", "paths": ["tests/test_mod.py"],
                 "summary": "1 failed", "output": "E   assert 2 == 3"} if len(seen) == 1 else
                {"ok": True, "service": "gaia-core", "paths": ["tests/test_mod.py"], "summary": "4 passed"})
    monkeypatch.setattr(codemind_tests, "run_test_step", fake_step)

    s._run_codemind_cycle()

    assert seen == [FIRST, SECOND]           # each proposal was tested as written to the candidate
    assert target.read_text() == SECOND      # the passing one is kept
    assert "1 failed" in prompts[1] and "assert 2 == 3" in prompts[1]  # the failure went back to PROPOSE
    assert (target.parent / "mod.py.bak").read_text() == ORIGINAL


def test_tests_keep_failing_leaves_original(setup):
    s, target, codemind_tests, monkeypatch = setup
    s._codemind_propose = MagicMock(return_value=FIRST)
    monkeypatch.setattr(codemind_tests, "run_test_step", lambda path, **kw: {
        "ok": False, "service": "gaia-core", "paths": [], "summary": "no test covers this module", "output": ""})
    s._run_codemind_cycle()
    assert target.read_text() == ORIGINAL
    assert s._codemind_propose.call_count == 2


def test_tests_off_applies_after_validation(setup):
    s, target, codemind_tests, monkeypatch = setup
    s.config.constants["CODEMIND"]["validation"]["pytest"] = False
    s._codemind_propose = MagicMock(return_value=FIRST)
    step = MagicMock()
    monkeypatch.setattr(codemind_tests, "run_test_step", step)
    s._run_codemind_cycle()
    assert target.read_text() == FIRST and not step.called

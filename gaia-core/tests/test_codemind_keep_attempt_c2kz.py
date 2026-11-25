"""A CodeMind attempt is kept whatever the tests said: the diff, the
regression test and the full report (c2kz trial)."""
from gaia_core.cognition.sleep_task_scheduler import SleepTaskScheduler


def test_attempt_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEMIND_ATTEMPTS_DIR", str(tmp_path / "attempts"))
    test_file = tmp_path / "test_codemind_gi_x.py"
    test_file.write_text("def test_x():\n    assert False\n")
    out = SleepTaskScheduler._codemind_keep_attempt(
        "cm-1", "gi-9168e4", 0, "/gaia/GAIA_Project/candidates/gaia-core/gaia_core/cognition/m.py",
        "a = 1\n", "a = 2\n",
        {"summary": "1 failed, 9 passed in 1.1s", "paths": ["t.py"], "output": "FAILED t.py::test_y - assert 2 == 1"},
        test_file)
    d = tmp_path / "attempts" / "cm-1" / "gi-9168e4_1"
    assert out == str(d)
    diff = (d / "patch.diff").read_text()
    assert "-a = 1" in diff and "+a = 2" in diff and "b/gaia-core/gaia_core/cognition/m.py" in diff
    assert (d / "test_codemind_gi_x.py").read_text().startswith("def test_x")
    assert "FAILED t.py::test_y" in (d / "tests.txt").read_text()


def test_unwritable_place_is_not_an_error(monkeypatch):
    monkeypatch.setenv("CODEMIND_ATTEMPTS_DIR", "/proc/no/such/place")
    assert SleepTaskScheduler._codemind_keep_attempt("c", "h", 0, "/x/m.py", "", "", {}, None) == ""


def test_a_rejected_attempt_is_kept_without_tests(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEMIND_ATTEMPTS_DIR", str(tmp_path / "a"))
    out = SleepTaskScheduler._codemind_keep_attempt(
        "cm-2", "gi-2fbe7e", 0, "/x/candidates/gaia-core/m.py", "a\n", "b\n",
        {"summary": "not applied: change ratio 80.4% exceeds max 50%"}, None)
    assert "not applied" in (tmp_path / "a" / "cm-2" / "gi-2fbe7e_1" / "tests.txt").read_text()
    assert out

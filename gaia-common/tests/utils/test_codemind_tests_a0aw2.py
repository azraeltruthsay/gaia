"""a0aw.2: CodeMind's TEST step picks the touched module's tests and runs them in the candidate."""

import io
import json

from gaia_common.utils import codemind_tests as T


def _repo(tmp_path):
    core = tmp_path / "candidates" / "gaia-core"
    (core / "gaia_core" / "cognition" / "tests").mkdir(parents=True)
    (core / "tests").mkdir()
    (core / "gaia_core" / "cognition" / "agent_core.py").write_text("x = 1\n")
    (core / "gaia_core" / "cognition" / "tests" / "test_agent_core_routing.py").write_text("def test_a(): pass\n")
    (core / "tests" / "test_haiku.py").write_text("from gaia_core.cognition.agent_core import AgentCore\n")
    (core / "tests" / "test_other.py").write_text("from gaia_core.cognition import agent_core\n")
    (core / "tests" / "test_domain.py").write_text("import json\n")  # 'main'-style false match guard
    (core / "tests" / "test_unrelated.py").write_text("from gaia_core.cognition.agent_core_extra import y\n")
    return tmp_path


def test_locate(tmp_path):
    root = _repo(tmp_path)
    f = root / "candidates/gaia-core/gaia_core/cognition/agent_core.py"
    assert T.locate(str(f), root)[0::2] == ("gaia-core", "gaia_core.cognition.agent_core")
    assert T.locate(str(root / "gaia-core/gaia_core/x.py"), root) is None  # production, not candidate
    assert T.locate(str(root / "candidates/gaia-web/gaia_web/x.py"), root) is None


def test_finds_named_and_importing_tests(tmp_path):
    root = _repo(tmp_path)
    service, paths = T.find_tests(str(root / "candidates/gaia-core/gaia_core/cognition/agent_core.py"), root)
    assert service == "gaia-core"
    assert paths[0] == "gaia_core/cognition/tests/test_agent_core_routing.py"
    assert set(paths[1:]) == {"tests/test_haiku.py", "tests/test_other.py"}


def test_regression_test_included_first(tmp_path):
    root = _repo(tmp_path)
    _, paths = T.find_tests(str(root / "candidates/gaia-core/gaia_core/cognition/agent_core.py"), root,
                            extra=["tests/test_mjcg_regression.py"])
    assert paths[0] == "tests/test_mjcg_regression.py"


def test_no_tests_is_not_a_pass():
    r = T.run_tests("gaia-core", [])
    assert not r["ok"] and "regression test" in r["summary"]


def test_runs_through_orchestrator(monkeypatch):
    sent = {}

    def fake_urlopen(req, timeout=None):
        sent["url"], sent["body"] = req.full_url, json.loads(req.data)
        return io.BytesIO(json.dumps({"ok": False, "summary": "1 failed, 3 passed", "output": "E assert"}).encode())

    monkeypatch.setattr(T.urllib.request, "urlopen", fake_urlopen)
    r = T.run_tests("gaia-core", ["tests/test_haiku.py"], endpoint="http://orch:6410")
    assert sent["url"] == "http://orch:6410/candidate/test"
    assert sent["body"]["paths"] == ["tests/test_haiku.py"] and sent["body"]["service"] == "gaia-core"
    assert r == {"ok": False, "summary": "1 failed, 3 passed", "output": "E assert"}


def test_unreachable_runner_fails(monkeypatch):
    def boom(req, timeout=None):
        raise OSError("connection refused")
    monkeypatch.setattr(T.urllib.request, "urlopen", boom)
    r = T.run_tests("gaia-core", ["tests/test_haiku.py"], endpoint="http://orch:6410")
    assert not r["ok"] and "could not run" in r["summary"]

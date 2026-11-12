"""a0aw.7: her issues reach CodeMind once she has said where the bug is."""

from gaia_common.utils import codemind_issues as C
from gaia_common.utils import issue_ledger as L
from gaia_common.utils.tools_registry import TOOLS

AGENT = '''class AgentCore:
    @staticmethod
    def _should_use_tool_routing(text):
        exec_indicators = ["run ", "cat "]
        return any(x in text for x in exec_indicators)

    def run_turn(self):
        return 1
'''


def _repo(tmp_path):
    pkg = tmp_path / "candidates" / "gaia-core" / "gaia_core" / "cognition"
    pkg.mkdir(parents=True)
    (pkg / "agent_core.py").write_text(AGENT)
    (pkg / "tests").mkdir()
    (pkg / "tests" / "test_agent_core.py").write_text("def _should_use_tool_routing(): pass\n")
    common = tmp_path / "candidates" / "gaia-common" / "gaia_common" / "utils"
    common.mkdir(parents=True)
    (common / "issue_ledger.py").write_text("def file_issue(): pass\n")
    return tmp_path


def _issue(base, **kw):
    L.record_shown(["tool routing fired (intent brainstorming)"], base=base)
    r = L.file_issue("Haiku routed to a tool", "a haiku about a cat went through tool routing",
                     ["tool routing fired (intent brainstorming)"], base=base, **kw)
    return r["id"]


def test_resolve(tmp_path):
    root = _repo(tmp_path)
    agent = root / "candidates/gaia-core/gaia_core/cognition/agent_core.py"
    assert C.resolve("AgentCore._should_use_tool_routing", root) == (agent, "_should_use_tool_routing")
    assert C.resolve("gaia_core/cognition/agent_core.py", root) == (agent, "")
    assert C.resolve("agent_core.py", root) == (agent, "")
    assert C.resolve("issue_ledger.py", root)[0].name == "issue_ledger.py"
    assert C.resolve("no_such_function", root) is None
    for form in ("gaia-core/gaia_core/cognition/agent_core.py:_should_use_tool_routing",
                 "agent_core.py::AgentCore._should_use_tool_routing", "agent_core.py#_should_use_tool_routing()"):
        assert C.resolve(form, root) == (agent, "_should_use_tool_routing"), form
    assert C.resolve("agent_core.py:not_there", root) is None
    assert C.resolve("../../etc/passwd.py", root) is None


def test_refs_in_her_words():
    refs = C.refs_in("The 'cat ' entry in AgentCore._should_use_tool_routing() matches; see `exec_indicators`.")
    assert "AgentCore._should_use_tool_routing" in refs and "exec_indicators" in refs


def test_open_issue_naming_a_place_becomes_a_detection(tmp_path):
    root, base = _repo(tmp_path), tmp_path / "issues"
    iid = _issue(base, suspected_cause="'cat ' in exec_indicators, AgentCore._should_use_tool_routing")
    [d] = C.detections(base, root)
    assert d["metadata"]["issue_id"] == iid and d["metadata"]["symbol"] == "_should_use_tool_routing"
    assert d["file_path"].endswith("agent_core.py") and "tool routing fired" in d["description"]


def test_detection_carries_playbook_and_original_prompt(tmp_path):
    """a0aw.3: a later replay step needs these on the detection metadata,
    not just the issue — empty when the issue has neither."""
    root, base = _repo(tmp_path), tmp_path / "issues"
    iid = _issue(base, suspected_cause="AgentCore._should_use_tool_routing",
                playbook="ts-misrouted-request",
                original_prompt="Write a haiku about a cat watching snow.")
    [d] = C.detections(base, root)
    assert d["metadata"]["playbook"] == "ts-misrouted-request"
    assert d["metadata"]["original_prompt"] == "Write a haiku about a cat watching snow."

    L.record_shown(["selector: no tool needed"], base=base)
    iid2 = L.file_issue("Another misroute entirely", "also routed wrongly, a different one",
                        ["selector: no tool needed"], suspected_cause="AgentCore.run_turn", base=base)["id"]
    [d2] = [x for x in C.detections(base, root) if x["metadata"]["issue_id"] == iid2]
    assert d2["metadata"]["playbook"] == "" and d2["metadata"]["original_prompt"] == ""
    assert iid != iid2


def test_no_place_no_detection_until_she_names_one(tmp_path):
    root, base = _repo(tmp_path), tmp_path / "issues"
    iid = _issue(base, suspected_cause="the routing trigger misread the request")
    assert C.detections(base, root) == []
    L.note_issue(iid, "found it", base=base, file="gaia_core/cognition/agent_core.py")
    assert L.get_issue(iid, base=base)["issue"]["file"] == "gaia_core/cognition/agent_core.py"
    [d] = C.detections(base, root)
    assert d["metadata"]["symbol"] == ""


def test_tried_once_waits_for_something_new(tmp_path):
    root, base = _repo(tmp_path), tmp_path / "issues"
    iid = _issue(base, suspected_cause="AgentCore._should_use_tool_routing")
    [d] = C.detections(base, root)
    C.record_attempt(d, kept=False, why="Tests failed: 2 failed", base=base)
    issue = L.get_issue(iid, base=base)["issue"]
    assert "kept nothing" in issue["notes"][-1]["text"] and "1 left" in issue["notes"][-1]["text"]
    assert C.detections(base, root) == []  # CodeMind's own note is not something new from her
    L.note_issue(iid, "the indicator list also has 'ls '", base=base)
    [d] = C.detections(base, root)
    C.record_attempt(d, kept=False, why="declined", base=base)
    assert C.detections(base, root) == []  # MAX_ATTEMPTS reached


def test_kept_fix_marks_proposed(tmp_path, monkeypatch):
    root, base = _repo(tmp_path), tmp_path / "issues"
    monkeypatch.setenv("GAIA_REPO_ROOT", str(root))
    iid = _issue(base, suspected_cause="AgentCore._should_use_tool_routing")
    [d] = C.detections(base, root)
    C.record_attempt(d, kept=True, why="candidate tests: 18 passed", base=base)
    issue = L.get_issue(iid, base=base)["issue"]
    assert issue["status"] == "proposed" and "18 passed" in issue["notes"][-1]["text"]
    assert "candidates/gaia-core/gaia_core/cognition/agent_core.py" in issue["notes"][-1]["text"]
    assert C.detections(base, root) == []


def test_closed_issue_not_sent(tmp_path):
    root, base = _repo(tmp_path), tmp_path / "issues"
    iid = _issue(base, suspected_cause="AgentCore._should_use_tool_routing")
    L.set_status(iid, "wontfix", base=base)
    assert C.detections(base, root) == []


def test_excerpt_is_the_named_function():
    text = C.excerpt(AGENT, "_should_use_tool_routing")
    assert text.startswith("    @staticmethod\n    def _should_use_tool_routing")
    assert "cat " in text and "run_turn" not in text
    assert C.excerpt(AGENT, "missing") is None and C.excerpt(AGENT, "") is None


def test_tool_schema_takes_file():
    assert "file" in TOOLS["issue_note"]["params"]["properties"]
    assert "file" in TOOLS["issue_status"]["params"]["properties"]

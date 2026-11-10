"""a0aw.9: she looks into an open issue in sleep and names where the bug is."""

from pathlib import Path

import pytest

from gaia_common.utils import codemind_issues as C
from gaia_common.utils import issue_ledger as L
from gaia_core.cognition import issue_investigation as I

AGENT = '''class AgentCore:
    def _should_use_tool_routing(self, text):
        exec_indicators = ["run ", "cat "]
        return any(x in text for x in exec_indicators)

    def run_turn(self):
        return 1
'''
PLAYBOOK = """---
name: ts-misrouted-request
seen_in: [turns]
---
# A request took the wrong path

## Find it
3. If routing fired: the trigger is `AgentCore._should_use_tool_routing`.
"""


@pytest.fixture
def world(tmp_path, monkeypatch):
    repo, base, skills = tmp_path / "repo", tmp_path / "issues", tmp_path / "skills"
    pkg = repo / "candidates" / "gaia-core" / "gaia_core" / "cognition"
    pkg.mkdir(parents=True)
    (pkg / "agent_core.py").write_text(AGENT + "".join(f"def filler_{i}():\n    return {i}\n\n" for i in range(1500)))
    (repo / "candidates" / "gaia-common" / "gaia_common").mkdir(parents=True)
    (skills / "ts-misrouted-request").mkdir(parents=True)
    (skills / "ts-misrouted-request" / "SKILL.md").write_text(PLAYBOOK)
    monkeypatch.setenv("GAIA_REPO_ROOT", str(repo))
    L.record_shown(["tool routing fired (intent brainstorming)"], base=base)
    iid = L.file_issue("Haiku routed to a tool", "a haiku about a cat went through tool routing",
                       ["tool routing fired (intent brainstorming)"], playbook="ts-misrouted-request",
                       suspected_cause="the routing trigger misread it", base=base)["id"]
    return repo, base, skills, iid


def _script(*replies):
    seen = []

    def generate(messages):
        seen.append([dict(m) for m in messages])
        return replies[min(len(seen) - 1, len(replies) - 1)]
    return generate, seen


def test_finds_and_names_the_function(world):
    repo, base, skills, iid = world
    gen, seen = _script("SEARCH: exec_indicators",
                        "READ: gaia-core/gaia_core/cognition/agent_core.py:3",
                        "PLACE: AgentCore._should_use_tool_routing\nNOTE: 'cat ' matches any message about cats.")
    r = I.run(gen, skills_dir=skills, base=base)
    assert r[0]["place"] == "AgentCore._should_use_tool_routing" and r[0]["steps"] == 3
    assert "Find it" in seen[0][1]["content"]                       # her playbook, all of it
    assert "agent_core.py:3:" in seen[1][-1]["content"]              # the search result came back
    assert "line 3 is in _should_use_tool_routing" in seen[2][-1]["content"]
    issue = L.get_issue(iid, base=base)["issue"]
    assert issue["file"] == "AgentCore._should_use_tool_routing" and "cats" in issue["notes"][-1]["text"]
    # the place she named is now one CodeMind can take
    [d] = C.detections(base, repo)
    assert d["metadata"]["symbol"] == "_should_use_tool_routing"
    assert I.candidates(base, repo) == []                            # it names a place now


def test_a_place_she_never_saw_is_refused(world):
    repo, base, skills, iid = world
    gen, seen = _script("PLACE: AgentCore.run_turn", "NOT_FOUND: I guessed.")
    r = I.run(gen, skills_dir=skills, base=base)
    assert r[0]["place"] == "" and "haven't read or searched" in seen[1][-1]["content"]
    assert "couldn't find where: I guessed." in L.get_issue(iid, base=base)["issue"]["notes"][-1]["text"]


def test_large_file_needs_a_function(world):
    repo, base, skills, iid = world
    gen, seen = _script("READ: gaia_core/cognition/agent_core.py", "PLACE: gaia_core/cognition/agent_core.py",
                        "PLACE: _should_use_tool_routing")
    r = I.run(gen, skills_dir=skills, base=base)
    assert "its functions and classes" in seen[1][-1]["content"]     # a large file reads as an outline
    assert "Name the function" in seen[2][-1]["content"]
    assert r[0]["place"] == "_should_use_tool_routing"


def test_repeats_and_no_action_are_answered(world):
    repo, base, skills, iid = world
    gen, seen = _script("SEARCH: exec_indicators", "SEARCH: exec_indicators", "I think it's routing.",
                        "NOT_FOUND: gave up")
    I.run(gen, skills_dir=skills, base=base)
    assert "did this already" in seen[2][-1]["content"]
    assert "No action found" in seen[3][-1]["content"]


def test_search_helps_with_sentences(world):
    repo, base, skills, iid = world
    view = I.CodeView(repo)
    out = view.search("exec_indicators and message.lower().startswith('cat')")
    assert "exec_indicators (" in out and "search one of them alone" in out
    assert "agent_core.py:3:" in view.search("exec_indicators in agent_core.py")
    both = view.search("exec_indicators AND run_turn")
    assert "SEARCH: exec_indicators" in both and "SEARCH: run_turn" in both


def test_not_investigated_twice_without_news(world):
    repo, base, skills, iid = world
    gen, _ = _script("NOT_FOUND: no idea")
    I.run(gen, skills_dir=skills, base=base)
    assert I.candidates(base, repo) == []              # her own investigation note is the fingerprint now
    L.note_issue(iid, "it happens on every cat message", base=base)
    assert [i["id"] for i in I.candidates(base, repo)] == [iid]
    I.run(gen, skills_dir=skills, base=base)
    L.note_issue(iid, "still happening", base=base)
    assert I.candidates(base, repo) == []              # MAX_ATTEMPTS


def test_parse_keeps_order_and_note():
    assert I.parse("SEARCH: a\nREAD: b") == [("SEARCH", "a"), ("READ", "b")]
    assert I.parse("<think>x</think>PLACE: f\nNOTE: why") == [("PLACE", "f", "why")]
    assert I.parse("nothing here") == []


def test_playbook_text_strips_front_matter(tmp_path):
    (tmp_path / "pb").mkdir()
    (tmp_path / "pb" / "SKILL.md").write_text(PLAYBOOK)
    assert I.playbook_text("pb", tmp_path).startswith("# A request")
    assert I.playbook_text("../x", tmp_path) == "" and I.playbook_text("", Path(tmp_path)) == ""

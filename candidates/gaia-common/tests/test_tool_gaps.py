"""Her repeated failed attempts become proposals, and nothing more (yi22)."""
import pytest

from gaia_common.utils import domain_tools, tool_gaps

UNKNOWN = "Unknown action 'find' for domain 'worldbuild'. Available: [...]"
TOOLS = {"worldbuild": {"actions": {"search": {"params": {"query": "string"}}, "list": {"params": {}}}}}


DAY = 86400


def attempt(session, packet, error=UNKNOWN, kind="tool_execution", tool="worldbuild", action="find", ok=False,
            keys=("action", "query"), day=0):
    return {"kind": kind, "session_id": session, "packet_id": packet, "tool": tool, "action": action,
            "ok": ok, "error": error, "param_keys": list(keys), "ts": 1000.0 + day * DAY}


def nothing_known(tool, verb):
    return False


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(tool_gaps, "STORE_DIR", tmp_path)
    return tmp_path


def test_repeats_across_conversations_become_a_proposal():
    rows = [attempt("web_1", "a"), attempt("web_2", "b"), attempt("web_2", "c")]
    p = tool_gaps.mine(rows, nothing_known)[0]
    assert (p["kind"], p["tool"], p["verb"], p["attempts"], p["occasions"], p["outside_tests"]) == \
        ("unknown_action", "worldbuild", "find", 3, 2, 3)
    assert p["params"] == ["query"]


def test_one_long_session_counts_when_it_returns_on_another_day():
    rows = [attempt("gaia_heartbeat_session", "a", day=0), attempt("gaia_heartbeat_session", "b", day=0),
            attempt("gaia_heartbeat_session", "c", day=3)]
    assert tool_gaps.mine(rows, nothing_known)[0]["occasions"] == 2


def test_too_few_or_all_in_one_conversation_is_not_demand():
    assert tool_gaps.mine([attempt("web_1", "a"), attempt("web_2", "b")], nothing_known) == []
    assert tool_gaps.mine([attempt("web_1", p) for p in "abcde"], nothing_known) == []


def test_selection_and_execution_of_one_turn_count_once():
    rows = [attempt("web_1", "a", kind="tool_selection"), attempt("web_1", "a"), attempt("web_2", "b")]
    assert tool_gaps.mine(rows, nothing_known) == []


def test_a_repeated_test_is_one_occasion_a_day_and_is_labelled():
    rows = [attempt(f"smoke-test-{i}", str(i)) for i in range(50)]
    assert tool_gaps.mine(rows, nothing_known) == []
    rows.append(attempt("smoke-test-z", "z", day=1))
    p = tool_gaps.mine(rows, nothing_known)[0]
    assert p["attempts"] == 51 and p["occasions"] == 2 and p["outside_tests"] == 0


def test_noise_verbs_do_not_count():
    noise = "Unknown action 'review' for domain 'file'. Available: []"
    assert tool_gaps.mine([attempt(f"web_{i}", str(i), error=noise) for i in range(5)], nothing_known) == []


@pytest.mark.parametrize("error", ["Path not allowed", "/knowledge/x.md is not a file", ""])
def test_failures_of_a_tool_that_took_the_call_are_not_gaps(error):
    rows = [attempt(f"web_{i}", str(i), error=error, tool="file", action="read") for i in range(5)]
    assert tool_gaps.mine(rows, nothing_known) == []


def test_a_gap_that_has_since_been_closed_is_not_proposed():
    rows = [attempt(f"web_{i}", str(i)) for i in range(4)]
    assert tool_gaps.mine(rows, lambda tool, verb: True) == []
    crash = "500 Server Error: Internal Server Error"
    rows = [attempt(f"web_{i}", str(i), error=crash, tool="knowledge", action="add", keys=("fact",)) for i in range(4)]
    assert tool_gaps.mine(rows, nothing_known)[0]["kind"] == "params"
    rows.append(attempt("web_9", "z", error="", tool="knowledge", action="add", ok=True))
    assert tool_gaps.mine(rows, nothing_known) == []


def test_a_tool_that_does_not_exist_is_a_new_tool():
    rows = [attempt(f"web_{i}", str(i), error="Tool 'poet' is not a valid, implemented tool.", tool="poet", action="rhyme") for i in range(3)]
    p = tool_gaps.mine(rows, nothing_known)[0]
    assert (p["kind"], p["tool"], p["verb"]) == ("new_tool", "poet", "rhyme")


def test_candidates_take_the_same_params(store):
    rows = [attempt(f"web_{i}", str(i)) for i in range(3)]
    assert tool_gaps.refresh(tool_gaps.mine(rows, nothing_known), TOOLS) == 1
    assert list(tool_gaps.load().values())[0]["candidates"] == ["search"]


def test_candidates_can_be_under_another_tool():
    tools = {"knowledge": {"actions": {"query": {"params": {"query": "string"}}}},
             "study": {"actions": {"train": {"params": {"curriculum": "string"}}}}}
    assert tool_gaps.candidates("knowledge", ["curriculum"], tools) == ["study.train"]
    assert tool_gaps.candidates("knowledge", ["query"], tools) == ["query"]
    assert tool_gaps.candidates("knowledge", [], tools) == []


def test_an_alias_can_reach_another_tool_and_stays_guarded(store):
    tool_gaps._save({"x": {"status": "approved", "kind": "unknown_action", "tool": "worldbuild", "verb": "wipe", "target": "knowledge.rebuild"}})
    action_to_legacy, _l, sensitive, _a = domain_tools._build_mappings()
    assert action_to_legacy[("worldbuild", "wipe")] == action_to_legacy[("knowledge", "rebuild")]
    assert ("worldbuild", "wipe") in sensitive


def test_a_decision_stands_when_the_evidence_grows(store):
    rows = [attempt(f"web_{i}", str(i)) for i in range(3)]
    tool_gaps.refresh(tool_gaps.mine(rows, nothing_known), TOOLS)
    pid = next(iter(tool_gaps.load()))
    tool_gaps.decide(pid, "rejected")
    rows.append(attempt("web_9", "z"))
    assert tool_gaps.refresh(tool_gaps.mine(rows, nothing_known), TOOLS) == 0
    rec = tool_gaps.load()[pid]
    assert rec["status"] == "rejected" and rec["attempts"] == 4


def test_only_an_approved_alias_changes_anything(store):
    rows = [attempt(f"web_{i}", str(i)) for i in range(3)]
    tool_gaps.refresh(tool_gaps.mine(rows, nothing_known), TOOLS)
    assert tool_gaps.approved_aliases() == []
    pid = next(iter(tool_gaps.load()))
    tool_gaps.decide(pid, "approved", target="search")
    assert tool_gaps.approved_aliases() == [("worldbuild", "find", "worldbuild", "search")]


def test_an_approved_alias_works_and_keeps_the_target_sensitivity(store):
    tool_gaps._save({"x": {"status": "approved", "kind": "unknown_action", "tool": "worldbuild", "verb": "find", "target": "search"},
                     "y": {"status": "approved", "kind": "unknown_action", "tool": "knowledge", "verb": "wipe", "target": "rebuild"},
                     "z": {"status": "pending", "kind": "unknown_action", "tool": "worldbuild", "verb": "seek", "target": "search"}})
    action_to_legacy, _l, sensitive, _a = domain_tools._build_mappings()
    assert action_to_legacy[("worldbuild", "find")] == action_to_legacy[("worldbuild", "search")]
    assert ("knowledge", "wipe") in sensitive
    assert ("worldbuild", "seek") not in action_to_legacy


def test_codemind_reads_a_gap_the_way_it_reads_any_other(store):
    import json
    text = tool_gaps.capability_gap_text({"tool": "poet", "verb": "rhyme", "params": ["words"], "attempts": 5, "occasions": 2})
    payload = json.loads(text.split("Payload: ", 1)[1])
    assert payload["attempted_tool_name"] == "poet" and payload["attempted_tool_action"] == "rhyme"


def test_what_she_was_asked_comes_along():
    rows = [{**attempt(f"web_{i}", str(i)), "request": f"find the {i}"} for i in range(5)]
    p = tool_gaps.mine(rows, nothing_known)[0]
    assert p["examples"] == ["find the 0", "find the 1", "find the 2"]
    assert "find the 0" in tool_gaps.capability_gap_text({**p, "params": []})

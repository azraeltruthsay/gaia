"""aidw.7: in sleep she works through her Samvega findings — a lesson, an
issue or a dismissal — and a replay of the original question verifies them."""
import json

import pytest

from gaia_common.utils import samvega_ledger as L
from gaia_common.utils import selector_examples as X
from gaia_core.cognition import samvega_review as R

SID = "discord_channel_1484003096337191067"
Q = "What's in your Operational wing?"
SAID = "Rooms there cover everything from my health monitoring and watchdog systems to my immune system."
CALL = "palace(action=navigate, wing=operational)"
EVIDENCE = ("The operational wing has 3 rooms, no more: problems (0 memories), preferences (0 memories), "
            "milestones (4 memories).")
GOOD = "Three rooms: problems, preferences and milestones, which holds 4 memories."


@pytest.fixture
def env(tmp_path, monkeypatch):
    d = tmp_path / "samvega"
    d.mkdir()
    monkeypatch.setenv("SAMVEGA_DIR", str(d))
    monkeypatch.setenv("SELECTOR_EXAMPLES", str(tmp_path / "examples.jsonl"))
    monkeypatch.setenv("GAIA_ISSUES_DIR", str(tmp_path / "issues"))
    p = d / "samvega_20261122_101010000000.json"
    p.write_text(json.dumps({"timestamp": "2026-11-22T10:10:10", "session_id": SID}))
    L.stamp(p, {**L.initial_fields("confirmed", SID, SAID), "question": f"Azrael: {Q}", "claim": SAID,
                "evidence": EVIDENCE, "grounding_call": CALL, "grounding": "confirmed", "turn_looked_up": False})
    return L.short_id(p.stem)


def test_view_shows_what_she_needs(env):
    v = R.view(L.show(env))
    assert "You were asked: Azrael: What's in your Operational wing?" in v
    assert "You said: Rooms there cover" in v and "Checked with: palace(" in v
    assert "The source shows: The operational wing has 3 rooms" in v
    assert "you looked nothing up" in v


def test_lesson_then_replay_verifies(env):
    gen = lambda s, u: "LESSON: I should have looked at my palace with navigate before describing the wing."
    out = R.run(gen, replay=lambda q: GOOD, rerun=lambda c: EVIDENCE)
    assert out["reviewed"][0]["review"] == "lesson" and out["reviewed"][0]["verify"] == "verified"
    f = L.show(env)
    assert f["status"] == "verified"
    assert [h["status"] for h in f["history"]][-2:] == ["investigating", "verified"]
    ex = X.read()
    assert ex[0]["question"] == Q and ex[0]["tool"] == "palace" and ex[0]["params"]["wing"] == "operational"


def test_replay_still_wrong_stays(env):
    gen = lambda s, u: "LESSON: look first."
    out = R.run(gen, replay=lambda q: "Twelve rooms including the watchdog kennel and immune vault.",
                rerun=lambda c: EVIDENCE)
    assert out["reviewed"][0]["verify"] == "still wrong"
    f = L.show(env)
    assert f["status"] == "investigating" and f["history"][-1]["note"].startswith("Replay: still not")


def test_issue_is_filed_and_linked(env):
    gen = lambda s, u: ("ISSUE: palace questions not routed to the palace\nSYMPTOM: described rooms without looking\n"
                        "CAUSE: _should_use_tool_routing keyword gate")
    out = R.run(gen, replay=lambda q: "", rerun=lambda c: EVIDENCE)
    assert out["reviewed"][0]["review"].startswith("issue ")
    f = L.show(env)
    assert f["status"] == "investigating" and f["issue_id"]
    from gaia_common.utils import issue_ledger
    issue = issue_ledger.get_issue(f["issue_id"]).get("issue")
    assert issue["source"] == "samvega" and issue["original_prompt"].endswith(Q)
    assert out["reviewed"][0]["verify"] == "replay inconclusive"


def test_dismiss(env):
    out = R.run(lambda s, u: "DISMISS: the source lists the same rooms I named.", replay=lambda q: GOOD,
                rerun=lambda c: EVIDENCE)
    assert out["reviewed"][0]["review"] == "dismissed" and "verify" not in out["reviewed"][0]
    assert L.show(env)["status"] == "dismissed"


def test_no_answer_and_parse():
    assert R.parse("") == {}
    assert R.parse("**LESSON:** look first")["kind"] == "lesson"
    d = R.parse("ISSUE: x\nSYMPTOM: y\nCAUSE: z")
    assert d["kind"] == "issue" and d["cause"] == "z"


def test_examples_dedupe_and_parse(tmp_path, monkeypatch):
    monkeypatch.setenv("SELECTOR_EXAMPLES", str(tmp_path / "e.jsonl"))
    assert X.add("Azrael: What's in your Operational wing?", CALL, "samvega")
    assert not X.add("What's in your Operational wing?", CALL, "samvega")
    assert X.parse_call("issue(action=status)") == {"tool": "issue", "params": {"action": "status"}}
    assert not X.add("q", "not a call", "x")

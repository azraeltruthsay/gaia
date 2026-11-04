"""The other mind answers blind; the code looks for what nothing backs (b9ts)."""
import json
import threading

import pytest

from gaia_core.cognition import second_mind as sm


def test_names_and_numbers_nothing_backs():
    assert sm.unbacked("Rome was founded by Romulus, who killed Titus Tatius.", "Romulus founded Rome") == ["Titus Tatius"]
    assert sm.unbacked("The boiling point is 2750 degrees.", "boiling point") == ["2750"]
    assert sm.unbacked("It's 6:47 PM.", "It is 6:47 PM") == []


def test_labels_and_sentence_openings_are_not_claims():
    assert sm.unbacked("[Prime]\n\nSteady, thanks. Nothing to report.", "") == []
    assert sm.unbacked("*[Searched the web for: x]*\n\nThe answer is yes.", "") == []


def test_a_reply_with_nothing_to_check_costs_no_generation():
    def never(prompt):
        raise AssertionError("should not generate")
    assert sm.check("How are you?", "Steady, thanks. How are you?", "", never) is None
    assert sm.check("Capital of France?", "It is in France, of course.", "", never) is None


def test_what_its_own_answer_backs_is_not_flagged():
    out = sm.check("Who wrote it?", "Written by Edgar Allan Poe.", "", lambda p: "Edgar Allan Poe wrote it.")
    assert out is None


def test_what_nothing_backs_is_flagged_and_its_answer_kept():
    out = sm.check("Where did Arthur get his sword?", "From the river, given by Lady Lancelot.", "",
                   lambda p: "From the Lady of the Lake.")
    assert out["verdict"] == "unbacked" and out["unbacked"] == ["Lady Lancelot"]
    assert out["own_answer"] == "From the Lady of the Lake."


def test_it_is_blind_to_the_reply():
    seen = []
    sm.check("Where did Arthur get his sword?", "Given by Lady Lancelot.", "known fact", lambda p: seen.append(p) or "x")
    assert "Lancelot" not in seen[0] and "known fact" in seen[0]


def test_a_fake_tool_result_needs_no_mind():
    def never(prompt):
        raise AssertionError("should not generate")
    assert sm.check("Fetch it", '<tool_result>{"ok": true}</tool_result> It is a page.', "", never)["verdict"] == "fake_tool_result"


def _wait():
    for t in threading.enumerate():
        if t.name == "second-mind":
            t.join(5)


def test_watching_writes_it_down_and_never_raises(tmp_path, monkeypatch):
    log = tmp_path / "o" / "log.jsonl"
    monkeypatch.setattr(sm, "LOG_PATH", str(log))
    sm.watch("Q?", "Given by Lady Lancelot.", "", lambda p: "unknown", "core", "s")
    _wait()
    entry = json.loads(log.read_text())
    assert entry["verdict"] == "unbacked" and entry["observer"] == "core"

    def boom(p):
        raise RuntimeError("no model")
    sm.watch("Q?", "Given by Lady Lancelot.", "", boom, "core", "s")
    _wait()
    assert json.loads(log.read_text().splitlines()[-1])["verdict"] == "error"


def test_it_skips_rather_than_queues(tmp_path, monkeypatch):
    log = tmp_path / "log.jsonl"
    monkeypatch.setattr(sm, "LOG_PATH", str(log))
    assert sm._busy.acquire(blocking=False)
    try:
        sm.watch("Q?", "Given by Lady Lancelot.", "", lambda p: "x", "core")
    finally:
        sm._busy.release()
    assert json.loads(log.read_text())["verdict"] == "skipped_busy"


def test_it_can_be_turned_off(tmp_path, monkeypatch):
    log = tmp_path / "log.jsonl"
    monkeypatch.setattr(sm, "LOG_PATH", str(log))
    monkeypatch.setenv("SECOND_MIND_ENABLED", "0")
    sm.watch("Q?", "Given by Lady Lancelot.", "", lambda p: "x", "core")
    assert not log.exists()

"""The other mind answers blind; the code looks for what nothing backs (b9ts)."""
import json

import pytest

from gaia_core.cognition import second_mind as sm


def test_names_and_numbers_nothing_backs():
    assert sm.unbacked("Rome was founded by Romulus, who killed Titus Tatius.", "Romulus founded Rome") == ["Titus Tatius"]
    assert sm.unbacked("The boiling point is 2750 degrees.", "boiling point") == ["2750"]
    assert sm.unbacked("It's 6:47 PM.", "It is 6:47 PM") == []


def test_part_of_a_known_word_is_backed():
    assert sm.unbacked("It has a Ryzen 9950 and runs Gemma.", "AMD Ryzen 9 9950X3D. Gemma4-E4B.") == []


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


def _review(tmp_path, monkeypatch, question, reply, generate, **kw):
    log = tmp_path / "o" / "log.jsonl"
    monkeypatch.setattr(sm, "LOG_PATH", str(log))
    return sm.review(question, reply, kw.pop("known", ""), generate, "prime", "s", **kw), log


@pytest.mark.parametrize("question,claims,had_source,want", [
    ("What are you running on?", ["Intel Xeon"], False, True),
    ("What did the page say?", ["Ahab"], True, True),
    ("Where did Arthur get his sword?", ["Nimue"], False, True),
    ("What is in your memory palace about The Raven?", ["Allen", "Poe", "Baltimore", "Boston"], False, False),
    ("What do you know about The Raven?", ["Perplexity", "Want", "Create", "Norway"], False, False),
    ("Tell me about the history of Tashkent.", ["Sogdians"], False, False),
    ("What is the history of Tashkent?", ["Sogdians", "Buddhism", "Central Asia", "Asian"], False, False),
])
def test_when_it_seems_necessary(question, claims, had_source, want):
    assert sm.is_necessary(question, {"verdict": "unbacked", "unbacked": claims}, False, had_source) is want


def test_a_fake_tool_result_is_only_fake_if_no_tool_ran():
    fake = {"verdict": "fake_tool_result", "unbacked": []}
    assert sm.is_necessary("Fetch it", fake, False, False)
    assert not sm.is_necessary("Fetch it", fake, True, False)
    assert not sm.is_necessary("Hi", None, False, False)


def test_the_followup_is_in_its_own_words_if_they_name_the_doubt(tmp_path, monkeypatch):
    def mind(prompt):
        return "I can't back up the Intel Xeon part. More text." if "listening" in prompt else "I don't know."
    said, log = _review(tmp_path, monkeypatch, "What are you running on?", "An Intel Xeon CPU.", mind)
    assert said == "I can't back up the Intel Xeon part."
    assert json.loads(log.read_text())["followup"] == said


@pytest.mark.parametrize("words", ["Everything looks fine to me.", "", '{"note": "Intel Xeon"}', "Intel Xeon " * 60 + "."])
def test_a_followup_that_misses_the_doubt_becomes_a_plain_one(tmp_path, monkeypatch, words):
    def mind(prompt):
        return words if "listening" in prompt else "I don't know."
    said, _log = _review(tmp_path, monkeypatch, "What are you running on?", "An Intel Xeon CPU.", mind)
    assert said.startswith("I can't back up what I just said about Intel Xeon")


def test_it_stays_quiet_when_it_is_not_necessary(tmp_path, monkeypatch):
    said, log = _review(tmp_path, monkeypatch, "Tell me about Tashkent.", "Founded by the Sogdians.", lambda p: "unknown")
    assert said == "" and json.loads(log.read_text())["verdict"] == "unbacked"


def test_a_failure_means_silence_and_a_note(tmp_path, monkeypatch):
    def boom(p):
        raise RuntimeError("no model")
    said, log = _review(tmp_path, monkeypatch, "What are you?", "An Intel Xeon.", boom)
    assert said == "" and json.loads(log.read_text())["verdict"] == "error"


def test_it_skips_rather_than_queues(tmp_path, monkeypatch):
    assert sm._busy.acquire(blocking=False)
    try:
        said, log = _review(tmp_path, monkeypatch, "What are you?", "An Intel Xeon.", lambda p: "x")
    finally:
        sm._busy.release()
    assert said == "" and json.loads(log.read_text())["verdict"] == "skipped_busy"


def test_it_can_be_turned_off(tmp_path, monkeypatch):
    monkeypatch.setenv("SECOND_MIND_ENABLED", "0")
    said, log = _review(tmp_path, monkeypatch, "What are you?", "An Intel Xeon.", lambda p: "x")
    assert said == "" and not log.exists()


ARTICLES = "### Article VII: Consent Principle\n- Consent must be freely given, reversible and specific."


@pytest.mark.parametrize("reply,opinion,want", [
    ("Article VII is the Consent Principle: consent must be freely given, reversible and specific.", False, True),
    ("Article VII: Sovereignty and Independence. GAIA holds all powers.", False, False),
    ("Article VII covers rules for everything in general and many unrelated topics.", False, False),
    ("", False, False),
    ("The Consent Principle matters to me: consent freely given feels right, and specific is wise.", True, True),
])
def test_keeping_to_the_text(reply, opinion, want):
    assert sm.keeps_to_text(reply, ARTICLES, "What does Article VII say?", opinion=opinion) is want

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
    ("Consent must be freely given here. " * 3, True, False),
    ("The Consent Principle matters to me: consent freely given feels right, and specific is wise.", True, True),
])
def test_keeping_to_the_text(reply, opinion, want):
    assert sm.keeps_to_text(reply, ARTICLES, "What does Article VII say?", opinion=opinion) is want


def test_a_word_opening_a_line_is_not_a_name():
    assert sm.unbacked("It is good.\n\nThat said, it holds.\n- While true, it is long.", "") == []
    assert sm.unbacked("It is good.\n\nWritten by Titus Tatius.", "") == ["Titus Tatius"]


def test_a_plural_is_backed_by_its_singular():
    assert sm.unbacked("It separates core and Role Personas.", "Tier II: Role Persona") == []


def test_a_section_the_document_lacks_means_she_carried_on_writing_it():
    text = "### Thought seeds\nA thought can be saved. Memory holds it and a heartbeat reviews it."
    assert sm.keeps_to_text("A thought can be saved. ### not a heading here", text, "q", opinion=True)
    assert not sm.keeps_to_text("A thought can be saved.\n\n### The spiral\nA heartbeat reviews it.", text, "q", opinion=True)


def _lifecycle(state):
    from types import SimpleNamespace
    return SimpleNamespace(get_state_sync=lambda timeout: SimpleNamespace(state=state))


@pytest.mark.parametrize("state,source,wakes", [
    ("parked", "discord", True), ("parked", "web", True), ("parked", "heartbeat", False),
    ("focusing", "discord", False), ("awake", "web", False), ("sleep", "web", False),
])
def test_a_person_speaking_wakes_her_from_parked(state, source, wakes, monkeypatch):
    import httpx
    from unittest.mock import MagicMock
    from gaia_core.main import _wake_for_turn
    post = MagicMock(return_value=MagicMock(json=lambda: {"ok": True}))
    monkeypatch.setattr(httpx, "post", post)
    assert _wake_for_turn(_lifecycle(state), "http://o", source) is wakes
    assert post.called is wakes


def test_a_failed_wake_does_not_stop_the_turn(monkeypatch):
    import httpx
    from gaia_core.main import _wake_for_turn
    def boom(*a, **k):
        raise httpx.ConnectError("down")
    monkeypatch.setattr(httpx, "post", boom)
    assert _wake_for_turn(_lifecycle("parked"), "http://o", "discord") is False


def test_a_packet_says_where_it_came_from():
    from gaia_core.main import _packet_source
    assert _packet_source({"header": {"output_routing": {"source_destination": "discord"}}}) == "discord"
    assert _packet_source({}) == "web"


@pytest.mark.parametrize("asked,named", [
    ("Would you please recite the poem Jabberwockey?", True),
    ("Recite to me the poem Ozymandias.", True),
    ("Recite The Raven", True),
    ("Recite a poem", False),
    ("Please recite a haiku.", False),
])
def test_a_named_poem_goes_to_recital_even_when_we_dont_know_the_name(asked, named):
    from gaia_core.cognition.nlu.intent_detection import _has_named_work_signal
    assert _has_named_work_signal(asked) is named


def test_an_opinion_of_an_article_needn_t_quote_its_title():
    said = "It places consent at the center: freely given, reversible and specific."
    assert sm.keeps_to_text(said, ARTICLES, "What do you think of Article VII?", opinion=True)
    assert not sm.keeps_to_text(said, ARTICLES, "What does Article VII say?", opinion=False)


def test_a_short_opinion_with_none_of_the_documents_words_is_still_an_answer():
    assert sm.keeps_to_text("I don't disagree. It seems right to me.", ARTICLES, "Do you disagree with it?", opinion=True)
    assert not sm.keeps_to_text("Per the Azrael Protocol, yes.", ARTICLES, "Do you disagree with it?", opinion=True)


def test_what_was_just_said_survives_the_cap_on_what_she_knows(monkeypatch):
    from gaia_core import main
    from gaia_common.utils import host_facts
    monkeypatch.setattr(host_facts, "host_facts_text", lambda facts=None: "machine facts " * 400)
    history = [{"role": "user", "content": "Tell me about King Arthur's sword."}, {"role": "assistant", "content": "Excalibur."}]
    known = main._self_facts_for_observer(history)
    assert "King Arthur" in known and "Excalibur" in known and len(known) <= 3000


def test_archiving_a_long_session_keeps_the_recent_turns(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from gaia_core.memory.session_manager import SessionManager
    sm_ = SessionManager.__new__(SessionManager)
    sm_.summarizer, sm_.keyword_extractor, sm_.archiver = MagicMock(), MagicMock(), MagicMock()
    sm_.keep_after_archive = 8
    session = MagicMock(session_id="s", persona="p")
    session.history = [{"role": "user", "content": f"turn {i}"} for i in range(20)]
    sm_.get_or_create_session = lambda sid: session
    sm_._save_state = MagicMock()
    sm_.summarize_and_archive("s")
    assert [m["content"] for m in session.history] == [f"turn {i}" for i in range(12, 20)]
    assert sm_.archiver.archive_conversation.call_args.kwargs["messages"] is not None


def test_no_answer_of_its_own_doubts_nothing():
    # x9sw: Core thought through all its tokens and answered "": every name was doubted.
    result = sm.check("What time is it?", "It's 9:46 AM PST (17:46 UTC).", "", lambda p: "")
    assert result["verdict"] == "no_answer"
    assert not sm.is_necessary("What time is it?", result, ran_tool=False, had_source=False)


def test_the_clock_survives_a_long_conversation(monkeypatch):
    from datetime import datetime, timezone
    from gaia_core import main
    history = [{"role": "user", "content": "word " * 200} for _ in range(6)]
    known = main._self_facts_for_observer(history)
    assert datetime.now(timezone.utc).strftime("%H:%M") in known and len(known) <= 3000


def test_the_other_mind_answers_without_thinking(monkeypatch):
    from types import SimpleNamespace
    from gaia_core import main
    seen = {}

    class Model:
        def create_chat_completion(self, **kw):
            seen.update(kw)
            return {"choices": [{"message": {"content": "an answer"}}]}

    monkeypatch.setattr(sm, "review", lambda q, r, k, generate, *a: generate("q"))
    agent = SimpleNamespace(_is_prime_available=True, model_pool=SimpleNamespace(models={"core": Model()}))
    assert main._second_mind_followup(agent, "q", "r", "", "s", False, False) == "an answer"
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}


def test_the_reply_under_check_does_not_back_itself():
    # x9sw: agent_core adds her reply to the history before the check, and it
    # backed every name it had ("Portland, Oregon" for a Richland clock).
    from gaia_core import main
    reply = "[Prime]\n\nIt's 11:57 AM PST in Portland, Oregon."
    history = [{"role": "user", "content": "What time is it?"},
               {"role": "assistant", "content": "It's 11:57 AM PST in Portland, Oregon."}]
    known = main._self_facts_for_observer(history, reply)
    assert "Portland" in sm.unbacked(reply, known)
    # An earlier reply of hers still counts: she can lean on what she said before.
    assert "Portland" not in sm.unbacked(reply, main._self_facts_for_observer(history))

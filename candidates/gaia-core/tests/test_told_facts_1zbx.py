"""Facts she is told are kept, and asked-for ones come from the palace (1zbx)."""
from unittest.mock import MagicMock

import pytest

from gaia_core.cognition.told_facts import (
    covers_each_side,
    disagreeing,
    is_question,
    recall_query,
    speaker,
    speaks_to,
    told_fact,
    told_facts_in,
)

FACT = "The Raven was written by Edgar Allan Poe in 1845."


@pytest.mark.parametrize("text", [
    f"Update my knowledge base with this fact: {FACT}",
    f"Please save the following to my knowledge base: {FACT}",
    f"add to your memory palace: {FACT}",
    f"Can you store this fact in your memory: {FACT}",
    f"Remember this: {FACT}",
    f"Please remember that {FACT}",
])
def test_requests_to_keep_a_fact(text):
    assert told_fact(text) == FACT


@pytest.mark.parametrize("text", [
    "What is in my knowledge base?",
    "Remember when we talked about The Raven?",
    "Remember that time we talked about The Raven?",
    "Remember to save your work.",
    "Update the README with the new ports.",
    "Save this file to /tmp/out.md: hello",
    "How do I add to my knowledge base?",
])
def test_not_requests_to_keep_a_fact(text):
    assert told_fact(text) is None


def test_questions_and_told_facts():
    assert is_question("Where is it kept?") and is_question("what is it") and not is_question("It is kept here.")
    hits = [{"kind": "search", "snippet": "a"}, {"kind": "told", "snippet": "b"}, {"kind": "told"}]
    assert told_facts_in(hits, "What do you know about it?") == [hits[1]]


def test_one_shared_word_does_not_answer_a_question():
    hit = [{"kind": "told", "snippet": FACT}]
    assert told_facts_in(hit, "What is The Raven about?") == []
    assert told_facts_in(hit, "Who was The Raven written by?") == hit
    assert told_facts_in(hit, "What do you know about The Raven?") == hit
    assert recall_query("What do you know about Quillfeather-7 from your knowledge base?") == "Quillfeather-7"


def test_a_reply_must_use_what_the_fact_adds():
    fact = "The lantern is kept in the north attic."
    assert speaks_to("It's in the north attic.", "Where is the lantern kept?", fact)
    assert not speaks_to("The user is asking about the lantern.", "Where is the lantern kept?", fact)


def _core(rpc, said="Done, I noted that."):
    from gaia_core.cognition.agent_core import AgentCore
    core = AgentCore.__new__(AgentCore)
    core.logger = MagicMock()
    core.generate_continuation = MagicMock(side_effect=lambda **kw: iter([{"type": "token", "value": said}]))
    core._palace_call = MagicMock(return_value=rpc)
    return core


def test_the_save_runs_whatever_she_would_have_done_and_she_words_the_reply():
    core = _core((True, {"ok": True}))
    assert core._told_fact_reply(f"Update my knowledge base with this fact: {FACT}", "s") == "Done, I noted that."
    params = core._palace_call.call_args.args[1]
    assert params["fact"] == FACT and params["_told_by"]["why"] == "asked me to keep it"
    assert f"Saved: {FACT}" in core.generate_continuation.call_args.kwargs["messages"][1]["content"]


def test_a_failed_save_cannot_be_reported_as_done():
    core = _core((False, {"error": "disk full"}), said="I've updated it.")
    assert "didn't work" in core._told_fact_reply(f"Update my knowledge base with this fact: {FACT}", "s")


def test_a_question_she_was_told_the_answer_to_comes_from_the_palace():
    core = _core((True, {"results": [{"kind": "told", "snippet": FACT}]}), said="Poe wrote it, in 1845.")
    assert core._told_fact_reply("Who was The Raven written by?", "s") == "Poe wrote it, in 1845."
    assert "not independently checked" in core.generate_continuation.call_args.kwargs["messages"][1]["content"]


def test_the_plain_result_stands_when_her_words_miss_what_happened():
    core = _core((True, {"ok": True}), said="I'll check it against what's there.")
    assert core._told_fact_reply(f"Update my knowledge base with this fact: {FACT}", "s") == f"Saved: {FACT}"
    core = _core((True, {"results": [{"kind": "told", "snippet": FACT}]}), said="They are asking who wrote it.")
    assert core._told_fact_reply("Who was The Raven written by?", "s").startswith(FACT)


def test_other_turns_go_on_as_before():
    core = _core((True, {"results": [{"kind": "search", "snippet": FACT}]}))
    assert core._told_fact_reply("Who was The Raven written by?", "s") is None
    assert core._told_fact_reply("Good morning.", "s") is None
    core.generate_continuation.assert_not_called()


def test_identity_comes_from_the_transport():
    who = speaker("discord", "s", {"user_id": "42", "author_name": "Bo", "is_dm": True})
    assert (who["who"], who["id"], who["where"]) == ("Bo", "discord:42", "in a Discord DM")
    who = speaker("web", "abc", {})
    assert (who["who"], who["id"], who["where"]) == ("someone", "session:abc", "on the web dashboard")
    assert who["when"]


def test_only_the_configured_account_is_the_architect(monkeypatch):
    monkeypatch.setenv("GAIA_OPERATOR_DISCORD_ID", "42")
    monkeypatch.setenv("GAIA_OPERATOR_NAME", "Azrael")
    who = speaker("discord", "s", {"user_id": "42", "author_name": "Anton"})
    assert who["operator"] and who["who"] == "Azrael"
    impostor = speaker("discord", "s", {"user_id": "43", "author_name": "azrael"})
    assert "operator" not in impostor and impostor["who"] == "someone calling themselves Azrael"
    assert "operator" not in speaker("web", "s", {"user_id": "42"})
    monkeypatch.delenv("GAIA_OPERATOR_DISCORD_ID")
    assert "operator" not in speaker("discord", "s", {"user_id": "42"})


FLAT = {"kind": "told", "snippet": "The kettle Brassnote-4 whistles in B flat."}
SHARP = {"kind": "told", "snippet": "The kettle Brassnote-4 whistles in C sharp."}
OTHER = {"kind": "told", "snippet": "The kettle Brassnote-4 was made in Leeds by Harrow and Sons."}


def test_facts_that_differ_about_one_subject_disagree():
    assert disagreeing([FLAT, SHARP]) == [FLAT, SHARP]
    assert disagreeing([FLAT, OTHER]) == []
    assert disagreeing([FLAT]) == []


@pytest.mark.parametrize("reply,ok", [
    ("Alice said B flat, and Bob said C sharp.", True),
    ("It whistles in B flat.", False),
    ("It whistles in C sharp and B flat, they're both right.", False),
    ("They are both correct: flat and sharp.", False),
    ("It whistles in B flat and C sharp. Both answers are plausible.", False),
    ("It whistles in both B flat and C sharp.", False),
])
def test_a_reply_must_give_each_side_without_calling_all_of_them_true(reply, ok):
    assert covers_each_side(reply, [FLAT, SHARP]) is ok


def test_when_her_words_pick_a_side_the_plain_account_stands():
    core = _core((True, {"results": [{**FLAT}, {**SHARP}]}), said="They're both right.")
    out = core._told_fact_reply("What note does the kettle Brassnote-4 whistle in?", "s")
    assert out.startswith("I've been told different things") and "B flat" in out and "C sharp" in out
    assert "These disagree." in core.generate_continuation.call_args.kwargs["messages"][1]["content"]


def test_whether_a_fact_was_checked_comes_from_the_record():
    told = {"kind": "told", "snippet": FACT, "told_by": [{"who": "Azrael", "operator": True}]}
    core = _core((True, {"results": [told]}), said="Yes, it is.")
    out = core._told_fact_reply("Is it verified that The Raven was written by Poe?", "s")
    assert out.startswith("No, not independently.")
    core.generate_continuation.assert_not_called()
    checked = {**told, "validated": {"by": "GAIA", "evidence": "x"}}
    core = _core((True, {"results": [checked]}))
    assert core._told_fact_reply("Is it verified that The Raven was written by Poe?", "s").startswith("Yes.")


def test_her_reply_cannot_give_the_architect_credit_they_did_not_earn():
    impostor = {"kind": "told", "snippet": FACT, "told_by": [{"who": "someone calling themselves Azrael", "id": "discord:9"}]}
    core = _core((True, {"results": [impostor]}), said="Azrael told me The Raven was written by Poe.")
    assert "calling themselves Azrael" in core._told_fact_reply("Who was The Raven written by?", "s")
    real = {"kind": "told", "snippet": FACT, "told_by": [{"who": "Azrael", "operator": True}]}
    core = _core((True, {"results": [real]}), said="Azrael told me Poe wrote it.")
    assert core._told_fact_reply("Who was The Raven written by?", "s") == "Azrael told me Poe wrote it."


FACTS = "This machine: Test GPU with 16.0 GB; Test Chip, 32 threads."


def _host_core(said, monkeypatch):
    import gaia_core.cognition.agent_core as ac
    monkeypatch.setattr(ac, "host_facts_text", lambda: FACTS)
    return _core((True, {}), said=said)


def test_asked_what_she_runs_on_she_answers_from_the_machine(monkeypatch):
    core = _host_core("A Test GPU with 16.0 GB, and a Test Chip.", monkeypatch)
    assert core._host_fact_reply("What hardware are you running on?", "s") == "A Test GPU with 16.0 GB, and a Test Chip."
    assert FACTS in core.generate_continuation.call_args.kwargs["messages"][1]["content"]


def test_if_she_adds_hardware_the_machine_did_not_report_the_facts_stand(monkeypatch):
    core = _host_core("A Core i7-8500B with an RTX 4060 Ti.", monkeypatch)
    assert core._host_fact_reply("What hardware are you running on?", "s") == FACTS


def test_other_questions_are_left_alone(monkeypatch):
    core = _host_core("x", monkeypatch)
    assert core._host_fact_reply("Who wrote The Raven?", "s") is None
    core.generate_continuation.assert_not_called()


def test_words_that_say_nothing_give_way_to_the_facts(monkeypatch):
    core = _host_core("Your environment is as reported.", monkeypatch)
    assert core._host_fact_reply("Tell me about your environment.", "s") == FACTS


CONSTITUTION = "### Article VII: Consent Principle\n- Consent must be freely given."


def _doc_core(said, read=(True, {"content": CONSTITUTION})):
    return _core(read, said=said)


def test_her_constitution_is_read_with_her_tool_and_answered_from():
    core = _doc_core("Article VII is the Consent Principle: consent must be freely given.")
    assert core._founding_doc_reply("What does Article VII say?", "s").startswith("Article VII is the Consent")
    assert core._palace_call.call_args.args[0] == "file"
    assert "freely given" in core.generate_continuation.call_args.kwargs["messages"][1]["content"]
    assert "freely given" in core.take_turn_source("s") and core.take_turn_source("s") == ""


def test_an_invented_article_gives_way_to_the_text():
    core = _doc_core("Article VII: Sovereignty and Independence.")
    out = core._founding_doc_reply("What does Article VII say?", "s")
    assert out.startswith("From the GAIA Constitution:") and "Consent must be freely given" in out


def test_if_it_cannot_be_read_the_turn_goes_on():
    core = _doc_core("x", read=(False, {"error": "nope"}))
    assert core._founding_doc_reply("What does your constitution say?", "s") is None


def test_other_questions_leave_the_documents_shut():
    core = _doc_core("x")
    assert core._founding_doc_reply("What does the US Constitution say?", "s") is None
    core._palace_call.assert_not_called()

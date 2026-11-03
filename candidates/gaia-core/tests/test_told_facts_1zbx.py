"""Facts she is told are kept, and asked-for ones come from the palace (1zbx)."""
from unittest.mock import MagicMock

import pytest

from gaia_core.cognition.told_facts import is_question, recall_query, speaks_to, told_fact, told_facts_in

FACT = "The Raven was written by Edgar Allan Poe in 1845."


@pytest.mark.parametrize("text", [
    f"Update my knowledge base with this fact: {FACT}",
    f"Please save the following to my knowledge base: {FACT}",
    f"add to your memory palace: {FACT}",
    f"Can you store this fact in your memory: {FACT}",
])
def test_requests_to_keep_a_fact(text):
    assert told_fact(text) == FACT


@pytest.mark.parametrize("text", [
    "What is in my knowledge base?",
    "Remember when we talked about The Raven?",
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
    core._palace_call.assert_called_once_with("add_document", {"fact": FACT})
    assert f"Saved: {FACT}" in core.generate_continuation.call_args.kwargs["messages"][1]["content"]


def test_a_failed_save_cannot_be_reported_as_done():
    core = _core((False, {"error": "disk full"}), said="I've updated it.")
    assert "didn't work" in core._told_fact_reply(f"Update my knowledge base with this fact: {FACT}", "s")


def test_a_question_she_was_told_the_answer_to_comes_from_the_palace():
    core = _core((True, {"results": [{"kind": "told", "snippet": FACT}]}), said="Poe wrote it, in 1845.")
    assert core._told_fact_reply("Who was The Raven written by?", "s") == "Poe wrote it, in 1845."
    assert "not verified" in core.generate_continuation.call_args.kwargs["messages"][1]["content"]


def test_the_plain_result_stands_when_her_words_miss_what_happened():
    core = _core((True, {"ok": True}), said="I'll check it against what's there.")
    assert core._told_fact_reply(f"Update my knowledge base with this fact: {FACT}", "s") == f"Saved: {FACT}"
    core = _core((True, {"results": [{"kind": "told", "snippet": FACT}]}), said="They are asking who wrote it.")
    assert core._told_fact_reply("Who was The Raven written by?", "s") == f"I was told: {FACT}"


def test_other_turns_go_on_as_before():
    core = _core((True, {"results": [{"kind": "search", "snippet": FACT}]}))
    assert core._told_fact_reply("Who was The Raven written by?", "s") is None
    assert core._told_fact_reply("Good morning.", "s") is None
    core.generate_continuation.assert_not_called()

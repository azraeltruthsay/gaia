"""Tests for the scaffold-echo strip (GAIA_Project-h1ok).

Core skips the deliberation <think> block on ~94% of turns; when it does,
pieces of the instructions it was given surface in the reply as if spoken to
the user ("Verify before claiming specifics:", "Did I just emit a deflection
instead of an answer?"). The cases below are taken from real journaled
replies. The strip works by provenance — text copied from the scaffold — so
grounding facts and questions that merely share a phrase are left alone.
"""

from gaia_core.cognition.deliberation import (
    _DELIBERATION_INSTRUCTIONS,
    _scaffold_text,
    _strip_scaffold_echo,
)

SCAFFOLD = _scaffold_text(
    "You are GAIA.\n[Candidate context — automated grounding may be incomplete]\n"
    "Paris is the capital of France.",
    _DELIBERATION_INSTRUCTIONS,
)


def test_drops_copied_instruction_line():
    reply = "Verify before claiming specifics:\n\nThe run finished at 04:10."
    out, dropped = _strip_scaffold_echo(reply, SCAFFOLD, "when did the run finish?")
    assert dropped == ["Verify before claiming specifics:"]
    assert out == "The run finished at 04:10."


def test_drops_self_critique_question_mid_reply():
    reply = ("I don't have that log in front of me. "
             "Did I just emit a deferral instead of an answer? "
             "The source of truth would be the doctor's status file.")
    out, dropped = _strip_scaffold_echo(reply, SCAFFOLD, "what did the doctor say?")
    assert len(dropped) == 1
    assert "deferral" not in out
    assert out.startswith("I don't have that log")


def test_drops_echoed_bracketed_header():
    reply = "[Candidate context — automated grounding may be incomplete] Paris is the capital."
    out, dropped = _strip_scaffold_echo(reply, SCAFFOLD, "capital of france?")
    assert dropped and out == "Paris is the capital."


def test_grounding_facts_are_not_scaffold():
    reply = "Paris is the capital of France. It has been since the 10th century."
    out, dropped = _strip_scaffold_echo(reply, SCAFFOLD, "capital of france?")
    assert dropped == [] and out == reply


def test_short_question_sharing_a_phrase_is_kept():
    reply = "Good question. What's the source of truth?"
    out, dropped = _strip_scaffold_echo(reply, SCAFFOLD, "hm")
    assert dropped == [] and out == reply


def test_words_the_user_used_are_not_echo():
    user = "Do you verify before claiming specifics?"
    reply = "I try to verify before claiming specifics. Mostly I succeed."
    out, dropped = _strip_scaffold_echo(reply, SCAFFOLD, user)
    assert dropped == [] and out == reply


def test_all_echo_reply_is_left_unchanged():
    reply = "Verify before claiming specifics."
    out, dropped = _strip_scaffold_echo(reply, SCAFFOLD, "?")
    assert dropped == [] and out == reply

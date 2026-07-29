"""Tests for the leading-vocative-echo strip (GAIA_Project-yswm).

Live Discord 2026-07-29: "Good morning, GAIA." -> "Good morning, glad I
could help. Anything you want to start with?" (odd non-sequitur). Replicated
via scripts/e2e_discord_sim.py against a seeded copy of the real session:
7/7 runs echoed the user's own greeting-vocative back verbatim as the reply's
opening — e.g. "Good morning, GAIA. How can I help you today?" — treating
GAIA's own name as if it were the user's.

Root cause confirmed via the deliberation journal: this path (chitchat/
casual, prompt_builder.py's `_chitchat` branch) hits _split_think_and_response's
fallback 100% of the time for greetings (fallback_used=True, zero voice
evidence) — the model never uses <think> here, so whatever it writes first
becomes the reply verbatim. This is a *different* symptom of the same
underlying fragility that caused lr3u, not something the lr3u fix (scaffold
salvage) addresses on its own, since the raw text here isn't a scaffold
recital — it's simple, coherent, but wrong (echoes the addressee).

Fixed as a POST-generation strip rather than a new prompt instruction:
CLAUDE.md's cognitive-architecture note and prompt_builder.py's own
casual-mode comments (search "backfire") both document that in-prompt
behavioral instructions backfire on Gemma4-E4B for this exact path — a
process-level fix avoids repeating that known failure mode.
"""

from gaia_core.cognition.deliberation import _strip_leading_input_echo


def test_strips_real_observed_vocative_echo():
    user_input = "Good morning, GAIA."
    final_response = "Good morning, GAIA. How can I help you today?"
    assert _strip_leading_input_echo(final_response, user_input) == "How can I help you today?"


def test_leaves_ordinary_reply_untouched_when_no_gaia_mention():
    user_input = "Good morning."
    final_response = "Good morning! How are you?"
    assert _strip_leading_input_echo(final_response, user_input) == final_response


def test_leaves_reply_untouched_when_gaia_mentioned_but_no_echo():
    # "gaia" appears in the input, but the reply doesn't echo it back —
    # nothing to strip, don't touch a normal, on-topic reply.
    user_input = "Hey GAIA, what containers do you run?"
    final_response = "Twelve services, plus HA candidates for most of them."
    assert _strip_leading_input_echo(final_response, user_input) == final_response


def test_handles_different_greeting_phrasing():
    user_input = "hey there gaia!"
    final_response = "hey there gaia! good to hear from you — what's up?"
    assert _strip_leading_input_echo(final_response, user_input) == "Good to hear from you — what's up?"


def test_empty_or_missing_input_is_a_safe_noop():
    assert _strip_leading_input_echo("", "hi gaia") == ""
    assert _strip_leading_input_echo("some reply", "") == "some reply"


def test_echo_stripped_down_to_nothing_falls_back_to_original():
    # If the whole reply IS the echo (nothing left after stripping), don't
    # return an empty string — keep the original rather than a blank reply.
    user_input = "Good morning, GAIA."
    final_response = "Good morning, GAIA."
    assert _strip_leading_input_echo(final_response, user_input) == final_response

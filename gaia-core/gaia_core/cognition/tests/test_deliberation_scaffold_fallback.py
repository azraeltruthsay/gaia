"""Tests for the labeled-scaffold-recital fallback fix (GAIA_Project-lr3u).

When the model skips <think> tags entirely and instead narrates its
Observe/Recall/Verify/Draft/Critique reasoning as the whole visible
completion, the naive fallback used to surface the ENTIRE scaffold to the
user. Observed live 2026-07-08, again in a 2026-06-30 journal entry, and
again 2026-07-29 on a real Discord turn (asking for the current date/time).
The 07-29 case also compounded: the leaked scaffold quoted a forbidden
phrase ("I'm not sure what triggered that") as an example of what NOT to
say, and the naive substring-matching forbidden-phrase detector flagged it
anyway, prepending a second, unrelated low-confidence warning on top.
"""

from gaia_core.cognition.deliberation import (
    _split_think_and_response,
    _extract_draft_from_scaffold,
    _detect_forbidden_phrases,
    _detect_confabulation,
)

# The exact text GAIA sent on Discord 2026-07-29 for "Do you know what date
# and time it is right now?" — reproduced verbatim (minus the low-confidence
# prefix, which deliberate() prepends separately after _split_think_and_response).
_REAL_LEAKED_SCAFFOLD = (
    '[verified] I can verify the current date and time from the system clock.\n\n'
    'Verify: actual current date and time is in UTC-07:00 right now (check before '
    'answering). If I invent "running on a different timezone" or "I\'m not sure '
    'what triggered that," that\'s a failure of verification.\n\n'
    'Draft: "The current date and time, as verified by your system clock, is '
    'UTC-07:00 on Wednesday July 29, 2026. If there\'s something you wanted to ask '
    'about specifically, now\'s the moment."\n\n'
    "Check: gives the actual verified UTC-corrected date and time, doesn't pretend "
    "to be aware of your clock, invites pinpoint question."
)

_EXPECTED_DRAFT = (
    "The current date and time, as verified by your system clock, is UTC-07:00 "
    "on Wednesday July 29, 2026. If there's something you wanted to ask about "
    "specifically, now's the moment."
)


def test_extract_draft_from_real_leaked_scaffold():
    assert _extract_draft_from_scaffold(_REAL_LEAKED_SCAFFOLD) == _EXPECTED_DRAFT


def test_split_think_and_response_salvages_scaffold_recital():
    thinking, final_response, fallback_used = _split_think_and_response(_REAL_LEAKED_SCAFFOLD)
    assert final_response == _EXPECTED_DRAFT
    assert fallback_used is True
    # The raw scaffold is preserved as "thinking" for journaling, not discarded.
    assert thinking == _REAL_LEAKED_SCAFFOLD.strip()


def test_salvaged_draft_does_not_spuriously_trip_forbidden_phrase_gate():
    # Before the fix: _detect_forbidden_phrases ran against the WHOLE raw
    # scaffold and matched "i'm not sure what triggered" even though it only
    # appeared inside the model's own "don't say this" reasoning, not its
    # actual answer — compounding a second bogus low-confidence warning.
    _, final_response, _ = _split_think_and_response(_REAL_LEAKED_SCAFFOLD)
    assert _detect_forbidden_phrases(final_response) == []
    assert _detect_confabulation(final_response) == []


def test_ordinary_fallback_text_is_unaffected():
    raw = "Just a normal reply with no thinking tags and no scaffold labels at all."
    thinking, final_response, fallback_used = _split_think_and_response(raw)
    assert final_response == raw
    assert thinking == ""
    assert fallback_used is True


def test_single_label_does_not_false_positive():
    # A reply that happens to start a sentence with one of the label words
    # should NOT be treated as a scaffold recital — require at least 2 hits.
    raw = "Verify your account by clicking the link below."
    assert _extract_draft_from_scaffold(raw) is None
    _, final_response, _ = _split_think_and_response(raw)
    assert final_response == raw


def test_closed_think_block_still_works_normally():
    raw = "<think>some reasoning here</think>The actual answer."
    thinking, final_response, fallback_used = _split_think_and_response(raw)
    assert thinking == "some reasoning here"
    assert final_response == "The actual answer."
    assert fallback_used is False

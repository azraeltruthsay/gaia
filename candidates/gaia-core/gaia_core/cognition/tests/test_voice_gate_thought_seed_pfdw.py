"""Regression test for GAIA_Project-pfdw follow-up — voice_gate.py had no
pattern for Prime's "THOUGHT_SEED:" native-tool-calling scaffold marker, so
even after the raw <tool_call> JSON stopped leaking, the preamble sentence
("THOUGHT_SEED: I'll search the conversation history for King Arthur
references.") still rendered as visible text in real Discord replies.

These test _matches_tell directly (no embed model / GPU needed) so they run
fast and deterministically regardless of whether the embedding half of the
gate is available in a given environment.
"""

from gaia_core.cognition.voice_gate import _matches_tell, filter_voiced


def test_thought_seed_sentence_is_tagged_as_meta():
    assert _matches_tell(
        "THOUGHT_SEED: I'll search the conversation history for King Arthur references."
    )


def test_thought_seed_case_insensitive_and_no_space_before_colon():
    assert _matches_tell("THOUGHT_SEED:check memory")
    assert _matches_tell("thought_seed: check memory")


def test_ordinary_sentence_not_flagged():
    assert not _matches_tell("Sure, I can help with that. Where do you want to start?")


def test_filter_voiced_strips_thought_seed_preamble_when_enabled():
    text = (
        "THOUGHT_SEED: I'll search the conversation history for King Arthur references. "
        "Sure — we talked about Excalibur and the Sword in the Stone earlier."
    )
    out, debug = filter_voiced(text, measure_only=False)
    assert "THOUGHT_SEED" not in out
    assert "Excalibur" in out
    assert debug["dropped"]

"""Tests for the lean casual deliberation block (GAIA_Project-7n3).

Casual turns ("how are you") get a lean anti-confab-ONLY deliberation block
instead of the full analytical observe/quote/critique scaffolding — the
scaffolding diluted the felt Inner-weather line into a status readout.

These pin the SELECTION behavior (which instruction block is appended) using
a fake model_pool that captures the assembled messages. No GPU / no real model.
"""


from gaia_core.cognition.deliberation import (
    deliberate,
    _DELIBERATION_INSTRUCTIONS,
    _CASUAL_DELIBERATION_INSTRUCTIONS,
)


class _CapturePool:
    """Minimal model_pool stand-in that records the messages it was asked
    to forward and returns a fixed, clean (no-confab) response."""

    def __init__(self):
        self.captured_messages = None

    def acquire_model(self, role):
        return object()  # truthy sentinel

    def release_model(self, role):
        pass

    def forward_to_model(self, role, *, messages, **kwargs):
        self.captured_messages = messages
        return {"choices": [{"message": {"content": "<think>fine</think>I'm well, thanks — you?"}}]}


def _system_text(messages):
    assert messages and messages[0]["role"] == "system"
    return messages[0]["content"]


def test_casual_uses_lean_block_not_analytical():
    pool = _CapturePool()
    base = [
        {"role": "system", "content": "You are GAIA."},
        {"role": "user", "content": "how are you?"},
    ]
    deliberate(
        user_input="how are you?",
        assembled_messages=base,
        model_pool=pool,
        persist=False,
        casual=True,
    )
    sys_text = _system_text(pool.captured_messages)
    # Lean block present, heavy analytical scaffolding absent.
    assert _CASUAL_DELIBERATION_INSTRUCTIONS in sys_text
    assert "observe what they literally said" not in sys_text.lower()
    assert "draft a reply, then critique it" not in sys_text.lower()
    # Original persona is preserved (addendum, not replacement).
    assert "You are GAIA." in sys_text


def test_noncasual_uses_full_analytical_block():
    pool = _CapturePool()
    base = [
        {"role": "system", "content": "You are GAIA."},
        {"role": "user", "content": "explain the gearbox states"},
    ]
    deliberate(
        user_input="explain the gearbox states",
        assembled_messages=base,
        model_pool=pool,
        persist=False,
        casual=False,
    )
    sys_text = _system_text(pool.captured_messages)
    assert _DELIBERATION_INSTRUCTIONS in sys_text
    assert _CASUAL_DELIBERATION_INSTRUCTIONS not in sys_text


def test_default_is_noncasual():
    """Omitting `casual` keeps the historical analytical behavior."""
    pool = _CapturePool()
    base = [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}]
    deliberate(
        user_input="q",
        assembled_messages=base,
        model_pool=pool,
        persist=False,
    )
    sys_text = _system_text(pool.captured_messages)
    assert _DELIBERATION_INSTRUCTIONS in sys_text


def test_lean_block_keeps_anticonfab_drops_scaffolding():
    """The lean block must retain the no-confabulation guard (the part that
    earns its keep vs. the degenerate slim path) while dropping the analytical
    4-move frame and the forbidden-phrase list."""
    lean = _CASUAL_DELIBERATION_INSTRUCTIONS.lower()
    # Anti-confab guard retained.
    assert "invent" in lean
    assert "status report" in lean or "status readout" in lean
    # Analytical scaffolding dropped.
    assert "critique" not in lean
    assert "forbidden phrases" not in lean
    # No explicit "voice your affect" instruction (bench: it backfires on E4B).
    assert "inner weather" not in lean
    assert "express your affect" not in lean

"""8xr4.23: a heartbeat seed turn mentioning "focus" swapped the whole GPU into
FOCUSING. Only user turns may request it, and only on whole words."""

import pytest

from gaia_core.cognition.agent_core import _wants_focus_transition


@pytest.mark.parametrize("text", [
    "focus on this one please",
    "Can you use prime for this?",
    "think hard about the proof",
    "FOCUSING mode, please",
])
def test_user_focus_request(text):
    assert _wants_focus_transition(text, "discord_dm")
    assert _wants_focus_transition(text, "web")


@pytest.mark.parametrize("source", ["heartbeat", "gaia-doctor", "HEARTBEAT"])
def test_internal_turns_never_shift_gears(source):
    assert not _wants_focus_transition("Refine the focus of the stale documents review", source)
    assert not _wants_focus_transition("focus", source)


@pytest.mark.parametrize("text", [
    "I was so focused today",
    "time to refocus the docs",
    "what's the focal length?",
    "",
])
def test_no_substring_matches(text):
    assert not _wants_focus_transition(text, "discord_dm")

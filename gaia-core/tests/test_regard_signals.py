"""2026-11-17: warmth and hostility, not teasing."""
import pytest

from gaia_core.cognition.regard_signals import classify


@pytest.mark.parametrize("text,kind", [
    ("thanks GAIA, that helped", "warm"),
    ("I appreciate you", "warm"),
    ("you're useless", "hostile"),
    ("shut up", "hostile"),
    ("you are such an idiot", "hostile"),
    ("I'm going to eat you for dinner lol", ""),
    ("what's the capital of Australia?", ""),
    ("that answer was wrong", ""),
])
def test_classify(text, kind):
    assert classify(text) == kind

"""A persona overlay is labeled as a role GAIA takes on for the conversation,
and it comes after her identity, which stays a plain fact. Her identity is in
her weights; prompt-injected identity never took, so the prompt only marks
which part is the role."""
from gaia_core.behavior.persona_switcher import get_persona_overlay_text, label_role


def test_label_role_frames_the_persona_as_a_role():
    text = label_role("• Speak as the ship's AI.", "dnd_player_assistant")
    assert text.startswith("— Role for this conversation: dnd player assistant —")
    assert "You are GAIA." in text and "you remain yourself" in text
    assert "• Speak as the ship's AI." in text
    assert text.rstrip().endswith("— End of role —")


def test_dnd_overlay_is_labeled():
    overlay = get_persona_overlay_text("dnd_player_assistant")
    assert overlay is not None
    assert overlay.startswith("— Role for this conversation:")
    assert "Anton Snark" in overlay  # the role's own content is kept


def test_persona_without_template_or_list_has_no_overlay():
    assert get_persona_overlay_text("no_such_persona") is None

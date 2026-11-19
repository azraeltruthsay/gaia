"""Engineering mode: which turns, and what an answer from memory says (oeql)."""
from types import SimpleNamespace as R

from gaia_core.cognition import engineering_mode as em


def test_turn_kind_from_her_notes_and_the_web():
    study = lambda s: R(source="study", score=s, content="x" * 80)
    web = R(source="web", score=0, content="y" * 80)
    assert em.turn_kind([]) is None
    assert em.turn_kind([study(0.40), web]) is None            # notes not near: not engineering
    assert em.turn_kind([study(0.62)]) == "notes"
    assert em.turn_kind([study(0.47), web]) == "web"
    assert em.turn_kind([study(0.47)]) == "memory"


def test_general_knowledge_is_marked_and_checked_by_the_other_mind():
    a = "The center of gravity is the average position of the weight; hang the object from two points."
    agreeing = lambda m: "Hang it from two points; the center of gravity is where the plumb lines cross, the average position of weight."
    text, verdict = em.check_general("How do I find the center of gravity?", a, agreeing)
    assert verdict == "agreed" and text.startswith(em.GENERAL)
    text, verdict = em.check_general("q", a, lambda m: "Use a torsion pendulum and measure inertia.")
    assert verdict == "disagreed" and text == em.UNSURE
    text, verdict = em.check_general("q", a, None)
    assert verdict == "unchecked" and text.startswith(em.GENERAL)
    text, verdict = em.check_general("q", a, lambda m: "")     # other mind asleep
    assert verdict == "unchecked"


def test_numbers_must_agree():
    assert not em.agree("The FE exam is 4 hours long.", "The FE exam is 6 hours long.")
    assert em.agree("The FE exam is 6 hours long.", "It lasts 6 hours; the FE exam has 110 questions.")


def test_adapter_off_by_default_and_card_read(monkeypatch, tmp_path):
    monkeypatch.delenv("ENGINEERING_ADAPTER", raising=False)
    assert em.adapter() == ""
    card = tmp_path / "role_card.md"
    card.write_text("# Role card\n\n> You are Andy's engineering intern.\n> You cite what you read.\n\nWhy...")
    monkeypatch.setattr(em, "ROLE_CARD_PATH", card)
    assert em.role_card() == "You are Andy's engineering intern. You cite what you read."

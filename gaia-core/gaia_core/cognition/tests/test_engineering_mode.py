"""Engineering mode: which turns, and what an answer from memory says (oeql)."""
from types import SimpleNamespace as R

from gaia_core.cognition import engineering_mode as em


def test_turn_kind_from_her_notes_and_the_web():
    study = lambda s: R(source="study", score=s, content="x" * 80, url="https://app.leg.wa.gov/wac/x")
    web = R(source="web", score=0, content="y" * 80)
    assert em.turn_kind([]) is None
    assert em.turn_kind([study(0.40), web]) is None            # notes not near: not engineering
    assert em.turn_kind([study(0.62)]) == "notes"
    assert em.turn_kind([study(0.47), web]) == "web"
    assert em.turn_kind([study(0.47)]) == "memory"
    # her note on herself (no web source) is not an engineering turn
    assert em.turn_kind([R(source="study", score=0.62, content="x" * 80, url="GAIA's own code")]) is None


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


def test_web_answer_backed_by_the_page_is_kept_unbacked_one_is_rechecked():
    page = ("Stress-strain analysis uses many simplifying assumptions; structures are often modelled as in "
            "static equilibrium, and the stress state is found from forces and moments.")
    good = "I looked it up: structures are often modelled as in static equilibrium, and the stress state is found from forces and moments."
    assert em.check_web("q", good, page, None) == (good, "kept")
    bad = ("My notes on Stress don't cover that, but I looked it up: engineers use the finite element method, "
           "meshing the part into tetrahedra and solving stiffness matrices. (Source: https://x/Stress)")
    text, verdict = em.check_web("q", bad, page, None)
    assert verdict == "unchecked" and text.startswith(em.GENERAL)
    assert "looked it up" not in text and "Source" not in text and "finite element" in text


def test_asked_for_engineering_mode():
    from gaia_core.cognition import engineering_mode as em
    assert em.asked_for("Can you shift to Gear 2, load your engineering adapter and try that again?")
    assert em.asked_for("switch to engineering mode please")
    assert not em.asked_for("How has your evening been?")

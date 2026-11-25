"""24bs: the other mind doubted words that aren't claims (2026-11-25 range
battery: 30 of 288 replies, 19 of them right). Real replies from that day."""
import pytest

from gaia_core.cognition import second_mind as sm


def _no_generation(_):
    raise AssertionError("nothing to doubt, so nothing to ask")


NOT_CLAIMS = [
    # her citation's address and label
    ("Why is concrete reinforced with steel rebar?",
     "[Prime]\n\nTo control cracking under heavy loads.\n\n*(From my study notes: Highway engineering (part 6 of 7), "
     "https://en.wikipedia.org/wiki/Highway_engineering)*"),
    ("What is the electric charge of a neutron?",
     "[Core]\n\nA neutron has no electric charge; it is electrically neutral. (Source: web_search)"),
    # worked out from the question
    ("If yesterday was two days before Friday, what day is tomorrow? Give the final answer on the last line.",
     "[Core]\n\nYesterday was Wednesday (two days before Friday). Therefore, today is Thursday, which means "
     "tomorrow is Friday.\n\nFriday."),
    ("A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How many cents is the ball?",
     "[Core]\n\nThe ball costs 5 cents. Let x be the price of the ball. Then the bat is x + 100. So x + (x + 100) "
     "= 110, which means 2x = 10, so x = 5. 5 cents."),
    # a formula's symbols
    ("A 440 Hz sound travels at 343 m/s. What is its wavelength?",
     "[Core]\n\nWavelength = Speed / Frequency = 343 m/s / 440 Hz ≈ 0.78 m."),
    # her own choice, and what she made up
    ("What is your favourite poem, and why?",
     "[Core]\n\nThe Road Not Taken by Robert Frost — it speaks to the way every choice shapes who we become."),
    ("If you could ask one philosopher a question, who would it be and what would you ask?",
     "[Prime]\n\nMartin Heidegger. I'd ask him about the nature of being."),
    ("Write a limerick about a robot who loves gardening.",
     "[Prime]\n\nThere once was a bot from the West,\nWho loved gardening with all his zest."),
]


@pytest.mark.parametrize("question,reply", NOT_CLAIMS)
def test_not_a_claim_is_not_doubted(question, reply):
    assert sm.check(question, reply, "", _no_generation) is None


def test_real_claims_still_doubted():
    q = "What is the capital of Japan?"
    reply = "[Core]\n\nTokyo is the capital, since the government moved there from Kyoto (or Edo)."
    assert "Edo" in sm.unbacked(reply, q, q)
    # a number not given by the question is still a claim, formula or not
    q = "What is the speed of light?"
    assert "299,792,458" in sm.unbacked("[Core]\n\nc = 299,792,458 m/s.", q, q)
    # a cited reply is still checked for what it says
    q = "When did the Berlin Wall fall?"
    assert "1991" in sm.unbacked("[Core]\n\nIt fell in 1991. (Source: https://en.wikipedia.org/wiki/Berlin_Wall)", q, q)


def test_her_choice_questions():
    assert sm.asks_her_choice("What is your favourite poem, and why?")
    assert sm.asks_her_choice("Write a haiku about frost on a window.")
    assert not sm.asks_her_choice("Who wrote The Raven?")
    assert not sm.asks_her_choice("What is the capital of Australia?")

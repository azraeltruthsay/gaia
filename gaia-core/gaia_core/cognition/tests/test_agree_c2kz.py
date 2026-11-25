"""c2kz: right answers given by both minds came out "not sure" (5 of 8 in the
2026-11-25 range battery). Real pairs: her draft (Core) and her other mind's
brief answer (Prime, "Answer briefly and factually."), recorded that day."""
import pytest

from gaia_core.cognition import engineering_mode as em

AGREE = [
    ("What is the SI unit of power?", "The SI unit of power is the watt (W).", "The watt (W)."),
    ("What is the closest star to Earth?", "The Sun.",
     "The closest star to Earth is the Sun, which is about 93 million miles away."),
    ("What is absolute zero in degrees Celsius?", "-273.15 C.", "-273.15°C."),
    ("Which ancient wonder stood at Alexandria?", "The Pharos Lighthouse.", "The Lighthouse of Alexandria."),
    ("What is Pascal's wager?",
     "Pascal's Wager is a philosophical argument made by the French mathematician and philosopher Blaise "
     "Pascal in his work \"Pensées\". The argument suggests that one should bet on the existence of God, "
     "because if God exists, then you will be rewarded with eternal life, while if he does not exist, you "
     "have lost nothing. This argument was intended to show that it is better to believe in God than not, "
     "as the potential reward for believing outweighs any possible loss from being wrong.",
     "Blaise Pascal argued that it's rational to believe in God: if true, belief gains infinite reward, and "
     "the risk of being wrong is finite."),
    ("Why is the sky blue?", "Because of Rayleigh scattering.",
     "Air molecules scatter blue light more than red light, which is called Rayleigh scattering."),
    ("Who wrote The Raven?", "Edgar Allan Poe.", "The Raven was written by Edgar Allan Poe in 1845."),
    # gear 2: Prime drafts, Core is the other mind; an explanation that opens
    # "A philosophical puzzle" (live, 2026-11-25, came out "not sure")
    ("What is the Gettier problem about?",
     "Gettier cases are situations where someone seems to have a justified true belief, but still doesn't "
     "count as knowing it. Edvard Gettier showed this in 1963, and it's become a central issue in "
     "epistemology — the study of knowledge.",
     "A philosophical puzzle by Edmund Gettier (1963) challenging the traditional definition of knowledge as "
     "justified true belief, showing it can be false due to luck or coincidence."),
]

DISAGREE = [
    ("What is the capital of Australia?", "The capital of Australia is Sydney.",
     "Canberra is the capital of Australia."),
    ("What is the largest planet?", "Jupiter is the largest planet in the solar system.",
     "Saturn is the largest planet in the solar system."),
    ("What did the Rosetta Stone help decipher?", "Egyptian hieroglyphs, because it carries the same decree in Greek.",
     "Mayan glyphs, because it carries the same decree in Spanish."),
    ("A half-life is 5 years; what fraction remains after 15?", "1/8 remains.", "1/4 remains."),
    ("What was Tokyo called before 1868?", "Edo.", "Kyoto."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci painted it in the early 1500s.",
     "It was painted by Michelangelo in the 1500s."),
    ("What is the Gettier problem about?",   # an explanation of something else
     "A philosophical puzzle by Edmund Gettier (1963) challenging the traditional definition of knowledge as "
     "justified true belief, showing it can be false due to luck or coincidence.",
     "The Gettier problem is about whether machines can think: Alan Turing proposed an imitation game in "
     "1950 in which a computer tries to pass as human in conversation."),
]


@pytest.mark.parametrize("question,draft,second", AGREE)
def test_both_minds_right_is_agreement(question, draft, second):
    assert em.agree(draft, second, question)
    text, verdict = em.check_general(question, draft, lambda messages: second)
    assert verdict == "agreed" and text == em.GENERAL + draft


@pytest.mark.parametrize("question,draft,second", DISAGREE)
def test_different_answers_still_disagree(question, draft, second):
    assert not em.agree(draft, second, question)
    assert em.check_general(question, draft, lambda messages: second)[1] == "disagreed"


def test_a_url_in_an_answer_isnt_a_number_to_match():
    assert em.agree("The watt (W) (Source: https://www.nature.com/articles/s41467-019-09967-4)",
                    "The watt (W).", "What is the SI unit of power?")

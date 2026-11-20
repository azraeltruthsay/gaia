"""answer_checks: arithmetic, page and notes-citation checks (2026-11-19, the SSD problem)."""
from types import SimpleNamespace as R

from gaia_core.cognition import answer_checks as ac
from gaia_core.cognition.nlu.intent_detection import is_greeting_only

HER_FIRST = """PRD = 73.33 ft/s × 2.5 s = 183.325 ft
BD = (73.33^2 - 0) / (2 × 11.2) ≈ 224.94 ft
SSD = PRD + BD = 183.325 ft + 224.94 ft ≈ 408.27 ft
Rounding off, the closest answer choice is A: 400ft."""

HER_RETRY = """V = 50 mph * (5280 ft/mile) / (3600 s/hour) ≈ 73.33 ft/s
SSD = 1.47 * 73.33 * (1.5 + 1.0) + (73.33)^2 / (2 * 32.2 * (0.3 ± 0))
SSD = 1.47 * 73.33 * 2.5 + (73.33)^2 / (2 * 32.2 * 0.3)
SSD ≈ 275.0 + 224.94
SSD ≈ 499.94 ft"""


def test_finds_her_braking_distance_slip():
    errs = ac.arithmetic_errors(HER_FIRST)
    assert len(errs) == 1
    expr, stated, actual = errs[0]
    assert stated.startswith("224.94") and abs(actual - 240.06) < 0.1


def test_finds_the_retry_slips_and_passes_right_steps():
    errs = ac.arithmetic_errors(HER_RETRY)
    stated = {s for _, s, _ in errs}
    assert "275.0 + 224.94" in stated   # 1.47*73.33*2.5 + 73.33^2/19.32 is 547.8
    assert not any("5280" in e for e, _, _ in errs)   # the conversion was right
    assert ac.arithmetic_errors("PRD = 1.47 × 50 × 2.5 = 183.75 ft\nBD = 1.075 × 50² / 11.2 = 239.96 ft") == []


def test_no_arithmetic_is_ok():
    assert ac.check_arithmetic("q", "The FE exam is 6 hours = a long day.", None) == (
        "The FE exam is 6 hours = a long day.", "ok")


def test_rewrite_with_right_values_then_note_if_still_wrong():
    fixed = "BD = (73.33^2) / (2 × 11.2) ≈ 240.06 ft, so SSD ≈ 423.4 ft: B."
    seen = []
    out, v = ac.check_arithmetic("SSD?", HER_FIRST, lambda m: seen.append(m) or fixed)
    assert v == "fixed" and out == fixed and "240.1" in seen[0][0]["content"]
    out, v = ac.check_arithmetic("SSD?", HER_FIRST, lambda m: HER_FIRST)
    assert v == "noted" and "Checked by calculation" in out


def test_computed_correction_without_her_rewrite():
    q = "A source reads 400 mR/hr at 2 m. What is the rate at 4 m behind 3 half-value layers?"
    steps = "I0 = 400  # mR/hr\nI4 = I0 * (2/4)**2  # mR/hr\nrate = I4 * 0.5**3  # mR/hr"
    out, v = ac.check_computation(q, "The dose rate at 4m would be 20 mR/hr.",
                                  lambda m: steps if m[0]["content"].startswith("Work") else "It is 20 mR/hr.")
    assert v == "rewritten" and out.endswith("is 12.5 mR/hr.") and "_" not in out


def test_plain_words():
    q = "A source reads 400 mR/hr at 2 m. What is the rate at 4 m behind 3 half-value layers?"
    steps = ac.run_steps("rate_at_2m = 400\nexposure_rate_at_4m_without_hvl = rate_at_2m * (2/4)**2\n"
                         "attenuation_factor = 0.5**3\nexposure_rate_at_4m_with_hvl = exposure_rate_at_4m_without_hvl * attenuation_factor")
    plain = ac.phrase_result(q, steps, {400.0, 2.0, 4.0, 3.0}, None)
    assert plain == ("Working it through: exposure rate at 4 m without half-value layers is 100 mR/hr, and "
                     "attenuation factor is 0.125. So the exposure rate at 4 m with half-value layers is 12.5 mR/hr.")
    hers = "The rate is 12.5 mR/hr: the inverse square law cuts 400 mR/hr to 100 mR/hr at 4 m, and three half-value layers divide that by 8."
    assert ac.phrase_result(q, steps, {400.0, 2.0, 4.0, 3.0}, lambda m: hers) == hers
    assert ac.phrase_result(q, steps, {400.0, 2.0, 4.0, 3.0}, lambda m: "exposure_rate = 13 mR/hr") == plain


PAGE = R(source="web", read=True, url="https://en.wikipedia.org/wiki/Stopping_sight_distance",
         content=("In US customary units stopping sight distance is SSD = 1.47 V t + 1.075 V^2 / a, "
                  "where V is the design speed in mph, t is the brake reaction time of 2.5 seconds "
                  "and a is the deceleration rate of 11.2 ft/s2."))


def test_page_rewrite_when_answer_not_backed():
    bad = ("The AASHTO formula uses perception time of 1.5 seconds and a separate reaction time of 1.0 second "
           "with the friction factor of the tires.")
    good = ("The page gives SSD = 1.47 V t + 1.075 V^2 / a, with V the design speed in mph, "
            "t the brake reaction time of 2.5 seconds and a the deceleration rate of 11.2 ft/s2.")
    out, v = ac.check_page("SSD formula?", bad, [PAGE], lambda m: good, None)
    assert v == "rewritten" and PAGE.url in out
    out, v = ac.check_page("SSD formula?", good, [PAGE], None, None)
    assert v == "kept"


def test_page_falls_back_to_general_check_without_engineering_wording():
    bad = "The answer involves the friction factor for icy pavements in northern climates."
    out, v = ac.check_page("q", bad, [PAGE], lambda m: bad, lambda m: "Something else entirely about bridges.",
                           engineering=False)
    assert v == "disagreed" and "Andy" not in out


def test_cite_notes():
    note = R(source="study", score=0.6, title="WAC 196-23 Stamping and seals",
             url="https://app.leg.wa.gov/wac/default.aspx?cite=196-23",
             content="Document Review: Reviewer must fully review, potentially draft a report with "
                     "calculations/sketches, and then seal/stamp, sign, and date that report.")
    ans = ("The reviewer must fully review the document, potentially draft a report with sketches "
           "or calculations, and then seal, sign, and date that report.")
    out = ac.cite_notes(ans, [note])
    assert out.endswith("cite=196-23)*") and "WAC 196-23" in out
    assert ac.cite_notes(ans + " (WAC 196-23)", [note]).count("study notes") == 0
    assert ac.cite_notes("Unrelated chat about the weather today.", [note]) == "Unrelated chat about the weather today."


def test_greeting_only():
    assert is_greeting_only("Hey GAIA, how are you?")
    assert is_greeting_only("Good evening GAIA!")
    assert not is_greeting_only("GAIA\n\n\nExample 1: Transportation / Geometric Design (Stopping Sight "
                                "Distance)\n\nProblem: A two-lane rural highway has a design speed of 50 mph.")


def test_worked_problem_is_grounded():
    from gaia_core.cognition.nlu.intent_detection import is_worked_problem
    assert is_worked_problem("Problem: A two-lane rural highway has a design speed of 50 mph and a deceleration "
                             "of 11.2 ft/s2. What is the required Stopping Sight Distance per AASHTO?")
    assert not is_worked_problem("I had 2 coffees and 3 cookies today, how about you?")
    assert not is_worked_problem("How has your evening been?")


def test_clock_only_when_asked():
    from gaia_core.cognition.knowledge_router import _query_system_context as sc
    assert sc("What time is it?") and sc("what's the date today") and sc("How long have you been running?")
    assert not sc("Assuming a driver perception-reaction time of 2.5 seconds, what is the SSD?")
    assert not sc("How long is the FE exam?")
    assert not sc("What did you do today?")


def test_shielding_steps_are_right():
    text = """Exposure rate after shielding = 120 mR/hr × (1/2)^4
Exposure rate after shielding = 120 mR/hr × 1/16
Exposure rate after shielding = 7.5 mR/hr
Intensity2 = 7.5 mR/hr × (1/2)^2
Intensity2 = 7.5 mR/hr × 1/4
Intensity2 = 1.875 mR/hr"""
    assert ac.arithmetic_errors(text) == []
    assert len(ac.arithmetic_errors(text.replace("1.875", "2.5"))) == 1


def test_snippets_are_not_pages():
    snippet = R(source="web", url="https://x", content="Stopping sight distance is the sum of two distances " * 3)
    assert ac.web_pages([snippet]) == []
    assert ac.check_page("q", "B: 425 ft", [snippet], None, None) == ("B: 425 ft", "no_page")


def test_her_name_in_front_is_still_a_world_question():
    from gaia_core.cognition.nlu.intent_detection import is_world_question
    assert is_world_question("GAIA, What code year of the IBC has the state of Washington adopted as of today?")
    assert is_world_question("Hey GAIA, who wrote the WAC 196 rules?")
    assert not is_world_question("GAIA, how are you?")


def test_search_terms():
    from gaia_core.cognition.knowledge_router import _ADDRESS_RE, _search_terms
    q = _ADDRESS_RE.sub("", "GAIA, What code year of the IBC has the state of Washington adopted as of today?")
    assert q.startswith("What code year")
    assert _search_terms(q) == "code year IBC Washington adopted"


def test_off_topic_search_results_dropped():
    from gaia_core.cognition.knowledge_router import GroundingResult as G, _on_topic
    q = "What code year of the IBC has the state of Washington adopted as of today?"
    vs = G(source="web", trust_tier="searched", content="Visual Studio Code - Build what comes next", url="https://code.visualstudio.com/")
    sb = G(source="web", trust_tier="searched", content="Codes in effect: 2021 IBC", url="https://sbcc.wa.gov/x", title="Statewide codes adopted in Washington")
    assert _on_topic([vs, sb], q) == [sb]


def test_other_mind_rewrites_when_first_cannot():
    wrong = "Washington has adopted the 2024 International Building Code as of today."
    right = ("The page gives SSD = 1.47 V t + 1.075 V^2 / a, with V the design speed in mph, "
             "t the brake reaction time of 2.5 seconds and a the deceleration rate of 11.2 ft/s2.")
    out, v = ac.check_page("SSD formula?", wrong, [PAGE], lambda m: wrong, lambda m: right)
    assert v == "rewritten" and out.startswith("The page gives")


def test_pointer_without_answer():
    stub = ("My notes on Washington WAC 196-23: Stamping and seals, from "
            "https://app.leg.wa.gov/wac/default.aspx?cite=196-23&full=true cover that — it outlines the rules "
            "for the use of professional seals and signatures in the state of Washington.")
    assert ac.points_without_answering(stub)
    assert not ac.points_without_answering("WAC 196-23 covers seals. Before sealing someone else's work, the PE "
                                           "must fully review it and seal a report of that review.")
    assert not ac.points_without_answering("My notes on WAC 196-23 don't cover that.")


def test_computation_catches_unworked_wrong_answer():
    q = "A 6-inch inside-diameter pipe carries water at 3 ft/s. What is the flow in cfs and gpm?"
    steps = ("D = 6 / 12  # ft\nA = pi * D**2 / 4  # ft2\nQ = A * 3  # cfs\ngpm = Q * 448.831  # gpm")
    fixed = "A = 0.196 ft2, Q = 0.589 cfs, about 264.4 gpm."

    def gen(m):
        return steps if m[0]["content"].startswith("Work this problem") else fixed
    out, v = ac.check_computation(q, "The flow rate is 5.76 cfs and 261.8 gpm.", gen)
    assert v == "rewritten" and out.startswith(fixed.rstrip(".")) and "Q = 0.589 cfs" in out
    out, v = ac.check_computation(q, "About 0.589 cfs, or 264.4 gpm.", gen)
    assert v == "ok"
    assert ac.check_computation("How are you?", "Fine", gen) == ("Fine", "skipped")


def test_run_steps():
    vals = {n: v for n, v, _ in ac.run_steps("x = 2\ny = sqrt(x * 8)  # units\nz = bogus(1)\nw = 5")}
    assert vals == {"x": 2.0, "y": 4.0}


def test_computation_needs_two_agreeing_tries():
    q = "A 6-inch inside-diameter pipe carries water at 3 ft/s. What is the flow in cfs and gpm?"
    tries = iter(["D = 3\nQ = pi * D**2 / 4 * 3", "D = 0.5\nQ = pi * D**2 / 4 * 3", "D = 1\nQ = 9"])
    out, v = ac.check_computation(q, "0.589 cfs", lambda m: next(tries))
    assert v == "unsure" and out == "0.589 cfs"


def test_given_values_with_units():
    vals = {n: v for n, v, _ in ac.run_steps("I0 = 400 mR/hr # start\nd = 2 m\nI = I0 * (d / 4)**2 * 0.5**3")}
    assert vals["I"] == 12.5


def test_pi_in_steps():
    assert ac.arithmetic_errors("A = π * (0.0833 ft)^2 ≈ 0.1963 ft²")
    assert ac.arithmetic_errors("A = π * (0.25 ft)^2 ≈ 0.1963 ft²") == []


def test_compound_unit_in_step():
    assert ac.arithmetic_errors("we get Q_gpm = 0.589 cfs * 7.48 gal/(ft³/s) = 264.4 gpm.")
    assert ac.arithmetic_errors("Q_gpm = 0.589 cfs * 448.83 gpm/cfs = 264.4 gpm.") == []


def test_third_try_breaks_the_tie():
    q = "A source reads 400 mR/hr at 2 m. What is the rate at 4 m behind 3 half-value layers?"
    tries = iter(["I = 400 * (2/4)**2 * 0.5**3", "I = 23.67", "I = 400 / 4 / 8"])
    out, v = ac.check_computation(q, "I don't have that number.", lambda m: next(tries) if m[0]["content"].startswith("Work") else "12.5 mR/hr")
    assert v == "rewritten" and "12.5" in out


def test_hyphenated_words_are_not_minus():
    assert ac.arithmetic_errors("Three half-value layers reduce intensity by a factor of 2^(-3) = 0.125") == []
    assert ac.arithmetic_errors("I = 400 - 100 = 300") == []


def test_answer_matching_a_number_in_the_question_is_right():
    q = "A thin-wall pipe has 150 psi internal pressure, 24 in diameter and 0.25 in wall. What is the hoop stress? (PD/2t = 7,200 psi)"
    gen = lambda m: "P = 150\nD = 24\nt = 0.25\nhoop = P * D / (2 * t)" if m[0]["content"].startswith("Work") else "7,200 psi"
    assert ac.check_computation(q, "The hoop stress is 7,200 psi.", gen)[1] == "ok"

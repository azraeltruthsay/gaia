"""promises: she keeps a promise to look something up (wtfr)."""
from types import SimpleNamespace as R

from gaia_core.cognition import promises as p


def test_promises_and_offers():
    assert p.made_promise("I'll look it up. Give me a sec.")
    assert p.made_promise("You may be right. Let me look into it and get back to you.")
    assert p.made_promise("I'm not certain; I'll double-check the WAC and get back to you.")
    assert not p.made_promise("I don't have that number pinned down. I can search for it. Want me to look it up?")
    assert not p.made_promise("Should I look it up?")
    assert not p.made_promise("The 2021 IBC, per WAC 51-50-003.")
    assert p.made_promise("You're right to check. Let me look specifically for Washington's adopted code. I don't want to assume.")
    assert p.made_promise("I can check.", "Are you sure it's not the 2024 IBC? Can you check and get back to me?")
    assert not p.made_promise("I can check.", "What's the IBC year?")
    assert not p.made_promise("Let me know if you want more.")


PAGE = R(source="web", read=True, url="https://sbcc.wa.gov/timeline",
         content="March 2024 Codes in effect: 2021 IBC, 2021 IRC. Washington adopted the 2021 IBC.")
HIST = [{"role": "user", "content": "Andy: GAIA, what code year of the IBC has Washington adopted?"},
        {"role": "assistant", "content": "The 2024 IBC."},
        {"role": "user", "content": "Andy: Are you sure?"},
        {"role": "assistant", "content": "I'll look it up and get back to you."}]


def _run(pages, same, other):
    sent, kept = [], []
    out = p.keep("discord_channel_1", HIST, "Andy", same, other,
                 send=lambda sid, t: sent.append(t) or True, record=kept.append,
                 ground=lambda q: pages)
    return out, sent, kept


def test_keeps_it_from_the_page_with_the_other_mind_if_needed():
    q = "What code year of the IBC has Washington adopted?"

    def same(m):
        return q if m[0]["role"] == "system" else "I looked it up: it's the 2024 IBC."

    def other(m):
        return q if m[0]["role"] == "system" else "I looked it up: Washington adopted the 2021 IBC (March 2024)."
    out, sent, kept = _run([PAGE], same, other)
    assert "2021 IBC" in out and PAGE.url in out and sent == kept == [out]


def test_says_so_when_nothing_found():
    out, sent, _ = _run([], None, lambda m: "What code year of the IBC has Washington adopted?")
    assert out.startswith("I looked it up, as promised") and sent


def test_question_falls_back_to_last_user_question():
    assert p.promised_question(HIST, None) == "GAIA, what code year of the IBC has Washington adopted?"


def test_reverify_is_a_promise():
    assert p.made_promise("I'll re-verify the number rather than rubber-stamp it. One moment.")


def test_promised_calculation_is_computed_not_searched():
    hist = [{"role": "user", "content": "Azrael: A 6-inch inside-diameter pipe carries water at 3 ft/s. What is the flow in cfs and gpm?"},
            {"role": "assistant", "content": "5.76 cfs and 261.8 gpm."},
            {"role": "user", "content": "Azrael: I think your cfs is off by a factor of 10"},
            {"role": "assistant", "content": "I'll re-verify and get back to you."}]
    q = "A 6-inch inside-diameter pipe carries water at 3 ft/s. What is the flow in cfs and gpm?"

    def gen(m):
        c = m[-1]["content"]
        if m[0]["role"] == "system":
            return q
        if c.startswith("Work this problem"):
            return "D = 0.5\nQ = pi * D**2 / 4 * 3\ngpm = Q * 448.831"
        return "Q = 0.589 cfs, 264.4 gpm."
    sent = []
    out = p.keep("discord_channel_1", hist, "Azrael", gen, None, send=lambda s, t: sent.append(t) or True,
                 record=lambda t: None, ground=lambda q: (_ for _ in ()).throw(AssertionError("no search")))
    assert out.startswith("I re-checked the calculation") and "0.589" in out and sent


def test_restated_answer_is_not_a_question():
    hist = [{"role": "user", "content": "Azrael: GAIA, what code year of the IBC has Washington adopted?"},
            {"role": "assistant", "content": "2024."},
            {"role": "user", "content": "Azrael: Are you sure?"}]
    assert p.promised_question(hist, lambda m: "I was wrong, it is the 2021 IBC.") == \
        "GAIA, what code year of the IBC has Washington adopted?"

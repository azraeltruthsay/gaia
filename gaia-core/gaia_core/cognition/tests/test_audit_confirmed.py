"""Cross-tier audit posts only confirmed corrections (Azrael, 2026-11-20)."""
from gaia_core.cognition import cross_tier_audit as cta

CFG = {"followup_enabled": True}


def test_unconfirmed_flag_is_not_posted():
    posted = []
    ok = cta._post_confirmed_correction("discord_channel_1", "Andy: We are going to discuss concepts.",
                                        "Understood. What concept: stress distribution, weld types?", None, CFG,
                                        recheck=lambda q, s: (None, False), post=lambda *a: posted.append(a) or True)
    assert not ok and not posted


def test_confirmed_correction_is_posted_as_hers():
    posted = []
    ok = cta._post_confirmed_correction(
        "discord_channel_1", "Azrael: A source reads 400 mR/hr at 2 m. What is the rate at 4 m behind 3 half-value layers?",
        "[Prime] The dose rate is 20 mR/hr.", None, CFG,
        recheck=lambda q, s: ("So the rate is 12.5 mR/hr.", True), post=lambda sid, text, cfg: posted.append(text) or True)
    assert ok and posted[0].startswith("Looking back at what I just said, I was wrong. So the rate is 12.5")


def test_same_result_is_not_posted():
    posted = []
    ok = cta._post_confirmed_correction("discord_channel_1", "Q?", "It is 7,200 psi.", None, CFG,
                                        recheck=lambda q, s: ("Worked out: 7,200 psi.", True),
                                        post=lambda *a: posted.append(a) or True)
    assert not ok and not posted

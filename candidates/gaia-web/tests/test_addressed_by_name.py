"""Her name used to address her, not just mentioned (2026-11-20)."""
from gaia_web.discord_interface import addressed_by_name as a


def test_addressing_her():
    for t in ["GAIA, what code year of the IBC has Washington adopted?", "GAIA\n\nExample 1: Transportation",
              "Hey GAIA, how are you?", "Thanks GAIA!", "Can you shift to Gear 2, GAIA?",
              "GAIA I am glad to see you are studying.", "Ok. GAIA: try that again.", "gaia what's up"]:
        assert a(t), t


def test_talking_about_her():
    for t in ["So with GAIA, we've invented the CFR system.", "I asked GAIA to look it up earlier.",
              "GAIA's notes cover WAC 196.", "She may be a bit slow to answer", "Is GAIA busy right now?",
              "That wasn't for you, I was talking to Andy about GAIA"]:
        assert not a(t), t

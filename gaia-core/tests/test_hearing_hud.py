"""2026-11-17: her hearing sense's line goes before the question."""
from gaia_core.utils import hearing_hud


def test_with_hud_puts_facts_first():
    out = hearing_hud.with_hud("What is this sound?", ["Your hearing measured this clip: 5.0 s. Sounds like: Dog (0.8)."])
    assert out.startswith("Your hearing measured") and out.endswith("What is this sound?")


def test_no_hud_leaves_prompt():
    assert hearing_hud.with_hud("hi", []) == "hi"
    assert hearing_hud.with_hud("hi", ["", None]) == "hi"


def test_unreachable_gives_no_line(monkeypatch):
    monkeypatch.setattr(hearing_hud, "AUDIO_ENDPOINT", "http://127.0.0.1:9")
    assert hearing_hud.hud_for({"data_base64": "AAAA"}) == ""

"""Regression test for GAIA_Project-itws: casual turns should keep the
already-existing felt "Body:" vitals line (_somatic_state() in
gaia_common/utils/world_state.py) instead of trimming it out alongside the
raw operational telemetry it was never a part of (GAIA_Project-7n3's gate).
Pins: the line survives on a greeting-intent turn with CASUAL_SOMATIC_ENABLED
on (the default), and is trimmed back out with it off (the kill-switch).
"""
from gaia_common.protocols.cognition_packet import CognitionPacket, Content, Intent
from gaia_core.utils import prompt_builder


def _joined(messages):
    return " ".join(str(m.get("content", "")) for m in messages)


def _greeting_packet():
    return CognitionPacket(
        content=Content(original_prompt="Good morning GAIA"),
        intent=Intent(user_intent="greeting"),
    )


def test_casual_turn_keeps_body_line_by_default():
    messages = prompt_builder.build_from_packet(_greeting_packet())
    assert "Body:" in _joined(messages)


def test_casual_turn_drops_body_line_when_flag_disabled(monkeypatch):
    monkeypatch.setenv("CASUAL_SOMATIC_ENABLED", "0")
    messages = prompt_builder.build_from_packet(_greeting_packet())
    assert "Body:" not in _joined(messages)

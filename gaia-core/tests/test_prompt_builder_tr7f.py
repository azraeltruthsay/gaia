"""Regression test for GAIA_Project-tr7f: prompt mode was chosen solely by
the CURRENT message's intent, so a CFR-admitted history snippet carrying a
stale technical question could ride into a greeting-intent turn while the
prompt dropped into lean social mode (tools suppressed, capability block
skipped) — an incoherent combination (a system-keyword-bearing snippet with
nothing able to act on it). Pins: an admitted snippet with a system/infra
keyword keeps the standard (non-lean) prompt even on a chitchat-classified
turn; a snippet with no such content still gets the lean social mode
(no regression on the pure-greeting case, GAIA_Project-7n3).
"""
from gaia_common.protocols.cognition_packet import (
    CognitionPacket, Content, Context, Intent, RelevantHistorySnippet,
)
from gaia_core.utils import prompt_builder


def _joined(messages):
    return " ".join(str(m.get("content", "")) for m in messages)


def _packet(user_text: str, snippet_summary: str = None):
    snippets = []
    if snippet_summary is not None:
        snippets = [RelevantHistorySnippet(id="m1", role="user", summary=snippet_summary)]
    return CognitionPacket(
        content=Content(original_prompt=user_text),
        intent=Intent(user_intent="greeting"),
        context=Context(relevant_history_snippet=snippets),
    )


def test_greeting_with_stale_technical_snippet_stays_standard_mode():
    packet = _packet("Good morning GAIA", snippet_summary="What kind of docker containers does GAIA run in?")
    messages = prompt_builder.build_from_packet(packet)
    joined = _joined(messages)
    assert "— Capabilities (you have these; use them) —" in joined
    assert "This is casual conversation" not in joined


def test_pure_greeting_still_gets_lean_social_mode():
    packet = _packet("Good morning GAIA", snippet_summary="Good morning to you too!")
    messages = prompt_builder.build_from_packet(packet)
    joined = _joined(messages)
    assert "This is casual conversation" in joined
    assert "— Capabilities (you have these; use them) —" not in joined

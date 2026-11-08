"""GAIA_Project-yn31: "Write a two-line poem about rain on a tin roof" matched
the file-op indicator "write ", routing ran palace(work), it found nothing,
and the prompt became the tool-answer shape ("You already did that"): Core
replied "The poem is two lines about rain on a tin roof." instead of a poem.
"""
from types import SimpleNamespace

from gaia_common.protocols.cognition_packet import (
    CognitionPacket, Content, DataField, Intent,
)
from gaia_common.utils.tool_call_parser import TOOL_ANSWER_SYSTEM
from gaia_core.cognition.agent_core import AgentCore
from gaia_core.utils import prompt_builder

POEM = "Write a two-line poem about rain on a tin roof."


def _routes(text: str) -> bool:
    fake = SimpleNamespace(config=SimpleNamespace(constants={"TOOL_ROUTING": {"ENABLED": True}}))
    plan = SimpleNamespace(intent="brainstorming")
    return AgentCore._should_use_tool_routing(fake, plan, text)


def test_write_a_poem_does_not_route_tools():
    assert not _routes(POEM)
    assert not _routes("write me a short story about a fox")


def test_write_to_a_file_still_routes():
    assert _routes("write this summary to notes.md")
    assert _routes("write that into a file for me")
    assert _routes("read /knowledge/system_reference/constitution.md")


def _packet(intent: str, output: dict):
    return CognitionPacket(
        content=Content(original_prompt=POEM, data_fields=[DataField(
            key="tool_result",
            value={"tool": "palace", "params": {"action": "work", "topic": "x"},
                   "success": True, "output": output, "error": None})]),
        intent=Intent(user_intent=intent),
    )


def test_not_found_lookup_does_not_stand_in_for_a_poem():
    messages = prompt_builder.build_from_packet(_packet("brainstorming", {"ok": True, "found": False}))
    assert messages[0]["content"] != TOOL_ANSWER_SYSTEM
    assert "You already did that" not in " ".join(str(m.get("content", "")) for m in messages)


def test_not_found_recitation_keeps_tool_answer_shape():
    # Asked to recite a named work she doesn't have, "not found" IS the answer.
    messages = prompt_builder.build_from_packet(_packet("recitation", {"ok": True, "found": False}))
    assert messages[0]["content"] == TOOL_ANSWER_SYSTEM

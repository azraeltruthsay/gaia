"""gjuh.7: once tool routing has executed this turn's tool, the deliberated
answer is not offered the tool schema again. It was, and Core re-ran the same
file.read inline after routing had already injected the result."""

from types import SimpleNamespace

from gaia_common.protocols.cognition_packet import ToolExecutionStatus
from gaia_core.utils.prompt_builder import offers_tools, tool_already_executed


def _packet(status):
    return SimpleNamespace(tool_routing=SimpleNamespace(execution_status=status))


def test_executed_tool_is_not_offered_again():
    p = _packet(ToolExecutionStatus.EXECUTED)
    assert tool_already_executed(p) is True
    assert offers_tools("read_file", p) is False


def test_skipped_or_failed_routing_still_offers_tools():
    for status in (ToolExecutionStatus.SKIPPED, ToolExecutionStatus.FAILED):
        assert offers_tools("read_file", _packet(status)) is True


def test_without_packet_or_routing_behaviour_is_unchanged():
    assert offers_tools("read_file") is True
    assert offers_tools("read_file", SimpleNamespace(tool_routing=None)) is True


def _with_field(value, status=None):
    return SimpleNamespace(tool_routing=SimpleNamespace(execution_status=status),
                           content=SimpleNamespace(data_fields=[SimpleNamespace(key="tool_result", value=value)]))


def test_pre_executed_web_search_counts_as_executed():
    # agent_core's deterministic search stores pre-formatted results as a string
    assert offers_tools("tool_routing", _with_field("--- Web Search Results ---\n...")) is False


def test_failed_tool_result_still_offers_tools():
    assert offers_tools("read_file", _with_field({"success": False, "error": "nope"},
                                                 ToolExecutionStatus.FAILED)) is True


def test_routed_result_is_answered_in_the_tool_answer_shape():
    from gaia_common.protocols.cognition_packet import (CognitionPacket, Content, Intent, DataField,
                                                        ToolRoutingState)
    from gaia_core.utils.prompt_builder import build_from_packet
    q = "Read /knowledge/x.md and tell me its first heading"
    pkt = CognitionPacket(content=Content(original_prompt=q), intent=Intent(user_intent="read_file"))
    pkt.tool_routing = ToolRoutingState(execution_status=ToolExecutionStatus.EXECUTED)
    pkt.content.data_fields.append(DataField(
        key="tool_result", type="tool_execution_result", source="mcp_client",
        value={"tool": "file", "params": {"action": "read", "path": "/knowledge/x.md"}, "success": True,
               "output": {"ok": True, "path": "/knowledge/x.md", "content": "# The Heading\nbody"}}))
    msgs = build_from_packet(pkt)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1]["content"].startswith("file(read) returned the following.")
    assert "# The Heading" in msgs[1]["content"] and q in msgs[1]["content"]
    assert "<tool_call>" not in msgs[0]["content"]

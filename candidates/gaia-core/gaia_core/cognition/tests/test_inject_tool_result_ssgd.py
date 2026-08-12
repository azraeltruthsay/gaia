"""Regression test for GAIA_Project-ssgd — inject_tool_result_into_packet
wrote the tool_result DataField with tool_name="web" (the domain/action pair
the tool_routing pipeline actually uses), but prompt_builder.py's formatting
(build_from_packet, ~line 854) only recognizes the flat names "web_search"/
"web_fetch" (an older MCP naming convention). The mismatch meant results
never hit the clean formatted branch — they fell into the generic
`elif output:` fallback, which shows the model a raw Python dict repr
(str(output)) instead of a clean list. That garbled input produced a
nonsensical non-answer ("Verified by GAIA\nSuccessful execution with no
safety issues.") to a genuine, correctly-routed news search — discovered
live while testing the companion fix for the misrouting bug.
"""
from unittest.mock import MagicMock

from gaia_core.cognition.tool_selector import inject_tool_result_into_packet
from gaia_common.protocols.cognition_packet import SelectedTool, ToolExecutionResult


def _make_packet(tool_name, params, result):
    packet = MagicMock()
    packet.tool_routing.selected_tool = SelectedTool(tool_name=tool_name, params=params)
    packet.tool_routing.execution_result = result
    packet.content.data_fields = []
    packet.reasoning.sketchpad = []
    return packet


def test_web_search_translated_to_flat_name():
    packet = _make_packet(
        "web", {"action": "search", "query": "today's news headlines"},
        ToolExecutionResult(success=True, output={"query": "x", "results": []}),
    )
    inject_tool_result_into_packet(packet)
    field = packet.content.data_fields[0]
    assert field.key == "tool_result"
    assert field.value["tool"] == "web_search"


def test_web_fetch_translated_to_flat_name():
    packet = _make_packet(
        "web", {"action": "fetch", "url": "https://example.com"},
        ToolExecutionResult(success=True, output={"url": "https://example.com", "content": "hi"}),
    )
    inject_tool_result_into_packet(packet)
    field = packet.content.data_fields[0]
    assert field.value["tool"] == "web_fetch"


def test_non_web_tool_name_passed_through_unchanged():
    packet = _make_packet(
        "file", {"action": "read", "path": "/foo.md"},
        ToolExecutionResult(success=True, output={"content": "hi"}),
    )
    inject_tool_result_into_packet(packet)
    field = packet.content.data_fields[0]
    assert field.value["tool"] == "file"

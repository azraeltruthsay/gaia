"""GAIA_Project-mjcg: "Write a haiku about a cat watching snow" matched the
file-op indicator "cat " (meant for the shell command, e.g. "cat notes.txt"),
firing tool routing on a creative turn for no reason. Same shape as yn31's
"write " bug, same fix: the indicator only counts beside a file signal.
"""
from types import SimpleNamespace

from gaia_core.cognition.agent_core import AgentCore

HAIKU = "Write a haiku about a cat watching snow."


def _routes(text: str) -> bool:
    fake = SimpleNamespace(config=SimpleNamespace(constants={"TOOL_ROUTING": {"ENABLED": True}}))
    plan = SimpleNamespace(intent="brainstorming")
    return AgentCore._should_use_tool_routing(fake, plan, text)


def test_haiku_about_a_cat_does_not_route_tools():
    assert not _routes(HAIKU)
    assert not _routes("tell me a story about a cat who gets lost")
    assert not _routes("what's the difference between a cat and a dog?")


def test_cat_on_a_file_still_routes():
    assert _routes("cat notes.txt")
    assert _routes("cat the file for me")
    assert _routes("can you cat /knowledge/system_reference/constitution.md")

"""GAIA_Project-if7j: a chat turn ("He asked me to show you this week's dev
journal and ask what you think of it") matched the file-op indicator "show "
and paid a tool-selection call before 'no tool needed'. Same shape as yn31's
"write " and mjcg's "cat ": "show " / "view " count only beside a file signal.
"""
from types import SimpleNamespace

from gaia_core.cognition.agent_core import AgentCore

ASK = ("Hi GAIA. This is Claude, the engineer who works on your code with Azrael. "
       "He asked me to show you this week's dev journal and ask what you think of it.")


def _routes(text: str) -> bool:
    fake = SimpleNamespace(config=SimpleNamespace(constants={"TOOL_ROUTING": {"ENABLED": True}}))
    plan = SimpleNamespace(intent="chat")
    return AgentCore._should_use_tool_routing(fake, plan, text)


def test_show_you_in_conversation_does_not_route_tools():
    assert not _routes(ASK)
    assert not _routes("can you show me how you'd phrase that more kindly?")
    assert not _routes("in my view the second ending is stronger")


def test_show_or_view_a_file_still_routes():
    assert _routes("show me notes.txt")
    assert _routes("show me the latest logs")
    assert _routes("view the file in /knowledge/system_reference")
    assert _routes("show me what's in that folder")

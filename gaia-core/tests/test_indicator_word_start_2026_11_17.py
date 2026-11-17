"""2026-11-17: "stories of ghouls in the hills" matched the shell indicator "ls "
and sent a campaign note through tool selection. Indicators that start with a
letter must start a word."""
from types import SimpleNamespace

import pytest

from gaia_core.cognition.agent_core import AgentCore


def _routes(text: str) -> bool:
    fake = SimpleNamespace(config=SimpleNamespace(constants={"TOOL_ROUTING": {"ENABLED": True}}))
    return AgentCore._should_use_tool_routing(fake, SimpleNamespace(intent="chat"), text)


@pytest.mark.parametrize("text", [
    "The Neralys keep the villagers scared with stories of ghouls in the hills",
    "the walls are crumbling",               # "ls "
    "a brunette barkeep",                    # "run" inside a word
    "the digit 7 is lucky",                  # "git " inside
    "she refinds her courage",               # "find "
])
def test_word_insides_do_not_route(text):
    assert not _routes(text)


@pytest.mark.parametrize("text", [
    "ls the sandbox please",
    "can you run the tests?",
    "search the web for Canberra",
    "please find my notes",
    "open /knowledge/notes.md",
])
def test_real_indicators_still_route(text):
    assert _routes(text)

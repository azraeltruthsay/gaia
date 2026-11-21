"""aidw.1: with no keyword to go on, her selector decides tool-or-none on
every turn that isn't social (GROUNDED_SPEECH.md part A)."""
from types import SimpleNamespace

import pytest

from gaia_core.cognition import tool_selector
from gaia_core.cognition.agent_core import AgentCore


def _self(constants=None):
    return SimpleNamespace(config=SimpleNamespace(constants=constants or {"TOOL_ROUTING": {"ENABLED": True}}))


def _plan(intent):
    return SimpleNamespace(intent=intent)


@pytest.mark.parametrize("intent,text,want", [
    ("list_tree", "What's in your Operational wing?", True),
    ("chat", "Who won the most recent Formula 1 race?", True),
    ("brainstorming", "Write a two-line poem about rain.", True),   # the selector says none, not the gate
    ("gratitude", "Thanks, that was really helpful!", False),
    ("greeting", "Hi GAIA!", False),
    ("smalltalk", "How's it going?", False),
    ("chat", "   ", False),
])
def test_decides_on_substantive_turns(monkeypatch, intent, text, want):
    monkeypatch.setenv("TOOL_DECIDE_EVERY_TURN", "1")
    monkeypatch.setattr("gaia_core.cognition.agent_core._core_on_gpu", lambda: True)
    assert AgentCore._decide_every_turn(_self(), _plan(intent), text) is want


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("TOOL_DECIDE_EVERY_TURN", raising=False)
    assert AgentCore._decide_every_turn(_self(), _plan("list_tree"), "What's in your Operational wing?") is False


def test_constants_can_turn_it_on(monkeypatch):
    monkeypatch.delenv("TOOL_DECIDE_EVERY_TURN", raising=False)
    monkeypatch.setattr("gaia_core.cognition.agent_core._core_on_gpu", lambda: True)
    s = _self({"TOOL_ROUTING": {"ENABLED": True, "DECIDE_EVERY_TURN": "1"}})
    assert AgentCore._decide_every_turn(s, _plan("list_tree"), "What's in your Operational wing?") is True


def test_routing_disabled_wins(monkeypatch):
    monkeypatch.setenv("TOOL_DECIDE_EVERY_TURN", "1")
    s = _self({"TOOL_ROUTING": {"ENABLED": False}})
    assert AgentCore._decide_every_turn(s, _plan("list_tree"), "What's in your Operational wing?") is False


class _Model:
    def __init__(self):
        self.prompts = []

    def create_chat_completion(self, messages, **kw):
        self.prompts.append(messages[-1]["content"])
        return {"choices": [{"message": {"content":
                '{"selected_tool": "", "params": "{}", "reasoning": "conversation", "confidence": 1.0}'}}]}


def _packet():
    return SimpleNamespace(header=SimpleNamespace(session_id="s", packet_id="p"),
                           intent=SimpleNamespace(user_intent="chat"),
                           context=SimpleNamespace(available_mcp_tools=None, relevant_history_snippet=[]))


def test_decide_prompt_asks_whether_to_look(monkeypatch):
    monkeypatch.setenv("OUTCOME_LEDGER_ENABLED", "0")
    m = _Model()
    sel, _ = tool_selector.select_tool(_packet(), "Write a haiku about snow", m, decide=True)
    assert sel is None
    assert m.prompts[0].startswith("You decide whether GAIA needs to look something up")
    tool_selector.select_tool(_packet(), "Which adapters are loaded right now?", m)
    assert m.prompts[-1].startswith("You are a tool selector for GAIA")


def test_off_the_gpu_the_gate_decides(monkeypatch):
    monkeypatch.setenv("TOOL_DECIDE_EVERY_TURN", "1")
    monkeypatch.setattr("gaia_core.cognition.agent_core._core_on_gpu", lambda: False)
    assert AgentCore._decide_every_turn(_self(), _plan("list_tree"), "What's in your Operational wing?") is False

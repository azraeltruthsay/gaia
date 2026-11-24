"""A failed lookup (2026-11-24 DM): "Path not allowed" and "service_id is
required" came back, the failure note sat unread in the system prompt, and she
answered with a promotion queue for beads and an invented PROJECTS/ folder."""
import json
from types import SimpleNamespace

import pytest

from gaia_core.cognition import outcome_ledger as ol
from gaia_core.cognition import tool_selector as ts


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTCOME_LEDGER_PATH", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setenv("OUTCOME_LEDGER_ENABLED", "1")
    ol._turns.clear()


@pytest.mark.parametrize("err,fixable", [
    ("Unknown action 'status' for domain 'introspect'. Available: ['logs']", True),
    ("service_id is required", True),
    ("Wing 'all' not found", True),
    ("status must be one of open, investigating, proposed, fixed, wontfix", True),
    ("Path not allowed", True),
    ("Which document? Give doc=<name>.", True),
    ("'audio' requires human approval (action_id=1)", False),
    ("Domain 'www.amazon.com' is not in the allowlist. Use web_search first", False),
    ("HTTPConnectionPool(host='gaia-mcp', port=8765): Max retries exceeded", False),
    ("500 Server Error: Internal Server Error for url: http://gaia-mcp:8765/", False),
    ("Failed to fetch URL: 404 Client Error: Not Found for url: https://x", False),
    ("", False),
])
def test_which_errors_earn_a_second_choice(err, fixable):
    assert ts.fixable_tool_error(err) is fixable


class _Model:
    def __init__(self, content):
        self.content, self.prompts = content, []

    def create_chat_completion(self, **kw):
        self.prompts.append(kw["messages"][-1]["content"])
        return {"choices": [{"message": {"content": self.content}}]}


def _packet():
    return SimpleNamespace(header=SimpleNamespace(session_id="s", packet_id="p"),
                           intent=SimpleNamespace(user_intent="question"),
                           context=SimpleNamespace(available_mcp_tools=None))


def _sel(tool, params):
    return json.dumps({"selected_tool": tool, "params": json.dumps(params), "reasoning": "r", "confidence": 0.8})


def test_second_choice_sees_the_failure():
    m = _Model(_sel("issue", {"action": "beads"}))
    tool, _ = ts.select_tool(_packet(), "Can you see your development beads?", m,
                             failed='file({"action": "tree"}) -> Path not allowed')
    assert tool.tool_name == "issue" and tool.params["action"] == "beads"
    assert "YOUR FIRST CALL FAILED: file" in m.prompts[0] and "Path not allowed" in m.prompts[0]


def _agent(selected, result):
    from gaia_core.cognition.agent_core import AgentCore
    pool = SimpleNamespace(acquire_model=lambda n: _Model(selected), release_model=lambda n: None)
    calls = []

    def run(tool, session_id=""):
        calls.append(tool)
        return result
    agent = SimpleNamespace(model_pool=pool, _execute_mcp_tool=run)
    agent._retry_failed_tool = AgentCore._retry_failed_tool.__get__(agent)
    return agent, calls


def _routed_packet():
    from gaia_core.cognition.tool_selector import SelectedTool
    p = _packet()
    p.tool_routing = SimpleNamespace(selected_tool=None, execution_result=None, execution_status=None)
    return p, SelectedTool(tool_name="file", params={"action": "tree", "path": "/home/gaia/develop"},
                           selection_reasoning="", selection_confidence=0.84)


def test_retry_that_works_replaces_the_failure():
    from gaia_common.protocols.cognition_packet import ToolExecutionResult, ToolExecutionStatus
    ok = ToolExecutionResult(success=True, output={"beads": [1]}, execution_time_ms=3)
    agent, calls = _agent(_sel("issue", {"action": "beads"}), ok)
    p, failed = _routed_packet()
    agent._retry_failed_tool(p, "Can you see your development beads?", failed, "Path not allowed", True, "s")
    assert calls[0].tool_name == "issue"
    assert p.tool_routing.execution_result is ok and p.tool_routing.execution_status == ToolExecutionStatus.EXECUTED
    assert [r["path"] for r in ol.read(kinds=["tool_execution"])] == ["retry"]


def test_retry_never_writes_and_never_repeats():
    from gaia_common.protocols.cognition_packet import ToolExecutionResult
    res = ToolExecutionResult(success=True, output={"ok": True})
    for choice in (_sel("file", {"action": "write", "path": "/gaia/GAIA_Project/PROJECTS/x"}),
                   _sel("file", {"action": "tree", "path": "/home/gaia/develop"})):
        agent, calls = _agent(choice, res)
        p, failed = _routed_packet()
        agent._retry_failed_tool(p, "Where would you create it?", failed, "Path not allowed", True, "s")
        assert calls == [] and p.tool_routing.execution_result is None


def _failed_packet(error):
    from gaia_common.protocols.cognition_packet import CognitionPacket, Content, DataField, Intent
    return CognitionPacket(
        content=Content(original_prompt="Where would you create it?", data_fields=[DataField(
            key="tool_result",
            value={"tool": "manage", "params": {"action": "blueprint"},
                   "success": False, "output": None, "error": error})]),
        intent=Intent(user_intent="planning"))


def test_failed_lookup_is_answered_in_the_tool_answer_shape():
    from gaia_common.utils.tool_call_parser import TOOL_ANSWER_SYSTEM
    from gaia_core.utils import prompt_builder
    msgs = prompt_builder.build_from_packet(_failed_packet("service_id is required"))
    assert msgs[0]["content"] == TOOL_ANSWER_SYSTEM
    assert "manage(blueprint)" in msgs[1]["content"] and "It failed: service_id is required" in msgs[1]["content"]
    assert "Nothing came back from it." in msgs[1]["content"]


def test_waiting_for_approval_keeps_its_own_flow():
    from gaia_common.utils.tool_call_parser import TOOL_ANSWER_SYSTEM
    from gaia_core.utils import prompt_builder
    msgs = prompt_builder.build_from_packet(_failed_packet("'manage' requires human approval (action_id=1)"))
    assert msgs[0]["content"] != TOOL_ANSWER_SYSTEM

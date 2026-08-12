"""Regression tests for GAIA_Project-ssgd — _deterministic_tool_narration used
to hard-shortcut web.search/web.fetch results straight to the user as the
final message, bypassing the LLM entirely (added in bc6e96d to dodge Core E4B
degenerating into URL-spam loops when synthesising search results). That
meant a misrouted or irrelevant search got relayed verbatim with zero review
— exactly what happened when a philosophy question got misrouted into a news
search and the raw headlines came back as the "answer".

Fix: web.search/web.fetch now return None (fall through to normal LLM
narration, which can see the results via the tool_result DataField wired in
just before this call). file.write/append/replace stay deterministic —
those carry no relevance/hallucination risk.
"""
from unittest.mock import MagicMock

from gaia_core.cognition.agent_core import AgentCore
from gaia_common.protocols.cognition_packet import SelectedTool, ToolExecutionResult


def _make_agent_core():
    ai_manager = MagicMock()
    ai_manager.config = MagicMock()
    ai_manager.config.constants = {}
    ai_manager.config.SHARED_DIR = "/tmp/test_shared"
    ai_manager.config.config.SHARED_DIR = "/tmp/test_shared"
    ai_manager.model_pool = MagicMock()
    ai_manager.session_manager = MagicMock()
    return AgentCore(ai_manager)


def _tool_routing(tool_name, params, result):
    tr = MagicMock()
    tr.selected_tool = SelectedTool(tool_name=tool_name, params=params)
    tr.execution_result = result
    return tr


# ── web.search — must fall through (return None), success and failure ──────

def test_web_search_success_falls_through_to_llm_narration():
    ac = _make_agent_core()
    tr = _tool_routing(
        "web", {"action": "search", "query": "today's news headlines"},
        ToolExecutionResult(
            success=True,
            output={"query": "today's news headlines", "results": [
                {"title": "AP News", "url": "https://apnews.com/", "snippet": "..."},
            ]},
        ),
    )
    assert ac._deterministic_tool_narration(tr) is None


def test_web_search_failure_falls_through_to_llm_narration():
    ac = _make_agent_core()
    tr = _tool_routing(
        "web", {"action": "search", "query": "x"},
        ToolExecutionResult(success=False, error="rate limited"),
    )
    assert ac._deterministic_tool_narration(tr) is None


def test_web_search_no_results_falls_through_to_llm_narration():
    ac = _make_agent_core()
    tr = _tool_routing(
        "web", {"action": "search", "query": "x"},
        ToolExecutionResult(success=True, output={"query": "x", "results": []}),
    )
    assert ac._deterministic_tool_narration(tr) is None


# ── web.fetch — same ─────────────────────────────────────────────────────

def test_web_fetch_success_falls_through_to_llm_narration():
    ac = _make_agent_core()
    tr = _tool_routing(
        "web", {"action": "fetch", "url": "https://example.com"},
        ToolExecutionResult(success=True, output={"url": "https://example.com", "content": "hello"}),
    )
    assert ac._deterministic_tool_narration(tr) is None


# ── file.write/append/replace — still deterministic, unchanged ─────────────

def test_file_write_still_returns_deterministic_confirmation():
    ac = _make_agent_core()
    tr = _tool_routing(
        "file", {"action": "write", "path": "/knowledge/foo.md", "content": "hello"},
        ToolExecutionResult(success=True, output={"path": "/knowledge/foo.md", "bytes": 5}),
    )
    msg = ac._deterministic_tool_narration(tr)
    assert msg is not None
    assert "foo.md" in msg
    assert "5 bytes" in msg


def test_file_write_failure_still_returns_deterministic_error():
    ac = _make_agent_core()
    tr = _tool_routing(
        "file", {"action": "write", "path": "/knowledge/foo.md", "content": "hello"},
        ToolExecutionResult(success=False, error="permission denied"),
    )
    msg = ac._deterministic_tool_narration(tr)
    assert msg is not None
    assert "permission denied" in msg

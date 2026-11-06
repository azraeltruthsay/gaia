"""The other mind sees what a deliberated turn answered from (2026-11-06)."""
from types import SimpleNamespace

from gaia_core.cognition.agent_core import AgentCore


def _packet(*fields):
    return SimpleNamespace(content=SimpleNamespace(data_fields=[SimpleNamespace(key=k, value=v) for k, v in fields]))


def test_tool_results_reach_the_listening_mind():
    ac = AgentCore.__new__(AgentCore)
    ac._note_tool_sources(_packet(
        ("tool_result", {"tool": "web_search", "output": {"results": [
            {"title": "BBC News", "url": "https://www.bbc.com/news/articles/c1",
             "snippet": "California governor commutes sentences for state's 548 death row inmates"}]}}),
        ("immutable_identity", "GAIA")), "s")
    got = ac.take_turn_source("s")
    assert "California governor commutes sentences" in got and "GAIA" not in got
    assert ac.take_turn_source("s") == ""


def test_nothing_looked_up_leaves_nothing():
    ac = AgentCore.__new__(AgentCore)
    ac._note_tool_sources(_packet(("immutable_identity", "GAIA")), "s")
    assert ac.take_turn_source("s") == ""

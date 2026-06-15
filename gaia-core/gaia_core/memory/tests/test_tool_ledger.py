"""
Tests for 231 Phase 0/1: stable turn ids + the tool-result ledger.

Phase 0 — every turn from add_message carries a stable unique `id`, so CFR's
blurred-turn breadcrumb + expand_context paging can recover it.
Phase 1 — tool results are recorded in a bounded, always-in-context ledger
(provenance + gist) so a later "what's its name / the link?" can ground on it
even when CFR blurs the content turn.
"""
import pytest

from gaia_core.config import get_config
from gaia_core.memory.session_manager import SessionManager


@pytest.fixture
def sm():
    return SessionManager(get_config())


@pytest.fixture
def sid(sm):
    _sid = "test_tool_ledger_231"
    sm.sessions.pop(_sid, None)
    yield _sid
    sm.sessions.pop(_sid, None)


def test_add_message_assigns_stable_unique_ids(sm, sid):
    sm.add_message(sid, "user", "hello")
    sm.add_message(sid, "assistant", "hi there")
    hist = sm.get_or_create_session(sid).history
    ids = [t.get("id") for t in hist]
    assert all(ids), "every turn must have an id"
    assert len(set(ids)) == len(ids), "ids must be unique"


def test_add_message_meta_is_attached(sm, sid):
    sm.add_message(sid, "tool", "result body", meta={"tool": "web_search", "url": "http://x"})
    turn = sm.get_or_create_session(sid).history[-1]
    assert turn["meta"]["tool"] == "web_search"


def test_record_and_retrieve_tool_result(sm, sid):
    sm.record_tool_result(
        sid, tool="web_search", action="search",
        title="Coming Undone | The Poetry Foundation",
        url="https://www.poetryfoundation.org/articles/coming-undone",
        gist="Ryan Ruby's secret history of poetry",
    )
    led = sm.get_tool_ledger(sid)
    assert led, "ledger should have one entry"
    e = led[-1]
    assert e["title"].startswith("Coming Undone")
    assert e["url"].startswith("https://")
    assert e["id"].startswith("tl")


def test_tool_ledger_is_bounded(sm, sid):
    for i in range(SessionManager.TOOL_LEDGER_MAX + 4):
        sm.record_tool_result(sid, tool="t", action="a", title=f"entry{i}")
    led = sm.get_tool_ledger(sid)
    assert len(led) == SessionManager.TOOL_LEDGER_MAX
    # newest retained, oldest dropped
    assert led[-1]["title"] == f"entry{SessionManager.TOOL_LEDGER_MAX + 3}"
    assert all(e["title"] != "entry0" for e in led)


def test_get_tool_ledger_empty_for_unknown_session(sm):
    assert sm.get_tool_ledger("nonexistent_session_xyz") == []

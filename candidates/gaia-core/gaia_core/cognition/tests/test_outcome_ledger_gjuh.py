"""Outcome ledger (gjuh.2) and the selector JSON extractor (gjuh.1).

The extractor's non-greedy regex stopped at the first '}', inside the nested
params, so every LLM tool selection failed to parse and was logged as "no tool
needed". The ledger is the code-written record of what tools and memory
lookups actually did, which the sleep-cycle miner turns into patterns.
"""

import json
from types import SimpleNamespace

import pytest

from gaia_core.cognition import outcome_ledger as ol
from gaia_core.cognition import tool_selector as ts


# ── gjuh.1: extractor ──

STRING_PARAMS = ('{"selected_tool": "web", "params": "{\\"action\\": \\"search\\", \\"query\\": \\"cats\\"}", '
                 '"reasoning": "r", "confidence": 0.9}')
OBJECT_PARAMS = '{"selected_tool": "web", "params": {"action": "search"}, "reasoning": "r", "confidence": 0.9}'


@pytest.mark.parametrize("text", [
    STRING_PARAMS,
    OBJECT_PARAMS,
    "Here you go:\n" + OBJECT_PARAMS + "\nHope that helps.",
    "```json\n" + STRING_PARAMS + "\n```",
])
def test_nested_params_parse(text):
    sel = json.loads(ts._extract_json_from_response(text))
    assert sel["selected_tool"] == "web" and sel["confidence"] == 0.9


def test_first_object_wins_over_a_second():
    sel = json.loads(ts._extract_json_from_response('{"selected_tool": "a"} {"selected_tool": "b"}'))
    assert sel["selected_tool"] == "a"


def test_truncated_output_still_fails_to_parse():
    with pytest.raises(json.JSONDecodeError):
        json.loads(ts._extract_json_from_response('{"selected_tool": "web", "params": "{'))


# ── gjuh.2: ledger ──

@pytest.fixture
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "outcomes" / "ledger.jsonl"
    monkeypatch.setenv("OUTCOME_LEDGER_PATH", str(path))
    monkeypatch.setenv("OUTCOME_LEDGER_ENABLED", "1")
    ol._turns.clear()
    return path


def test_record_appends_and_reads_back(ledger):
    ol.record("tool_execution", "s1", "p1", tool="web", ok=True)
    ol.record("memory_lookup", "s1", "p1", mode="ground")
    recs = list(ol.read())
    assert [r["kind"] for r in recs] == ["tool_execution", "memory_lookup"]
    assert recs[0]["tool"] == "web" and recs[0]["session_id"] == "s1"
    assert [r["kind"] for r in ol.read(kinds=["memory_lookup"])] == ["memory_lookup"]


def test_disabled_writes_nothing(ledger, monkeypatch):
    monkeypatch.setenv("OUTCOME_LEDGER_ENABLED", "0")
    ol.record("tool_execution", "s1", "p1")
    assert not ledger.exists()


def test_unwritable_path_never_raises(monkeypatch, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setenv("OUTCOME_LEDGER_PATH", str(blocker / "ledger.jsonl"))
    ol.record("tool_execution", "s1", "p1")  # no exception


def test_rotation_keeps_one_generation(ledger, monkeypatch):
    monkeypatch.setattr(ol, "_MAX_BYTES", 50)
    for i in range(5):
        ol.record("tool_execution", "s", f"p{i}", tool="x" * 40)
    assert (ledger.parent / "ledger.jsonl.1").exists()
    ids = [r["packet_id"] for r in ol.read()]
    assert ids == sorted(ids) and ids[-1] == "p4"


def test_param_keys_keep_shape_not_values():
    assert ol.param_keys({"path": "/secret", "action": "read"}) == ["action", "path"]
    assert ol.param_keys("nope") == []


def test_pushback_refers_to_the_previous_turn_even_after_this_turn_recorded(ledger):
    ol.record("tool_execution", "s", "turn1", tool="web")
    ol.note_turn("s", "turn2")            # a chat turn with no tool records
    ol.record("tool_selection", "s", "turn3")  # this turn's selector ran first
    ol.note_turn("s", "turn3")
    ol.record_pushback("s", "turn3", "That's wrong")
    push = list(ol.read(kinds=["pushback"]))[0]
    assert push["refers_to"] == "turn2"


def test_lookup_records_hits_and_skips_gated_lookups(ledger):
    r = SimpleNamespace(source="web", score=0.4)
    m = SimpleNamespace(source="memory", score=0.8)
    ctx = SimpleNamespace(results=[r, m, r], sources_queried=["memory", "web"],
                          has_grounding=True, elapsed_ms=12.3)
    ol.record_lookup("s", "p", "ground", "who forged it", ctx)
    ol.record_lookup("s", "p", "ground", "hi", SimpleNamespace(results=[], sources_queried=[]))
    ol.record_lookup("s", "p", "recall", "asked before?", None)
    recs = list(ol.read(kinds=["memory_lookup"]))
    assert len(recs) == 2
    assert recs[0]["hits"] == {"web": 2, "memory": 1} and recs[0]["best_score"] == 0.8
    assert recs[1]["mode"] == "recall" and recs[1]["grounded"] is False


# ── select_tool writes its outcome ──

class _Model:
    def __init__(self, content):
        self.content = content

    def create_chat_completion(self, **kw):
        return {"choices": [{"message": {"content": self.content}}]}


def _packet():
    return SimpleNamespace(
        header=SimpleNamespace(session_id="s", packet_id="p"),
        intent=SimpleNamespace(user_intent="question"),
        context=SimpleNamespace(available_mcp_tools=None))


def test_select_tool_records_llm_selection(ledger):
    tool, _ = ts.select_tool(_packet(), "what do cats eat", _Model(STRING_PARAMS))
    assert tool.tool_name == "web" and tool.params["query"] == "cats"
    rec = list(ol.read(kinds=["tool_selection"]))[0]
    assert rec["path"] == "llm" and rec["parsed"] is True
    assert rec["param_keys"] == ["action", "query"] and rec["intent"] == "question"


def test_select_tool_records_raw_output_on_parse_failure(ledger):
    tool, _ = ts.select_tool(_packet(), "what do cats eat", _Model('{"selected_tool": "web", "params": "{'))
    assert tool is None
    rec = list(ol.read(kinds=["tool_selection"]))[0]
    assert rec["parsed"] is False and rec["raw"].startswith('{"selected_tool"')


def test_current_packet_is_the_newest_turn(ledger):
    assert ol.current_packet("s") == ""
    ol.note_turn("s", "t1")
    ol.note_turn("s", "t2")
    assert ol.current_packet("s") == "t2" and ol.previous_packet("s", "t2") == "t1"


# ── approval calls are signed (401 on every approval-gated reflex call) ──

def test_approval_and_pending_requests_carry_auth_headers(monkeypatch):
    from gaia_core.utils import mcp_client
    seen = []

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"pending": []}

    monkeypatch.setattr(mcp_client, "_signed", lambda m, p: {"X-GAIA-Auth": f"{m} {p}"})
    monkeypatch.setattr(mcp_client.requests, "post", lambda url, **kw: seen.append(kw["headers"]) or R())
    monkeypatch.setattr(mcp_client.requests, "get", lambda url, **kw: seen.append(kw["headers"]) or R())
    monkeypatch.setenv("MCP_LITE_ENDPOINT", "http://gaia-mcp:8765/jsonrpc")
    mcp_client.approve_action_via_mcp("a1", "ABCDE")
    mcp_client.get_pending_action("a1")
    assert seen == [{"X-GAIA-Auth": "POST /approve_action"}, {"X-GAIA-Auth": "GET /pending_approvals"}]


def test_direct_reflex_helper_outcome_is_recorded(ledger):
    from gaia_core.cognition.agent_core import AgentCore
    ol.note_turn("s", "t1")
    fake = SimpleNamespace(_DIRECT_FAILURE_RE=AgentCore._DIRECT_FAILURE_RE)
    assert AgentCore._ledger_direct(fake, "s", "list_files", lambda: "a.md\nb.md") == "a.md\nb.md"
    AgentCore._ledger_direct(fake, "s", "list_files", lambda: "Approval failed: 401")
    ok, bad = list(ol.read(kinds=["tool_execution"]))
    assert ok["path"] == "direct" and ok["ok"] is True and ok["packet_id"] == "t1"
    assert bad["ok"] is False and bad["error"].startswith("Approval failed")


def _sel(tool, params, conf=0.9):
    return json.dumps({"selected_tool": tool, "params": json.dumps(params), "reasoning": "r", "confidence": conf})


def test_unknown_tool_never_executes_and_is_recorded(ledger):
    tool, _ = ts.select_tool(_packet(), "look this over", _Model(_sel("review", {})))
    assert tool is None
    rec = list(ol.read(kinds=["tool_selection"]))[0]
    assert rec["valid"] is False and "review" in rec["error"]


def test_unknown_action_for_a_real_domain_is_invalid(ledger):
    tool, _ = ts.select_tool(_packet(), "look this over", _Model(_sel("file", {"action": "teleport"})))
    assert tool is None


def test_llm_confidence_stays_under_the_review_skip(ledger):
    tool, _ = ts.select_tool(_packet(), "what do cats eat", _Model(_sel("web", {"action": "search", "query": "x"}, 0.99)))
    assert tool.selection_confidence < 0.9

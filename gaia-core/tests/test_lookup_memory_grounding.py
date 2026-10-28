"""0vuz: grounding consults what GAIA looked up before — answering from a
strong fresh memory without searching, revalidating on pushback — and the
prompt cites where each remembered fact came from. 6afh: "strong" means
confirmed for this question by an earlier answer that held up."""

from unittest.mock import patch

from gaia_core.cognition import knowledge_router as kr
from gaia_core.cognition.knowledge_router import GroundingResult


def _mem(score=0.9, stale=False, rid="m1", qmatch=0.8, confirmed=0.8, unhelpful=0.0):
    return GroundingResult(
        source="memory", trust_tier="remembered", query="q",
        content="The Lady of the Lake gives King Arthur the sword Excalibur.",
        url="https://en.wikipedia.org/wiki/Lady_of_the_Lake",
        title="Lady of the Lake - Wikipedia", retrieved_at="2026-10-27T12:00:00+00:00",
        stale=stale, score=score, question_match=qmatch, memory_id=rid,
        confirmed_match=confirmed, unhelpful_match=unhelpful)


def _web():
    return GroundingResult(source="web", trust_tier="verified_external",
                           content="Excalibur was given by the Lady of the Lake.", url="https://w")


Q = ["Where did he get the sword?"]
CTX = ["Can you tell me the name of King Arthur's sword?", "Excalibur"]


def test_followup_answers_from_strong_memory_without_searching():
    with patch.object(kr, "_query_lookup_memory", return_value=[_mem()]), \
            patch.object(kr, "_query_web") as qw:
        ctx = kr.ground_conversation(Q, context_turns=CTX)
    qw.assert_not_called()
    assert ctx.sources_queried == ["memory"] and ctx.results[0].source == "memory"


def test_followup_searches_when_memory_is_weak_or_stale():
    # weak relevance, stale, or relevant-but-a-different-question (the live
    # "where did he get it" vs remembered "what's it called" case)
    # …or relevant and asked before, but never confirmed to answer it (6afh)
    for mem in (_mem(score=0.4), _mem(stale=True), _mem(score=0.75, qmatch=0.29),
                _mem(confirmed=0.0)):
        with patch.object(kr, "_query_lookup_memory", return_value=[mem]), \
                patch.object(kr, "_query_web", return_value=[_web()]) as qw:
            ctx = kr.ground_conversation(Q, context_turns=CTX)
        qw.assert_called_once()
        assert [r.source for r in ctx.results] == ["web", "memory"]


def test_pushback_revalidates_memory_against_the_web():
    calls = []
    with patch.object(kr, "_query_lookup_memory", return_value=[_mem()]), \
            patch("gaia_core.utils.mcp_client.call_jsonrpc",
                  side_effect=lambda m, p: calls.append((m, p)) or {"ok": True}), \
            patch.object(kr, "_query_web", side_effect=lambda *a, **k: calls.append(("web",)) or [_web()]):
        ctx = kr.ground_conversation(["Wasn't it Merlin who gave it to him?"], reason="pushback")
    # Marked BEFORE the search, so a re-returned result is refreshed (cleared).
    assert calls[0] == ("palace_dispute_lookup", {"ids": ["m1"]})
    assert calls[1] == ("web",)
    assert [r.source for r in ctx.results] == ["web"]


def test_pushback_falls_back_to_stale_marked_memory_when_web_is_empty():
    with patch.object(kr, "_query_lookup_memory", return_value=[_mem()]), \
            patch("gaia_core.utils.mcp_client.call_jsonrpc", return_value={"ok": True}), \
            patch.object(kr, "_query_web", return_value=[]):
        ctx = kr.ground_conversation(["Wasn't it Merlin?"], reason="pushback")
    assert [r.source for r in ctx.results] == ["memory"] and ctx.results[0].stale


def test_ground_query_skips_web_on_strong_memory():
    with patch.object(kr, "_query_system_context", return_value=[]), \
            patch.object(kr, "_query_mempalace", return_value=[]), \
            patch.object(kr, "_query_lookup_memory", return_value=[_mem()]), \
            patch.object(kr, "_query_journal", return_value=[]), \
            patch.object(kr, "_query_knowledge_graph", return_value=[]), \
            patch.object(kr, "_query_web") as qw:
        ctx = kr.ground_query("Where did King Arthur get Excalibur?")
    qw.assert_not_called()
    assert "memory" in ctx.sources_queried


def test_aaak_only_palace_memories_are_not_grounding():
    aaak = {"results": [{"palace_id": "p1", "content":
            'technical|architecture|2026-10-20|system_clock|correction|util=0\n'
            '0:ACT+KIN+ART|actually_king_arthur|"Did you know that"|TECHNICAL'}]}
    with patch("gaia_core.utils.mcp_client.call_jsonrpc",
               return_value={"ok": True, "response": {"result": aaak}}):
        assert kr._query_mempalace("King Arthur") == []


def test_lookup_memory_recall_parses_mcp_hits():
    hit = {"id": "abc", "snippet": "The Lady of the Lake gives Arthur Excalibur.", "url": "u",
           "title": "", "domain": "en.wikipedia.org", "retrieved_at": "2026-10-27T12:00:00+00:00",
           "stale": False, "score": 0.75}
    with patch("gaia_core.utils.mcp_client.call_jsonrpc",
               return_value={"ok": True, "response": {"result": {"results": [hit]}}}) as rpc:
        res = kr._query_lookup_memory("King Arthur Excalibur")
    assert rpc.call_args[0][0] == "palace_recall_lookup"
    assert res[0].title == "en.wikipedia.org" and res[0].memory_id == "abc" and res[0].score == 0.75


def test_lookup_memory_recall_failure_is_empty_not_fatal():
    with patch("gaia_core.utils.mcp_client.call_jsonrpc", side_effect=RuntimeError("mcp down")):
        assert kr._query_lookup_memory("King Arthur") == []


def _packet(results, disputed=""):
    from gaia_common.protocols.cognition_packet import CognitionPacket, Content, DataField
    info = {"results": results, "source": "knowledge_router:memory", "query": "q"}
    if disputed:
        info["disputed_answer"] = disputed
    return CognitionPacket(content=Content(
        original_prompt="Where did he get the sword?",
        data_fields=[DataField(key="web_grounding", value={"q": info}, type="json")]))


def _text(packet):
    from gaia_core.utils import prompt_builder
    return " ".join(str(m.get("content", "")) for m in prompt_builder.build_from_packet(packet))


REMEMBERED = {"title": "Lady of the Lake - Wikipedia", "snippet": "The Lady of the Lake gives Arthur Excalibur.",
              "url": "https://en.wikipedia.org/wiki/Lady_of_the_Lake", "remembered": True,
              "retrieved_at": "2026-10-27T12:00:00+00:00", "stale": False}


def test_remembered_lookups_render_as_memory_with_citation():
    text = _text(_packet([REMEMBERED]))
    assert "From your memory of earlier lookups" in text
    assert "Lady of the Lake - Wikipedia, looked up 2026-10-27" in text
    assert "Candidate context" not in text


def test_stale_memory_is_flagged_and_mixed_results_keep_web_header():
    stale = dict(REMEMBERED, stale=True)
    web = {"title": "web", "snippet": "Excalibur was given by the Lady of the Lake.", "url": "https://w"}
    text = _text(_packet([web, stale]))
    assert "may be out of date" in text
    assert "From your memory of earlier lookups" not in text


class _Resp:
    def __init__(self, body):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        import json
        return json.dumps(self._body).encode()


def _probe_rpc(recall_hits):
    """Fake gaia-mcp for semantic_probe: records which methods were called."""
    import json
    calls = []

    def fake(req, timeout=None):
        method = json.loads(req.data)["method"]
        calls.append(method)
        if method == "palace_recall_lookup":
            return _Resp({"result": {"results": recall_hits}})
        return _Resp({"result": {"results": [{"title": "Excalibur", "snippet": "web says", "url": "https://w"}]}})
    return fake, calls


PROBE_HIT = {"id": "a", "title": "Excalibur - Wikipedia", "snippet": "Excalibur is the legendary sword of King Arthur.",
             "url": "https://en.wikipedia.org/wiki/Excalibur", "retrieved_at": "2026-10-27T12:00:00+00:00",
             "stale": False, "score": 1.0, "question_match": 1.0}


def test_semantic_fallback_uses_memory_instead_of_searching():
    from gaia_core.cognition import semantic_probe
    hit = PROBE_HIT  # looked up for this very entity: a cached search
    fake, calls = _probe_rpc([hit])
    with patch("urllib.request.urlopen", side_effect=fake):
        out = semantic_probe._web_search_fallback(["Excalibur"])
    assert calls == ["palace_recall_lookup"]
    assert out["Excalibur"]["source"] == "lookup_memory"
    assert out["Excalibur"]["results"][0]["remembered"] is True


def test_semantic_fallback_searches_when_nothing_is_remembered():
    from gaia_core.cognition import semantic_probe
    fake, calls = _probe_rpc([])
    with patch("urllib.request.urlopen", side_effect=fake):
        out = semantic_probe._web_search_fallback(["Excalibur"])
    assert calls == ["palace_recall_lookup", "web_search"]
    assert out["Excalibur"]["source"] == "web_search"


def test_semantic_fallback_searches_when_memory_was_for_another_question():
    # 76hb: pages kept for "Where did King Arthur get Excalibur?" recalled
    # for the entity "Excalibur" are another search's results — search.
    from gaia_core.cognition import semantic_probe
    fake, calls = _probe_rpc([dict(PROBE_HIT, question_match=0.2)])
    with patch("urllib.request.urlopen", side_effect=fake):
        out = semantic_probe._web_search_fallback(["Excalibur"])
    assert calls == ["palace_recall_lookup", "web_search"]
    assert out["Excalibur"]["source"] == "web_search"


def test_semantic_fallback_confirmed_memory_stands_in():
    from gaia_core.cognition import semantic_probe
    fake, calls = _probe_rpc([dict(PROBE_HIT, question_match=0.2, times_confirmed=2)])
    with patch("urllib.request.urlopen", side_effect=fake):
        out = semantic_probe._web_search_fallback(["Excalibur"])
    assert calls == ["palace_recall_lookup"] and out["Excalibur"]["source"] == "lookup_memory"


def test_semantic_fallback_never_stands_in_with_memory_that_failed():
    from gaia_core.cognition import semantic_probe
    fake, calls = _probe_rpc([dict(PROBE_HIT, times_confirmed=2, unhelpful_match=0.8)])
    with patch("urllib.request.urlopen", side_effect=fake):
        semantic_probe._web_search_fallback(["Excalibur"])
    assert calls == ["palace_recall_lookup", "web_search"]


def test_semantic_fallback_uses_other_memory_when_web_is_empty():
    import json
    from gaia_core.cognition import semantic_probe

    def fake(req, timeout=None):
        if json.loads(req.data)["method"] == "palace_recall_lookup":
            return _Resp({"result": {"results": [dict(PROBE_HIT, question_match=0.2)]}})
        return _Resp({"result": {"results": []}})
    with patch("urllib.request.urlopen", side_effect=fake):
        out = semantic_probe._web_search_fallback(["Excalibur"])
    assert out["Excalibur"]["source"] == "lookup_memory"
    assert out["Excalibur"]["results"][0]["remembered"] is True


def test_recall_answer_needs_a_strong_question_match():
    with patch.object(kr, "_query_lookup_memory", return_value=[_mem()]), \
            patch.object(kr, "_query_web") as qw:
        ctx = kr.recall_answer("Where did King Arthur get Excalibur?")
    qw.assert_not_called()
    assert ctx is not None and ctx.sources_queried == ["memory"]
    assert ctx.query == "Where did King Arthur get Excalibur?"
    # Same subject, different question — or chitchat — gives nothing (None,
    # so it isn't counted as a knowledge gap either).
    for mem in ([_mem(qmatch=0.2)], []):
        with patch.object(kr, "_query_lookup_memory", return_value=mem):
            assert kr.recall_answer("How are you today?") is None


def test_recall_answer_searches_again_when_memory_never_answered():
    # 6afh live case: the Excalibur pages stored for "where did he get it"
    # never say where; unconfirmed (or stale), they must not stand in for a
    # search — but the question is a known knowledge question, so search.
    for mem in (_mem(confirmed=0.0), _mem(stale=True)):
        with patch.object(kr, "_query_lookup_memory", return_value=[mem]), \
                patch.object(kr, "_query_web", return_value=[_web()]) as qw:
            ctx = kr.recall_answer("Where did King Arthur get Excalibur?")
        qw.assert_called_once()
        assert ctx.sources_queried == ["web", "memory"]
        assert [r.source for r in ctx.results] == ["web", "memory"]


def test_memory_that_failed_this_question_is_dropped():
    hits = [{"id": "bad", "snippet": "Excalibur is the mythical sword of King Arthur.", "url": "u",
             "score": 1.0, "question_match": 0.8, "unhelpful_match": 0.8},
            {"id": "ok", "snippet": "The Lady of the Lake gives Arthur Excalibur.", "url": "v",
             "score": 0.9, "question_match": 0.8, "confirmed_match": 0.7}]
    with patch("gaia_core.utils.mcp_client.call_jsonrpc",
               return_value={"ok": True, "response": {"result": {"results": hits}}}):
        res = kr._query_lookup_memory("Where did King Arthur get Excalibur?")
    assert [r.memory_id for r in res] == ["ok"] and res[0].confirmed_match == 0.7
    assert kr._strong_memory(res)


def test_web_results_carry_their_memory_id():
    web = {"results": [{"title": "Lady of the Lake", "snippet": "The Lady of the Lake gives Arthur Excalibur.",
                        "url": "https://w", "memory_id": "abc123"}]}
    with patch("gaia_core.utils.mcp_client.call_jsonrpc",
               return_value={"ok": True, "response": {"result": web}}):
        res = kr._query_web("Where did King Arthur get Excalibur?")
    assert res[0].memory_id == "abc123"


def _feedback_packet(source="knowledge_router:web"):
    from gaia_common.protocols.cognition_packet import CognitionPacket, Content, DataField
    info = {"results": [{"title": "web", "snippet": "The Lady of the Lake gives Arthur Excalibur.",
                         "url": "https://w", "memory_id": "abc123"}],
            "source": source, "query": "Where did he get it?",
            "memory_query": "Where did he get the sword? King Arthur Excalibur"}
    return CognitionPacket(content=Content(
        original_prompt="Where did he get it?",
        data_fields=[DataField(key="web_grounding", value={"q": info}, type="json")]))


def _run_feedback(packet, answer="<think>hmm</think>From the Lady of the Lake."):
    import threading
    from gaia_core.cognition.agent_core import AgentCore
    sent = []
    done = threading.Event()

    def fake(method, params):
        sent.append((method, params))
        done.set()
        return {"response": {"result": {"verdict": "supported", "records": 1}}}
    with patch("gaia_core.utils.mcp_client.call_jsonrpc", side_effect=fake):
        AgentCore._lookup_feedback(object.__new__(AgentCore), answer, packet)
        done.wait(2)
    return sent


def test_answer_feedback_goes_to_lookup_memory_under_the_recall_question():
    sent = _run_feedback(_feedback_packet())
    assert sent[0][0] == "palace_lookup_feedback"
    params = sent[0][1]
    assert params["question"] == "Where did he get the sword? King Arthur Excalibur"
    assert params["answer"] == "From the Lady of the Lake."
    assert params["sources"][0]["memory_id"] == "abc123"


def test_no_feedback_for_semantic_probe_grounding():
    assert _run_feedback(_feedback_packet(source="web_search")) == []


def test_recall_answer_leaves_questions_about_the_user_to_the_conversation():
    # ecnc smoke 16: "What is my favorite color?" was web-searched once (a
    # hedge), then re-searched as "looked up before", burying "cerulean".
    with patch.object(kr, "_query_lookup_memory", return_value=[_mem(confirmed=0.0)]) as qm, \
            patch.object(kr, "_query_web") as qw:
        assert kr.recall_answer("What is my favorite color?") is None
        assert kr.recall_answer("Where did I park, do we know?") is None
    qm.assert_not_called()
    qw.assert_not_called()
    assert not kr._about_the_user("Where did King Arthur get Excalibur?")
    assert not kr._about_the_user("Who is Imelda Marcos?")  # 'i' only as a word
    assert not kr._about_the_user("Can you tell me the name of King Arthur's sword?")
    assert not kr._about_the_user("Show us where Excalibur came from.")


def test_epistemic_grounding_does_not_web_search_questions_about_the_user():
    # ecnc smoke 16: a hedge on "What is my favorite color?" searched the web
    # and regenerated "Purple." over "cerulean" from the turn before.
    from gaia_core.cognition.agent_core import AgentCore
    core = object.__new__(AgentCore)
    import logging
    core.logger = logging.getLogger("test")
    with patch("gaia_core.utils.mcp_client.call_jsonrpc") as rpc:
        out = AgentCore._ground_and_regenerate(core, "core", "What is my favorite color?",
                                               "Your favorite color is cerulean, I think.", "hedging_detected")
    rpc.assert_not_called()
    assert out == "Your favorite color is cerulean, I think."

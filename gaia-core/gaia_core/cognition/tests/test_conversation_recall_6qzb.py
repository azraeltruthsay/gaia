"""Conversation recall (6qzb): the palace's read-only record of what was said.

"What did we talk about regarding the memory palace?" was skipped as chat
(the "we" read as a question about the user), no record of past exchanges
was searched, and Core promised "I'll look through our memory banks" and
stopped.
"""

import json
from types import SimpleNamespace

import pytest

from gaia_common.utils import conversation_memory as cm
from gaia_core.cognition import knowledge_router as kr
from gaia_core.cognition.nlu.intent_detection import conversation_topic, is_conversation_recall


@pytest.fixture
def palace(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMPAL_PALACE_PATH", str(tmp_path))
    return cm.ConversationMemory()


# ── detection ──

@pytest.mark.parametrize("text", [
    "What did we talk about regarding the memory palace?",
    "Did I ever tell you about my sister?",
    "Have we discussed the gearbox before?",
    "Remind me what we talked about yesterday",
    "What was our last conversation about?",
    "do you remember what I said about Rust",
    "What did we decide to name the cat?",
])
def test_conversation_recall_questions(text):
    assert is_conversation_recall(text)


@pytest.mark.parametrize("text", [
    "I told you I'm tired",                 # a statement, not a recall
    "What is the memory palace?",
    "Tell me about Excalibur",
    "We talked to the vendor and they said no.",
    "How are you?",
])
def test_not_conversation_recall(text):
    assert not is_conversation_recall(text)


@pytest.mark.parametrize("text,topic", [
    ("What did we talk about regarding the memory palace?", "memory palace"),
    ("Did I ever tell you about my sister?", "sister"),
    ("What did we talk about last time?", ""),
    ("What did we decide to name the cat?", "cat"),
])
def test_topic_drops_the_recall_phrasing(text, topic):
    assert conversation_topic(text) == topic


# ── the record ──

def test_record_and_recall_by_topic(palace):
    palace.record_exchange("discord_dm_111111111111111111", "Let's design the memory palace wings", "Sure: identity, technical…")
    palace.record_exchange("discord_dm_111111111111111111", "What's the weather?", "Rainy.")
    palace.record_exchange("discord_dm_222222222222222222", "memory palace for bob", "ok")  # another person
    hits = palace.recall("discord_dm_111111111111111111", "memory palace")
    assert [h["user"] for h in hits] == ["Let's design the memory palace wings"]
    assert hits[0]["score"] == 1.0


def test_synthetic_sessions_are_never_recorded(palace):
    for sid in ("smoke-test-abc", "gaia_heartbeat_session", "gaia_initiative_loop_session", "",
                "cogtest-0159d0de", "wsl_push3", "discord_dm_999999000001", "web_ssgd_test1"):
        assert palace.record_exchange(sid, "hi there", "hello") is False
    assert not any(palace.root.glob("*.jsonl")) if palace.root.exists() else True


def test_think_tags_and_length_are_cleaned(palace):
    palace.record_exchange("web_0a1b2c3d", "q", "<think>private</think>answer " + "x" * 5000)
    rec = palace.latest("web_0a1b2c3d", 1)[0]
    assert "private" not in rec["reply"] and len(rec["reply"]) <= cm.MAX_SIDE_CHARS + 1


def test_backfill_is_idempotent(palace, tmp_path):
    arch = tmp_path / "history" / "conversations" / "default"
    arch.mkdir(parents=True)
    msgs = [{"role": "user", "content": "palace wings?", "timestamp": "2026-10-01T10:00:00"},
            {"role": "tool", "content": "…"},
            {"role": "assistant", "content": "Six wings.", "timestamp": "2026-10-01T10:00:05"}]
    (arch / "web_0000000a.json").write_text(json.dumps({"session_id": "web_0000000a", "messages": msgs}))
    (arch / "smoke-test-1.json").write_text(json.dumps({"session_id": "smoke-test-1", "messages": msgs}))
    assert palace.backfill_from_archives(str(tmp_path / "history")) == {"web_0000000a": 1}
    assert palace.backfill_from_archives(str(tmp_path / "history")) == {}
    assert palace.count("web_0000000a") == 1


def test_no_write_method_is_exposed_to_tools():
    from gaia_common.utils.domain_tools import ACTION_TO_LEGACY
    # cfr_review_conversation reads the working set; nothing writes the record
    writers = [v for v in ACTION_TO_LEGACY.values() if "conversation" in v
               and any(w in v for w in ("store", "record", "write", "add", "edit", "delete"))]
    assert writers == []


# ── router ──

def test_recall_leaves_out_what_the_prompt_already_shows(palace):
    palace.record_exchange("web_0a1b2c3d", "memory palace layout?", "six wings")
    palace.record_exchange("web_0a1b2c3d", "and the memory palace archive?", "a sweep")
    history = [{"role": "user", "content": "and the memory palace archive?"}]
    ctx = kr.recall_conversation("web_0a1b2c3d", "What did we talk about regarding the memory palace?", history)
    assert len(ctx.results) == 1 and "six wings" in ctx.results[0].content
    assert ctx.results[0].content.startswith("On ") and "they said:" in ctx.results[0].content
    assert ctx.results[0].source == "conversation" and ctx.sources_queried == ["conversation"]


def test_recall_with_nothing_on_record_is_searched_and_empty(palace):
    ctx = kr.recall_conversation("web_0000beef", "What did we talk about regarding the memory palace?")
    assert not ctx.has_grounding and ctx.sources_queried == ["conversation"]
    assert ctx.query == "memory palace"


# ── rendering ──

def _prompt(value):
    from gaia_common.protocols.cognition_packet import CognitionPacket, Content, DataField, Intent
    from gaia_core.utils.prompt_builder import build_from_packet
    q = "What did we talk about regarding the memory palace?"
    pkt = CognitionPacket(content=Content(original_prompt=q), intent=Intent(user_intent="chat"))
    pkt.content.data_fields.append(DataField(key="web_grounding", type="json",
                                             source="knowledge_router", value={q[:50]: value}))
    return build_from_packet(pkt)


def test_recalled_exchanges_are_answered_result_first():
    msgs = _prompt({"conversation": True, "source": "knowledge_router:conversation", "results": [
        {"title": "earlier conversation", "snippet": 'On 2026-10-20, they said: "palace wings?" — and you replied: "Six."',
         "url": "", "conversation": True, "retrieved_at": "2026-10-20T10:00:00"}]})
    assert [m["role"] for m in msgs] == ["system", "user"]
    user = msgs[1]["content"]
    assert user.startswith("Searching your record of earlier conversations with them returned")
    assert '- On 2026-10-20, they said: "palace wings?"' in user
    assert "regarding the memory palace" in user


def test_nothing_on_record_is_stated():
    msgs = _prompt({"conversation": True, "topic": "memory palace", "results": [], "on_record": 4,
                    "source": "knowledge_router:conversation"})
    assert "Nothing about memory palace." in msgs[1]["content"]


def test_a_first_conversation_is_named_as_one():
    msgs = _prompt({"conversation": True, "topic": "", "results": [], "on_record": 0,
                    "source": "knowledge_router:conversation"})
    assert "this is your first conversation together" in msgs[1]["content"]


def test_ordinary_grounding_keeps_the_full_prompt():
    msgs = _prompt({"source": "knowledge_router:web", "results": [
        {"title": "w", "snippet": "a page about palaces", "url": "http://x"}]})
    assert len(msgs) > 2


def test_session_manager_records_each_exchange(palace, tmp_path, monkeypatch):
    from gaia_core.memory import session_manager as smod
    sm = smod.SessionManager.__new__(smod.SessionManager)
    sm.sessions = {}
    sm.max_active_messages = 1000
    monkeypatch.setattr(sm, "_save_state", lambda: None, raising=False)
    monkeypatch.setattr(sm, "_update_markdown_log", lambda *a: None, raising=False)
    monkeypatch.setattr(sm, "get_or_create_session",
                        lambda sid, persona="default": sm.sessions.setdefault(
                            sid, SimpleNamespace(history=[])), raising=False)
    sm.add_message("discord_dm_999999999999999999", "user", "Did the palace sweep run?")
    sm.add_message("discord_dm_999999999999999999", "assistant", "Yes, it archived 12 records.")
    hits = palace.recall("discord_dm_999999999999999999", "palace sweep")
    assert hits and hits[0]["reply"] == "Yes, it archived 12 records."

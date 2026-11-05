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
    assert user.startswith("Searching your record of earlier conversations returned")
    assert "said in another conversation" not in user
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


# ── other conversations (02m5): shared memory, with judgment ──

ANDY = "web_3132edcd"
DM = "discord_dm_596925786208993283"


@pytest.mark.parametrize("text", [
    "I'm well, thanks for asking. Do you remember Andy from yesterday?",
    "do you remember andy from yesterday",
    "No this was Andy my friend. You spoke with him briefly yesterday in a web dashboard chat, not here.",
    "Nothing exciting. He just spoke with you for a bit and asked you to recite a poem.",
    "Have you met my friend?",
])
def test_a_conversation_with_someone_is_recall(text):
    assert is_conversation_recall(text)


@pytest.mark.parametrize("text", [
    "Do you remember how to tie a knot?",
    "My boss talked to the vendor.",
    "You should talk to him about it.",
])
def test_not_a_conversation_with_someone(text):
    assert not is_conversation_recall(text)


@pytest.mark.parametrize("text", [
    "No this was Andy my friend. You spoke with him on the web chat.",
    "I'm well. Do you remember Andy from yesterday?",
    "do you remember andy from yesterday",
])
def test_the_person_is_found_in_another_conversation(palace, text):
    palace.record_exchange(ANDY, "Hello GAIA. I am Andy. I am Azrael's friend.", "Hello, Andy.")
    palace.record_exchange(ANDY, "What is a raven?", "A bird.")
    ctx = kr.recall_conversation(DM, text)
    assert len(ctx.results) == 1
    assert ctx.results[0].content.startswith("In a separate web chat on ")
    assert "the person there said" in ctx.results[0].content and "I am Andy" in ctx.results[0].content


def test_elsewhere_leaves_room_for_this_conversation(palace):
    for i in range(4):
        palace.record_exchange(ANDY, f"palace wings {i}", "six")
        palace.record_exchange(DM, f"palace rooms {i}", "many")
    ctx = kr.recall_conversation(DM, "What did we talk about regarding the palace?")
    kinds = [r.title for r in ctx.results]
    assert kinds.count("another conversation") == 2 and kinds.count("earlier conversation") == 1


def test_no_topic_stays_in_this_conversation(palace):
    palace.record_exchange(ANDY, "secret plan", "ok")
    palace.record_exchange(DM, "hello there", "hi")
    ctx = kr.recall_conversation(DM, "What did we talk about last time?")
    assert all("secret" not in r.content for r in ctx.results) and ctx.results


def test_said_elsewhere_comes_with_judgment():
    msgs = _prompt({"conversation": True, "source": "knowledge_router:conversation", "results": [
        {"title": "another conversation", "url": "", "conversation": True, "retrieved_at": "",
         "snippet": 'In a separate web chat on 2026-11-05, the person there said: "I am Andy"'}]})
    assert "Use your judgment about what belongs here" in msgs[1]["content"]


def test_palace_tool_finds_conversations_only_when_asked(tmp_path):
    from gaia_common.utils.mempalace import MemPalace
    cm.ConversationMemory(tmp_path / "conversations").record_exchange(ANDY, "I am Andy.", "Hello, Andy.")
    palace = MemPalace({"root_dir": str(tmp_path), "kg_path": str(tmp_path / "kg.sqlite3"), "wings": {}})
    assert palace.recall("andy")["conversations"] == []
    found = palace.recall("andy", include_conversations=True)
    assert found["conversations"][0]["where"] == "web chat"
    assert "I am Andy" in found["conversations"][0]["they_said"] and found["conversations_note"]


def test_a_pronoun_keeps_the_person_from_the_turns_before(palace):
    palace.record_exchange(ANDY, "I am Andy.", "Hello, Andy.")
    history = [{"role": "user", "content": "Do you remember Andy from yesterday?"},
               {"role": "assistant", "content": "Yes."}]
    ctx = kr.recall_conversation(DM, "That's not accurate. He just spoke with you for a bit.", history)
    assert ctx.query == "Andy" and "I am Andy" in ctx.results[0].content


def test_what_did_he_say_follows_a_conversation_question_only():
    from gaia_core.cognition.nlu.intent_detection import is_conversation_followup as f
    asked = [{"role": "user", "content": "Do you remember Andy from yesterday?"}]
    story = [{"role": "user", "content": "Tell me about Lincoln"}]
    assert f("What did he say about himself?", asked)
    assert not f("What did he say about himself?", story)
    assert not f("How old is he?", asked)


def test_a_reply_they_corrected_is_not_recalled_as_said(palace):
    palace.record_exchange(DM, "Do you remember Andy?", "He's a systems person on the storage project.")
    palace.record_exchange(DM, "Interesting, that's not accurate. Andy is my friend.", "Sorry.")
    found = palace.recall(DM, "Andy") + palace.recall_elsewhere("web_00000000", "Andy")
    assert found and all("storage" not in h["reply"] for h in found)
    assert any("my friend" in h["user"] for h in found)


def test_a_short_conversation_is_not_crowded_out_by_a_long_one(palace):
    palace.record_exchange(ANDY, "Hello. I am Andy.", "Hello, Andy.")
    palace.record_exchange(ANDY, "Goodbye from Andy.", "Goodbye.")
    for i in range(5):
        palace.record_exchange(DM, f"Andy again {i}", "ok")
    found = palace.recall_elsewhere("web_00000000", "Andy", top_k=2)
    assert {h["session_id"] for h in found} == {ANDY, DM}
    assert next(h for h in found if h["session_id"] == ANDY)["user"] == "Hello. I am Andy."


def test_the_rest_of_what_they_said_comes_along(palace):
    from gaia_common.utils.people import PeopleRegistry
    PeopleRegistry().create("Andy", [{"kind": "session", "id": ANDY}])
    palace.record_exchange(ANDY, "I am Andy.", "Hello.")
    palace.record_exchange(ANDY, "Recite the poem Jabberwockey?", "Twas brillig")
    ctx = kr.recall_conversation(DM, "Do you remember Andy from yesterday?")
    assert 'they said: "Recite the poem Jabberwockey?"' in ctx.results[0].content
    assert "Twas brillig" not in ctx.results[0].content


# ── around the deploy ──

@pytest.mark.parametrize("params,message,url", [
    ({}, "Fetch the page at https://httpbin.org/html and tell me.", "https://httpbin.org/html"),
    ({"url": ""}, "Read https://a.org/x.", "https://a.org/x"),
    ({"url": "https://b.org"}, "Read https://a.org/x", "https://b.org"),
    ({}, "Compare https://a.org and https://b.org", None),
    ({}, "Fetch that page", None),
])
def test_a_fetch_with_no_url_uses_the_one_they_named(params, message, url):
    from gaia_common.utils.tool_call_parser import fill_missing_url
    assert fill_missing_url("web", "fetch", params, message).get("url") == (url or params.get("url"))


def test_only_web_fetch_is_filled():
    from gaia_common.utils.tool_call_parser import fill_missing_url
    assert fill_missing_url("web", "search", {}, "see https://a.org") == {}
    assert fill_missing_url("file", "fetch", {}, "see https://a.org") == {}


def test_someone_elses_conversation_brings_only_the_match(palace):
    palace.record_exchange(DM, "Is Andy around?", "Not today.")
    palace.record_exchange(DM, "Who forged Excalibur?", "The Lady.")
    ctx = kr.recall_conversation("web_0000beef", "Do you remember Andy from yesterday?")
    assert ctx.results and "Excalibur" not in ctx.results[0].content

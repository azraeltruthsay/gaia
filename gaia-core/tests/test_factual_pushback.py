"""8xr4.19: "Didn't he get it from the lady in the lake?" was filed under
"clarification", and Core's deliberation adapter read the user's claim as a
question about its own prior turn ("Which turn are we talking about?").
Pushback on the last answer now deliberates in deep mode (Prime), and the
knowledge router no longer feeds GAIA's own per-turn deliberation transcripts
back to her as grounding."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gaia_core.cognition import knowledge_router
from gaia_core.cognition.nlu.intent_detection import is_factual_pushback

HIST = [
    {"role": "user", "content": "Where did he get the sword?"},
    {"role": "assistant", "content": "From the wizard accompanying him on his mission."},
]


@pytest.mark.parametrize("text", [
    "Didn't he get it from the lady in the lake?",
    "Wasn't it Merlin who gave it to him?",
    "But isn't Paris the capital?",
    "Doesn't the Lady of the Lake give it to him?",
    "I thought it was the Lady of the Lake.",
    "Are you sure about that?",
    "I'm pretty sure it was Merlin.",
    "No, it was the Lady of the Lake.",
    "Actually, he got it from a lake.",
    "That's wrong, it came from the lake.",
    "It was the Lady of the Lake, wasn't it?",
])
def test_pushback_detected(text):
    assert is_factual_pushback(text, HIST)


@pytest.mark.parametrize("text", [
    "Where did he get the sword?",
    "Didn't you say it was Merlin?",
    "Don't I need to restart it first?",
    "Thanks, that's great!",
    "Tell me more about Merlin.",
    "Didn't he get it from the lady in the lake.",  # no question mark, no phrase
])
def test_not_pushback(text):
    assert not is_factual_pushback(text, HIST)


def test_needs_a_previous_answer():
    q = "Didn't he get it from the lady in the lake?"
    assert not is_factual_pushback(q, [])
    assert not is_factual_pushback(q, [{"role": "user", "content": "hi"}])
    assert not is_factual_pushback("", HIST)


def _entry(eid, context, body="Where did he get the sword? From the wizard."):
    return SimpleNamespace(id=eid, context=context, body=body, tags=[], significance=3)


def test_journal_grounding_skips_cognition_transcripts():
    hits = [
        (_entry("cognition-1", "cognition"), 5.0),
        (_entry("cognition-2", "cognition"), 4.5),
        (_entry("self-1", "self", "Consolidation: talked about Arthurian legend."), 3.0),
        (_entry("heimr-1", "heimr", "In-world: the sword in the lake."), 2.0),
    ]
    with patch("gaia_core.memory.journal.search_entries", return_value=hits):
        results = knowledge_router._query_journal("where did he get the sword", k=3)
    ids = [r.content.split()[2] for r in results]
    assert ids == ["self-1", "heimr-1"]


def test_journal_grounding_empty_when_only_cognition():
    hits = [(_entry(f"cognition-{i}", "cognition"), 5.0) for i in range(6)]
    with patch("gaia_core.memory.journal.search_entries", return_value=hits):
        assert knowledge_router._query_journal("sword", k=3) == []


def test_conversation_query_carries_the_subject():
    q = knowledge_router.conversation_query([
        "Can you tell me the name of King Arthur's sword?",
        "Where did he get the sword?",
        "Didn't he get it from the lady in the lake?",
    ])
    assert q == "name King Arthur's sword lady lake"


def test_conversation_query_caps_length():
    q = knowledge_router.conversation_query(["alpha " * 5 + " ".join(f"w{i}" for i in range(80))])
    assert len(q) <= 120


def test_ground_pushback_uses_web_only():
    web = [knowledge_router.GroundingResult(
        source="web", trust_tier="verified_external",
        content="Lady of the Lake: gives Arthur the sword Excalibur.", query="q")]
    with patch.object(knowledge_router, "_query_web", return_value=web) as qw, \
            patch.object(knowledge_router, "_query_journal") as qj, \
            patch.object(knowledge_router, "_query_mempalace") as qm:
        ctx = knowledge_router.ground_pushback(
            ["Can you tell me the name of King Arthur's sword?",
             "Didn't he get it from the lady in the lake?"])
    qw.assert_called_once_with("name King Arthur's sword lady lake", max_results=3, snippet_chars=500)
    qj.assert_not_called()
    qm.assert_not_called()
    assert ctx.has_grounding and ctx.sources_queried == ["web"]


def _grounded_packet(disputed):
    from gaia_common.protocols.cognition_packet import CognitionPacket, Content, DataField
    info = {"results": [{"title": "web", "snippet": "Lady of the Lake gives Arthur Excalibur.",
                         "url": "https://en.wikipedia.org/wiki/Lady_of_the_Lake"}],
            "source": "knowledge_router:web", "query": "q"}
    if disputed:
        info["disputed_answer"] = disputed
    return CognitionPacket(content=Content(
        original_prompt="Didn't he get it from the lady in the lake?",
        data_fields=[DataField(key="web_grounding", value={"q": info}, type="json")]))


def _joined(messages):
    return " ".join(str(m.get("content", "")) for m in messages)


def test_disputed_grounding_is_framed_neutrally():
    # 8xr4.25: naming the dispute made Prime concede to false corrections.
    from gaia_core.utils import prompt_builder
    text = _joined(prompt_builder.build_from_packet(_grounded_packet("From the wizard Oberon.")))
    assert "Search results for the point the user raised" in text
    assert "Lady of the Lake gives Arthur Excalibur." in text
    assert "disputing" not in text
    assert "Candidate context" not in text


def test_ordinary_grounding_keeps_candidate_hedge():
    from gaia_core.utils import prompt_builder
    text = _joined(prompt_builder.build_from_packet(_grounded_packet("")))
    assert "Candidate context" in text
    assert "disputing" not in text


@pytest.mark.parametrize("raw,clean", [
    ("[Core]\n\nFrom the wizard.", "From the wizard."),
    ("[Prime]\n[Core]\nYes — the Lady of the Lake.", "Yes — the Lady of the Lake."),
    ("Plain reply [Core] mid-text", "Plain reply [Core] mid-text"),
    ("[Not a mind] stays", "[Not a mind] stays"),
])
def test_mind_tag_stripped_from_history(raw, clean):
    from gaia_core.memory.session_manager import _strip_mind_tag
    assert _strip_mind_tag(raw) == clean


DETOUR = [
    {"role": "user", "content": "Who threw it back into the lake when he died?"},
    {"role": "assistant", "content": "The sword was thrown back into the lake by Arthur's servant Bedivere."},
    {"role": "user", "content": "How are you feeling today?"},
    {"role": "assistant", "content": "Let me see. What's the current system health?"},
]


def test_disputed_exchange_skips_an_unrelated_detour():
    q, a = knowledge_router.disputed_exchange(DETOUR, "Wasn't it Lancelot who threw it back?")
    assert q == "Who threw it back into the lake when he died?"
    assert "Bedivere" in a


def test_disputed_exchange_defaults_to_the_last_answer():
    q, a = knowledge_router.disputed_exchange(DETOUR, "Are you sure?")
    assert q == "How are you feeling today?"
    assert knowledge_router.disputed_exchange([], "Are you sure?") == ("", "")

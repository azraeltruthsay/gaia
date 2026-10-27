"""8xr4.20: "Where did he get the sword?" after Excalibur was filed as chat, so
nothing grounded it, and Stage 0 web-searched {'ENTITY': 'he', 'PLACE':
'sword'} verbatim (Onimusha, katana prints, Roblox codes). Follow-ups now take
their search query from the whole exchange, and Stage 0 drops subject-less
entities and spends at most one web search per turn."""

import json
from unittest.mock import MagicMock, patch

import pytest

from gaia_core.cognition import knowledge_router
from gaia_core.cognition.agent_core import _clean_grounding_entities
from gaia_core.cognition.nlu import intent_detection
from gaia_core.cognition.nlu.intent_detection import is_knowledge_followup

HIST = [
    {"role": "user", "content": "Can you tell me the name of King Arthur's sword?"},
    {"role": "assistant", "content": "The name of King Arthur's sword is Excalibur."},
]


@pytest.mark.parametrize("text", [
    "Where did he get the sword?",
    "And who gave it to him?",
    "What happened to it after he died?",
    "So when did they find it?",
    "Which knight returned it to the lake?",  # "it"
])
def test_followup_detected(text):
    assert is_knowledge_followup(text, HIST)


@pytest.mark.parametrize("text", [
    "How are you feeling?",
    "What do you think about it?",
    "What time is it?",
    "Where did I leave off?",
    "Tell me more about Merlin.",
    "Thanks!",
    "Where is Camelot?",  # names its own subject
])
def test_not_followup(text):
    assert not is_knowledge_followup(text, HIST)


def test_followup_needs_a_previous_exchange():
    q = "Where did he get the sword?"
    assert not is_knowledge_followup(q, [])
    assert not is_knowledge_followup(q, HIST[:1])
    assert not is_knowledge_followup(q, None)


def test_conversation_grounding_queries_the_exchange():
    with patch.object(knowledge_router, "_query_web", return_value=[]) as qw:
        knowledge_router.ground_conversation(
            [HIST[0]["content"], "Where did he get the sword?"])
    qw.assert_called_once_with("name King Arthur's sword", max_results=3, snippet_chars=500)


@pytest.mark.parametrize("raw,clean", [
    ([{"ENTITY": "he", "PLACE": "sword"}, {"EVENT": "get the sword"}], ["sword", "get the sword"]),
    (["he", "it", "King Arthur", "king arthur", " Excalibur. "], ["King Arthur", "Excalibur"]),
    (["", None, "x"], []),
])
def test_clean_grounding_entities(raw, clean):
    assert _clean_grounding_entities(raw) == clean


def _nano_reply(content):
    resp = MagicMock()
    resp.read.return_value = json.dumps(
        {"choices": [{"message": {"content": content}}]}).encode()
    resp.__enter__.return_value = resp
    return resp


def test_extractor_flattens_category_dicts():
    reply = _nano_reply('[{"ENTITY": "he", "PLACE": "sword"}]')
    with patch.object(intent_detection, "urlopen", return_value=reply):
        assert intent_detection.extract_entities_neural("Where did he get the sword?") == ["he", "sword"]


def test_stage0_spends_one_web_search_per_turn():
    from gaia_core.cognition.agent_core import AgentCore
    agent = AgentCore.__new__(AgentCore)
    agent.config = MagicMock()
    agent.config.constants = {"GROUNDING": {"web_fallback": True}}
    agent.logger = MagicMock()
    calls = []

    def rpc(method, params):
        calls.append(method)
        if method == "web_search":
            return {"ok": True, "response": {"result": {"results": [{"title": "t"}]}}}
        return {"ok": True, "response": {"result": {"results": []}}}

    with patch("gaia_core.utils.mcp_client.call_jsonrpc", side_effect=rpc):
        out = agent._run_grounding_probes(["he", "Excalibur", "Lady of the Lake", "Avalon"], "s")
    assert calls.count("web_search") == 1
    assert "he" not in out


def test_followup_query_keeps_the_question():
    q = knowledge_router.followup_query(
        "Where did he get the sword?",
        [HIST[0]["content"], HIST[1]["content"]])
    assert q.startswith("Where did he get the sword?")
    assert "Arthur's" in q and "Excalibur" in q
    assert q.lower().count("sword") == 1  # already asked, not re-added


def test_followup_grounding_uses_context_turns():
    with patch.object(knowledge_router, "_query_web", return_value=[]) as qw:
        knowledge_router.ground_conversation(
            ["Where did he get the sword?"], context_turns=[HIST[1]["content"]])
    (query,), _ = qw.call_args
    assert query.startswith("Where did he get the sword?") and "Excalibur" in query


def test_followup_grounding_header_and_snippet_budget():
    from gaia_common.protocols.cognition_packet import CognitionPacket, Content, DataField
    from gaia_core.utils import prompt_builder
    snippet = "x" * 260 + " given to Arthur by the Lady of the Lake."
    pkt = CognitionPacket(content=Content(
        original_prompt="Where did he get the sword?",
        data_fields=[DataField(key="web_grounding", type="json", value={"q": {
            "results": [{"title": "web", "snippet": snippet, "url": "u"}],
            "source": "knowledge_router:web", "query": "q",
            "disputed_answer": "", "followup": True}})]))
    text = " ".join(str(m.get("content", "")) for m in prompt_builder.build_from_packet(pkt))
    assert "read with the conversation so far" in text
    assert "Lady of the Lake" in text
    assert "Candidate context" not in text


@pytest.mark.parametrize("answer,names", [
    ("According to the legend, King Arthur acquired Excalibur from the Lady of the Lake.",
     "King Arthur Excalibur Lady Lake"),
    ("Excalibur.", "Excalibur"),
    ("It was Bedivere. He threw it in.", "Bedivere"),
])
def test_proper_nouns_keep_names_drop_prose(answer, names):
    assert knowledge_router.proper_nouns(answer) == names

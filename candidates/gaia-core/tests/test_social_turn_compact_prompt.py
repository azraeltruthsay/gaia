"""Core's 4096 window puts every live turn on the compact prompt, which dropped
the casual anchor: greetings and "how are you" got a rules-and-tools block and
answered "I'm holding an open tool (search)". Pins that social turns keep the
anchor and the affect line, and that factual chat turns keep their tools.
"""

import pytest

from gaia_common.protocols.cognition_packet import CognitionPacket, Content, Intent
from gaia_core.utils import prompt_builder
from gaia_core.utils.prompt_builder import is_check_in, is_social_turn


@pytest.mark.parametrize("text", [
    "How are you today?", "how are you, really?", "Hey, how's it going?",
    "How are you feeling?", "What have you been up to?", "hi gaia, how have you been?",
    "How's your day?", "what's new",
])
def test_check_ins(text):
    assert is_check_in(text)


@pytest.mark.parametrize("text", [
    "How are you able to read files?", "How are you built?",
    "What's new in Python 3.13?", "What have you been up to with the engine logs?",
    "What is the capital of France?", "", None,
])
def test_not_check_ins(text):
    assert not is_check_in(text)


@pytest.mark.parametrize("intent,text,expected", [
    ("greeting", "Good afternoon", True),
    ("chat", "How are you today?", True),
    ("greeting", "Hi, what's the capital of France?", False),
    ("chat", "Thanks!", False),
    ("chat", "the capital of Mongolia", False),
])
def test_social_turn(intent, text, expected):
    assert is_social_turn(intent, text) is expected


def _system(text, intent):
    p = CognitionPacket(content=Content(original_prompt=text), intent=Intent(user_intent=intent))
    p.header.model.name = "core"
    return prompt_builder.build_from_packet(p)[0]["content"]


def _compact_core(monkeypatch):
    cfg = prompt_builder.Config()
    monkeypatch.setitem(cfg.constants, "USE_META_VERBS", True)
    monkeypatch.setitem(cfg.MODEL_CONFIGS, "core", {**cfg.MODEL_CONFIGS.get("core", {}), "max_model_len": 4096})


def test_social_turn_keeps_anchor_drops_tools(monkeypatch):
    _compact_core(monkeypatch)
    s = _system("How are you today?", "greeting")
    assert "ongoing research collaboration with Azrael" in s
    assert "This is casual conversation" in s
    for gone in ("Tools:", "KV cache prefix", "THOUGHT_SEED", "access to tools", "Cite relevant"):
        assert gone not in s


def test_factual_chat_keeps_tools(monkeypatch):
    _compact_core(monkeypatch)
    s = _system("the capital of Mongolia", "chat")
    assert "Tools:" in s
    assert "This is casual conversation" not in s


def test_asked_substrate_on_a_greeting_turn_keeps_the_fact(monkeypatch):
    _compact_core(monkeypatch)
    s = _system("Hey, what model are you running on", "greeting")
    assert "Gemma 4 E4B" in s

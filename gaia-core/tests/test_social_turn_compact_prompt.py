"""Core's 4096 window puts every live turn on the compact prompt, which dropped
the casual anchor: greetings and "how are you" got a rules-and-tools block and
answered "I'm holding an open tool (search)". Pins that social turns keep the
anchor and the affect line, and that factual chat turns keep their tools.
"""

import pytest

from gaia_common.protocols.cognition_packet import CognitionPacket, Content, Intent
from gaia_core.utils import prompt_builder
from gaia_core.utils.prompt_builder import is_check_in, is_social_turn, recent_work_fact


@pytest.fixture(autouse=True)
def _known_minds(monkeypatch):
    """What the engines would report; the tests shouldn't need them running."""
    from gaia_common.utils import self_model
    monkeypatch.setattr(self_model, "minds", lambda facts=None: {
        "Prime": {"weights": "Qwen3-VL-8B-GAIA-Prime-v1-abliterated-NF4", "name": "Qwen3-VL-8B", "device": "GPU", "live": True},
        "Core": {"weights": "Gemma4-E4B-GAIA-Core-Multimodal", "name": "Google's Gemma 4 E4B", "device": "CPU", "live": True}})


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


_LOG = [{"hash": "a", "date": "2026-11-02T09:00:00-08:00", "subject": "fix(rbc9): greetings reach Core"},
        {"hash": "b", "date": "2026-11-01T09:00:00-08:00", "subject": "plain subject"},
        {"hash": "c", "date": "2026-10-01T09:00:00-08:00", "subject": "feat: too old"}]


def _now():
    from datetime import datetime
    return datetime.fromisoformat("2026-11-02T12:00:00-08:00")


def _with_log(monkeypatch, log):
    from gaia_common.utils import code_evolution
    monkeypatch.setattr(code_evolution, "recent_git_log", lambda limit=3: log)


def test_activity_question_gets_recent_commits(monkeypatch):
    _with_log(monkeypatch, _LOG)
    fact = recent_work_fact("What have you been up to?", now=_now())
    assert "greetings reach Core; plain subject." in fact
    assert "fix(" not in fact and "too old" not in fact


def test_a_quiet_week_is_stated(monkeypatch):
    _with_log(monkeypatch, _LOG[2:])
    assert "nothing in the past week" in recent_work_fact("what's new", now=_now())


def test_unreadable_log_adds_nothing(monkeypatch):
    _with_log(monkeypatch, [])
    assert recent_work_fact("What have you been up to?", now=_now()) == ""


@pytest.mark.parametrize("text", ["How are you today?", "Good afternoon",
                                  "What's new in Python 3.13?"])
def test_other_turns_get_no_activity_line(monkeypatch, text):
    _with_log(monkeypatch, _LOG)
    assert recent_work_fact(text, now=_now()) == ""


@pytest.mark.parametrize("text", ["what's new?", "How are you today?", "What have you been up to?"])
def test_check_ins_fast_track_to_greeting(text):
    from gaia_core.cognition.nlu.intent_detection import _fast_track_intent_detection
    assert _fast_track_intent_detection(text) == "greeting"

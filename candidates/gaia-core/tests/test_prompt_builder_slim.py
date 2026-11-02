"""Regression test for GAIA_Project-bte: REFLEX-mode few-shot exemplar
ordering was leaking an unrelated topic ("quantum physics") into identity
answers because the ESCALATE exemplar sat immediately before the real user
turn. Pins the fixed ordering: identity exemplar last, no bare topic-word
ESCALATE example adjacent to the real input.
"""

import pytest

from gaia_common.protocols.cognition_packet import CognitionPacket, Content
from gaia_core.utils import prompt_builder
from gaia_core.utils.prompt_builder import NANO_SELF_INTRO, is_self_intro


def _slim_prompt(user_text: str):
    packet = CognitionPacket(content=Content(original_prompt=user_text))
    return prompt_builder.build_from_packet(packet, slim_mode=True)


def test_identity_exemplar_is_last_before_real_turn():
    messages = _slim_prompt("Can you tell me who you are?")
    assert messages[-1] == {"role": "user", "content": "Can you tell me who you are?"}
    assert messages[-2] == {
        "role": "assistant",
        "content": "I'm GAIA, a sovereign AI created by Azrael.",
    }
    assert messages[-3] == {"role": "user", "content": "Who are you?"}


def test_no_quantum_physics_exemplar():
    messages = _slim_prompt("hello")
    joined = " ".join(str(m.get("content", "")) for m in messages)
    assert "quantum physics" not in joined.lower()


def test_escalate_exemplars_not_adjacent_to_real_turn():
    messages = _slim_prompt("hello")
    # The real user turn is last; its immediate predecessor must not be an
    # ESCALATE exemplar (that adjacency is what caused the leak).
    assert messages[-2]["content"] != "ESCALATE"


@pytest.mark.parametrize("text", [
    "Tell me about yourself", "tell me a bit about yourself.", "What are you?", "Describe yourself",
    "Hi GAIA, introduce yourself please", "What is GAIA?", "Can you tell me about yourself?",
])
def test_self_intro_questions_get_the_facts(text):
    assert is_self_intro(text)
    messages = _slim_prompt(text)
    assert messages[-2]["content"] == NANO_SELF_INTRO
    assert messages[-1]["content"] == text


@pytest.mark.parametrize("text", [
    "Who are you?", "What's your name?", "What are you doing?", "What are you able to read?",
    "Tell me about yourself and the engine logs", "What is the capital of France?", "How are you?", "", None,
])
def test_other_turns_do_not(text):
    assert not is_self_intro(text)
    if text:
        assert NANO_SELF_INTRO not in " ".join(str(m["content"]) for m in _slim_prompt(text))


def test_fast_mode_is_core():
    system = _slim_prompt("hello")[0]["content"]
    assert "Core is your fast mode of thought and Prime your deep one" in system
    assert "fast-reflex" not in system

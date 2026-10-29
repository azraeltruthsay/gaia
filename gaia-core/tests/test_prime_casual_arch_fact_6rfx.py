"""6rfx: Prime keeps its architecture fact on casual turns.

7n3 drops the arch fact on casual turns because Core's identity is weight-baked
and the spec recital crowded out her felt state. Prime's weights instead carry a
wrong base-model fact (Prime-v1 learned "built on Google's Gemma 4 E4B"): without
the prompt-borne fact Prime said Gemma 4 E4B on 'chat' turns, with it Qwen3-VL-8B.
"""
from gaia_common.protocols.cognition_packet import CognitionPacket, Content, Intent
from gaia_core.utils import prompt_builder


def _system(target, intent="chat", text="What are you, really?"):
    p = CognitionPacket(content=Content(original_prompt=text), intent=Intent(user_intent=intent))
    p.header.model.name = target
    return prompt_builder.build_from_packet(p)[0]["content"]


def test_prime_casual_turn_keeps_its_architecture_fact():
    s = _system("prime")
    assert "This is casual conversation" in s
    assert "Qwen3-VL-8B" in s


def test_core_casual_turn_still_drops_it():
    s = _system("core", text="hey, how are you?")
    assert "Gemma 4 E4B" not in s

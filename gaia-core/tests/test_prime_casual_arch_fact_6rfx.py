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


def test_core_casual_question_about_her_model_keeps_the_fact():
    # "What model are you running on right now?" is classified chat; without
    # the fact Core said "my own hardware" or refused to disclose it.
    s = _system("core", text="What model are you running on right now?")
    assert "This is casual conversation" in s
    assert "Gemma 4 E4B" in s


def test_casual_mention_of_models_does_not_bring_the_spec_back():
    assert "Gemma 4 E4B" not in _system("core", text="I built a model train set this weekend!")


def test_nano_fast_prompt_names_the_base_model():
    # nano's few-shot prompt had no base model; it said "my own hardware".
    p = CognitionPacket(content=Content(original_prompt="What model are you running on?"),
                        intent=Intent(user_intent="chat"))
    s = prompt_builder.build_from_packet(p, slim_mode=True)[0]["content"]
    assert "Gemma 4 E4B" in s and "Qwen3-VL-8B" in s


def test_arch_fact_matches_the_tier():
    assert "running on the Prime tier" in prompt_builder.arch_fact_for("prime")
    assert "running on the Core tier" in prompt_builder.arch_fact_for("core")
    assert "running on the Core tier" in prompt_builder.arch_fact_for("")

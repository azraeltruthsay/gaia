"""Native-thinking deliberation (GAIA_Project-h1ok).

Core ignored the <think> instructions ~94% of the time (its fine-tunes had
erased Gemma 4's native thinking); the core_thinking_v1 adapter restores
the model's own thought channel. In native mode deliberation opens the system
turn with <|think|>, adds no <think> instructions, and takes the thought from
the engine's reasoning_content rather than parsing it out of the reply.
"""

import os
from unittest import mock

from gaia_core.cognition import deliberation as d


class Pool:
    def __init__(self, message):
        self.message = message
        self.calls = []

    def acquire_model(self, role):
        return object()

    def release_model(self, role):
        pass

    def forward_to_model(self, role, **kw):
        self.calls.append(kw)
        return {"choices": [{"message": dict(self.message)}]}


MSGS = [{"role": "system", "content": "You are GAIA."}, {"role": "user", "content": "2+2?"}]


def _run(message, **kw):
    pool = Pool(message)
    r = d.deliberate("2+2?", MSGS, pool, persist=False, offer_tools=False, **kw)
    return r, pool.calls[0]["messages"][0]["content"]


def test_native_mode_opens_system_turn_and_adds_no_instructions():
    r, system = _run({"content": "Four.", "reasoning_content": "Two plus two is four."},
                     native_thinking=True, adapter_name="core_thinking_v1")
    assert system.startswith("<|think|>\nYou are GAIA.")
    assert "<think>" not in system and "DELIBERATION" not in system
    assert r.thinking == "Two plus two is four."
    assert r.final_response == "Four."
    assert r.fallback_used is False


def test_thought_cut_off_before_answer_is_never_voiced():
    r, _ = _run({"content": "", "reasoning_content": "Still working it out"}, native_thinking=True)
    assert r.final_response == "" and r.fallback_used is True


def test_native_mode_without_channel_falls_back_to_parsing():
    r, _ = _run({"content": "Four."}, native_thinking=True)
    assert r.final_response == "Four." and r.fallback_used is True


def test_instruction_mode_unchanged():
    r, system = _run({"content": "<think>easy</think>Four."})
    assert "DELIBERATION" in system and not system.startswith("<|think|>")
    assert r.thinking == "easy" and r.final_response == "Four."


def _turn_kwargs(env, **kw):
    seen = {}

    def fake(**k):
        seen.update(k)
        return d.DeliberationResult(thinking="", final_response="ok", raw_output="ok",
                                    voice_evidence={}, forbidden_hits=[], elapsed_ms=0.0)
    with mock.patch.dict(os.environ, env), mock.patch.object(d, "deliberate", side_effect=fake):
        d.run_deliberated_turn("hi", MSGS, object(), config={"DELIBERATION": {}}, **kw)
    return seen


def test_flag_selects_thinking_adapter_on_core():
    seen = _turn_kwargs({"DELIBERATION_NATIVE_THINKING": "1"}, model_role="core")
    assert seen["native_thinking"] is True
    assert seen["adapter_name"] == "core_thinking_v1"
    assert seen["max_total_tokens"] > 900


def test_flag_never_applies_to_prime():
    seen = _turn_kwargs({"DELIBERATION_NATIVE_THINKING": "1"}, model_role="prime")
    assert seen["native_thinking"] is False and seen["adapter_name"] is None


def test_flag_off_keeps_deliberation_adapter():
    seen = _turn_kwargs({"DELIBERATION_NATIVE_THINKING": "0"}, model_role="core")
    assert seen["native_thinking"] is False and seen["adapter_name"] == "core_deliberation_v1"

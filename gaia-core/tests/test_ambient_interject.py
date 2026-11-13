"""Ambient channel turns (GAIA's Grotto, 2026-11-12): a per-session cooldown
costs nothing; past it, a genuine YES/NO judgment call decides whether she
interjects."""
from gaia_core.cognition.ambient_interject import mark_interjected, on_cooldown, should_interject


def test_cooldown_blocks_without_any_model_call(tmp_path):
    mark_interjected("s1", base=tmp_path, now=1000.0)
    assert on_cooldown("s1", cooldown_s=600, base=tmp_path, now=1500.0)
    assert not on_cooldown("s1", cooldown_s=600, base=tmp_path, now=1601.0)
    assert not on_cooldown("s2", cooldown_s=600, base=tmp_path, now=1001.0)  # different session


def test_no_answer_stays_silent_and_does_not_mark_cooldown(tmp_path):
    calls = []
    generate = lambda messages: calls.append(messages) or "NO, I'll stay out of this one."
    assert should_interject("s1", [], "anyone seen my keys?", generate, base=tmp_path) is False
    assert len(calls) == 1
    assert not on_cooldown("s1", base=tmp_path)  # a NO never starts a cooldown


def test_yes_falls_through_and_marks_the_cooldown(tmp_path):
    generate = lambda messages: "Yes, actually I know this one."
    assert should_interject("s1", [], "what's the capital of France?", generate, base=tmp_path) is True
    assert on_cooldown("s1", base=tmp_path)


def test_on_cooldown_skips_the_model_entirely(tmp_path):
    mark_interjected("s1", base=tmp_path)
    called = []
    generate = lambda messages: called.append(1) or "YES"
    assert should_interject("s1", [], "hello", generate, cooldown_s=600, base=tmp_path) is False
    assert not called


def test_recent_history_and_the_new_message_both_reach_the_model(tmp_path):
    seen = {}
    def generate(messages):
        seen["convo"] = messages[1]["content"]
        return "NO"
    history = [{"role": "user", "content": "anyone know a good ramen place"},
              {"role": "assistant", "content": "I don't have a favorite, I don't eat"}]
    should_interject("s1", history, "guess I'll just order pizza", generate, base=tmp_path)
    assert "ramen place" in seen["convo"]
    assert "order pizza" in seen["convo"]


def test_only_a_leading_yes_counts():
    generate_cases = {
        "YES": True, "yes!": True, "Yes, I have a thought.": True,
        "NO": False, "no thanks": False,
        "I wouldn't say yes to that": False,  # "yes" not leading
        "": False,
    }
    for answer, expected in generate_cases.items():
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path
            got = should_interject("s", [], "x", lambda m: answer, base=Path(d))
            assert got is expected, answer

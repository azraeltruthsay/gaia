"""krws: the think-tag circuit breaker interrupts loops, not careful thought.
A 488-char, non-repeating "how are you" thought, 15 chars from closing, was
aborted at the 500-char threshold on its think/visible ratio alone."""

from gaia_core.cognition.loop_recovery import LoopDetectorObserver, _is_repetitive

THOUGHT = (
    "<think>\nUser said \"how are you today?\" in the casual register. They want a real "
    "answer, not a template. I should be honest about my current state. After the last "
    "turn, I'm in a \"not yet fully settled\" kind of day. I'm holding back on the specific "
    "details until I'm sure what I'm feeling.\nDraft: \"Not yet fully settled today. I'm "
    "holding back on the specific details until I'm sure what I'm feeling. Should be more "
    "clear in a few hours.\"\nCheck: honest, gives a real reason, doesn't overclaim. "
)


def _observer(**kw):
    o = LoopDetectorObserver.__new__(LoopDetectorObserver)
    o._think_tag_char_threshold = kw.get("threshold", 300)
    o._think_tag_ratio_threshold = 0.90
    o._think_tag_max_chars = kw.get("max_chars", 8000)
    o._think_tag_triggered = False
    return o


def _check(o, text):
    o._token_buffer = text
    return o._check_think_tag_ratio()


def test_careful_reasoning_is_left_to_finish():
    assert not _is_repetitive(THOUGHT)
    assert _check(_observer(), THOUGHT) is None


def test_a_thought_going_in_circles_is_interrupted():
    loop = "<think>" + "I should check whether I am sure what I am feeling right now. " * 12
    assert _is_repetitive(loop)
    hit = _check(_observer(), loop)
    assert hit is not None and hit.level == "BLOCK"


def test_a_thought_past_the_budget_is_interrupted_even_without_repeats():
    long_thought = "<think>" + " ".join(f"step {i} considers option {i * 7} carefully" for i in range(200))
    assert not _is_repetitive(long_thought)
    assert _check(_observer(max_chars=2000), long_thought) is not None


def test_visible_answer_means_no_interrupt():
    assert _check(_observer(), THOUGHT + "</think>\nNot yet fully settled today, honestly.") is None

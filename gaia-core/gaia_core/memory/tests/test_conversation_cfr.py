"""
Tests for 231 Phase 3: CFR deixis/stickiness for meta-reference follow-ups.

A short deictic question ("what's its name?", "the second one", "that link")
refers back to recent content but embeds poorly against that content's body, so
plain cosine relevance would blur the very turn the user is asking about. The
detector widens the recency anchor + relaxes the floor so recent turns survive.
"""
import pytest

from gaia_core.memory.conversation_cfr import looks_like_reference_query, select_focus_turns


@pytest.mark.parametrize("msg", [
    "Can you tell me its name?",
    "What's the second one?",
    "that link please",
    "You just said something interesting",
    "what was that?",
    "which one is better",
])
def test_reference_queries_detected(msg):
    assert looks_like_reference_query(msg) is True


@pytest.mark.parametrize("msg", [
    "Tell me about quantum physics",
    "Write a function to sort a list",
    "Good morning GAIA",
    # long query carries its own topical signal; the 'it' alone shouldn't trip it
    "I would like you to explain in great detail exactly how the transformer "
    "attention mechanism computes its weighted sum over all token positions",
])
def test_non_reference_queries_not_detected(msg):
    assert looks_like_reference_query(msg) is False


def _hist(n):
    # role alternates user/assistant; distinct content so ids are stable by index
    return [{"id": f"m{i}", "role": "user" if i % 2 == 0 else "assistant",
             "content": f"turn number {i} about topic {i}"} for i in range(n)]


def test_reference_query_widens_anchor_keeps_recent_turns():
    history = _hist(6)
    # Tight budget: without the rescue only ~3 turns survive.
    focus_ref, dbg_ref = select_focus_turns(history, "what's its name?", max_focus=3, anchor_n=1)
    # Reference rescue anchors the last 4 turns regardless of cosine score.
    assert dbg_ref.get("reference") is True
    assert len(focus_ref) >= 4
    # the turn 4-back (m2) — which a low-cosine meta query would otherwise blur —
    # is now retained
    assert any(t["id"] == "m2" for t in focus_ref)


def test_non_reference_query_does_not_widen():
    history = _hist(6)
    _, dbg = select_focus_turns(history, "tell me about quantum physics", max_focus=3, anchor_n=1)
    assert dbg.get("reference") in (False, None)


def test_empty_history_safe():
    focus, dbg = select_focus_turns([], "what's its name?", max_focus=3)
    assert focus == []

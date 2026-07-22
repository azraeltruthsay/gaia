"""
Tests for 231 Phase 3: CFR deixis/stickiness for meta-reference follow-ups.

A short deictic question ("what's its name?", "the second one", "that link")
refers back to recent content but embeds poorly against that content's body, so
plain cosine relevance would blur the very turn the user is asking about. The
detector widens the recency anchor + relaxes the floor so recent turns survive.
"""
from datetime import datetime, timedelta, timezone

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


def _iso(dt):
    return dt.isoformat()


def test_stale_dangling_question_not_anchored_into_greeting():
    """tr7f: a 3-day-old unanswered technical question, sitting as the very
    last turn before a much-later greeting, must NOT ride in via the
    anchor (which previously force-kept the trailing turn unconditionally)."""
    now = datetime.now(timezone.utc)
    old_ts = _iso(now - timedelta(hours=72))
    history = [
        {"id": "m1", "role": "user",
         "content": "What kind of docker containers does GAIA run in?",
         "timestamp": old_ts},
    ]
    focus, dbg = select_focus_turns(history, "Good morning GAIA", max_focus=6, anchor_n=1)
    assert dbg.get("anchor", 0) == 0
    assert not any(t["id"] == "m1" for t in focus)


def test_fresh_last_turn_still_anchored():
    """The same shape, but the prior turn is seconds old — normal live
    continuity must be unaffected by the staleness gate."""
    now = datetime.now(timezone.utc)
    fresh_ts = _iso(now - timedelta(seconds=5))
    history = [
        {"id": "m1", "role": "user", "content": "What kind of docker containers does GAIA run in?",
         "timestamp": fresh_ts},
    ]
    focus, dbg = select_focus_turns(history, "Good morning GAIA", max_focus=6, anchor_n=1)
    assert dbg.get("anchor", 0) == 1
    assert any(t["id"] == "m1" for t in focus)


def test_stale_but_topically_relevant_turn_still_excluded_by_scoring():
    """Defense-in-depth for the scored (non-anchor) path: even when a stale
    turn IS genuinely on-topic with the current message (so raw cosine
    relevance alone would admit it), real elapsed time must still gate
    eligibility — a 3-day-old technical turn several positions back in an
    otherwise-unrelated conversation shouldn't outlive its wall-clock
    relevance just because turn-distance decay is weak."""
    now = datetime.now(timezone.utc)
    old_ts = _iso(now - timedelta(hours=72))
    fresh_ts = _iso(now - timedelta(seconds=10))
    history = [
        {"id": "old", "role": "user",
         "content": "What kind of docker containers does GAIA run in?",
         "timestamp": old_ts},
        {"id": "a1", "role": "assistant", "content": "Let's talk later.", "timestamp": old_ts},
        {"id": "u1", "role": "user", "content": "Sounds good, see you then.", "timestamp": fresh_ts},
        {"id": "a2", "role": "assistant", "content": "See you!", "timestamp": fresh_ts},
    ]
    focus, dbg = select_focus_turns(
        history, "which containers am I running again?", max_focus=6, anchor_n=1)
    assert not any(t["id"] == "old" for t in focus)

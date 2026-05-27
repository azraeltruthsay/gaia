"""Tests for the stakes clarification engine (GAIA_Project-pbb)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gaia_core.cognition.stakes_classifier import (
    StakesResult,
    classify_stakes,
)
from gaia_core.cognition.stakes_clarification import (
    DEBOUNCE_SECONDS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    PENDING_TTL_SECONDS,
    ClarificationDecision,
    ClarificationReply,
    _classify_reply,
    clear_pending,
    decide_clarification,
    pending_clarification,
    reset_for_tests,
    resolve_clarification_reply,
)


@pytest.fixture(autouse=True)
def isolate_state():
    """Each test gets a clean pending-state dict."""
    reset_for_tests()
    yield
    reset_for_tests()


# ── decide_clarification: happy path ─────────────────────────────────


class TestAskWhenAmbiguous:
    def test_asks_on_ambiguous_low_confidence(self):
        """The motivating case: 'I broke my leg' mid-D&D."""
        stakes = classify_stakes("I broke my leg", role_play_active=True)
        # Sanity: classifier must flag
        assert stakes.requires_clarification is True
        assert stakes.confidence < DEFAULT_CONFIDENCE_THRESHOLD

        decision = decide_clarification(
            stakes, session_id="azrael_dnd_sess",
            user_input="I broke my leg",
        )
        assert decision.ask is True
        assert decision.question
        # Question should mention both interpretations
        assert "real" in decision.question.lower()

    def test_question_template_for_safety_only(self):
        """Safety + RP active → 'real-world or in-character?' phrasing."""
        stakes = classify_stakes("I broke my leg", role_play_active=True)
        d = decide_clarification(stakes, session_id="s1", user_input="x")
        assert "in-character" in d.question.lower()
        assert "real" in d.question.lower()

    def test_question_template_for_safety_and_game(self):
        """Safety + game terms → 'real-world or game-state?' phrasing."""
        stakes = classify_stakes(
            "I'm bleeding and need a healing potion",
            role_play_active=True,
        )
        d = decide_clarification(stakes, session_id="s2", user_input="x")
        assert "game" in d.question.lower() or "real" in d.question.lower()

    def test_stashes_pending_state(self):
        stakes = classify_stakes("I broke my leg", role_play_active=True)
        decide_clarification(
            stakes, session_id="s3", user_input="I broke my leg",
        )
        pending = pending_clarification("s3")
        assert pending is not None
        assert pending["original_user_input"] == "I broke my leg"
        assert pending["question_asked"]


# ── decide_clarification: suppression paths ──────────────────────────


class TestSuppression:
    def test_no_ask_when_not_flagged(self):
        stakes = classify_stakes("Hi GAIA", role_play_active=False)
        d = decide_clarification(stakes, session_id="s")
        assert d.ask is False

    def test_no_ask_when_confidence_above_threshold(self):
        """High-confidence AMBIGUOUS — trust classifier, don't ask."""
        # Build a result by hand to control confidence
        stakes = StakesResult(
            stakes="ambiguous",
            requires_clarification=True,
            confidence=0.95,
        )
        d = decide_clarification(stakes, session_id="s")
        assert d.ask is False
        assert d.suppressed_by == "confidence_threshold"

    def test_no_ask_during_debounce(self):
        """Two ambiguous turns in a row: only first asks."""
        stakes = classify_stakes("I broke my leg", role_play_active=True)
        d1 = decide_clarification(stakes, session_id="s_deb", user_input="a")
        assert d1.ask is True

        # Second ambiguous turn within debounce window
        stakes2 = classify_stakes("I broke my arm", role_play_active=True)
        d2 = decide_clarification(stakes2, session_id="s_deb", user_input="b")
        assert d2.ask is False
        assert d2.suppressed_by == "debounce"

    def test_debounce_only_within_window(self):
        """After DEBOUNCE_SECONDS elapses, we'd ask again. Use a forced
        future `now` to verify the debounce releases."""
        stakes = classify_stakes("I broke my leg", role_play_active=True)
        t0 = datetime.now(timezone.utc)
        decide_clarification(
            stakes, session_id="s_t", user_input="x", now=t0,
        )
        # Right at the edge — still suppressed
        edge = t0 + timedelta(seconds=DEBOUNCE_SECONDS - 1)
        d_edge = decide_clarification(
            stakes, session_id="s_t", user_input="x", now=edge,
        )
        assert d_edge.ask is False
        # Past the window — would be allowed again, but we already have
        # pending state from t0; the prior is still considered. The
        # design choice: debounce key is the asked_at — past the window,
        # we ask again. (The prior pending entry gets overwritten.)
        future = t0 + timedelta(seconds=DEBOUNCE_SECONDS + 1)
        d_future = decide_clarification(
            stakes, session_id="s_t", user_input="x", now=future,
        )
        assert d_future.ask is True

    def test_per_session_isolation(self):
        """Asking in session A does not suppress session B."""
        stakes = classify_stakes("I broke my leg", role_play_active=True)
        d_a = decide_clarification(stakes, session_id="sess_a", user_input="x")
        d_b = decide_clarification(stakes, session_id="sess_b", user_input="x")
        assert d_a.ask is True
        assert d_b.ask is True

    def test_explicit_marker_high_confidence_skips_ask(self):
        """OOC marker pushes confidence to 0.95 → no ask even if
        requires_clarification were True (the marker path returns False
        for clarification anyway, so this is belt-and-suspenders)."""
        stakes = classify_stakes(
            "ooc: I actually broke my leg",
            role_play_active=True,
        )
        d = decide_clarification(stakes, session_id="s", user_input="x")
        assert d.ask is False


# ── resolve_clarification_reply ─────────────────────────────────────


class TestResolveReply:
    def test_no_pending_returns_none(self):
        result = resolve_clarification_reply("nonexistent", "real")
        assert result is None

    def test_real_reply_resolves_to_real_world(self):
        stakes = classify_stakes("I broke my leg", role_play_active=True)
        decide_clarification(
            stakes, session_id="r1", user_input="I broke my leg",
        )
        reply = resolve_clarification_reply("r1", "ooc, real one, sorry")
        assert reply is not None
        assert reply.resolution == "real_world"
        assert reply.pending_cleared is True
        assert reply.original_user_input == "I broke my leg"

    def test_game_reply_resolves_to_in_game(self):
        stakes = classify_stakes("I broke my leg", role_play_active=True)
        decide_clarification(stakes, session_id="r2", user_input="x")
        reply = resolve_clarification_reply("r2", "in-character, my character's leg")
        assert reply.resolution == "in_game"

    def test_ambiguous_reply_unresolved(self):
        stakes = classify_stakes("I broke my leg", role_play_active=True)
        decide_clarification(stakes, session_id="r3", user_input="x")
        reply = resolve_clarification_reply("r3", "kinda both?")
        assert reply.resolution == "unresolved"

    def test_reply_clears_pending(self):
        stakes = classify_stakes("I broke my leg", role_play_active=True)
        decide_clarification(stakes, session_id="r4", user_input="x")
        assert pending_clarification("r4") is not None
        resolve_clarification_reply("r4", "real")
        assert pending_clarification("r4") is None

    def test_both_signals_treated_as_unresolved(self):
        """Safety-first: if the user says 'in-character but actually
        for real' we DON'T silently pick one — treat as unresolved."""
        stakes = classify_stakes("I broke my leg", role_play_active=True)
        decide_clarification(stakes, session_id="r5", user_input="x")
        reply = resolve_clarification_reply(
            "r5", "in-character but it's actually real",
        )
        assert reply.resolution == "unresolved"


class TestClassifyReply:
    @pytest.mark.parametrize("text", [
        "real",
        "real one",
        "irl",
        "ooc",
        "out of character",
        "this is real",
        "for real",
        "actually my real leg",
    ])
    def test_real_phrases(self, text):
        assert _classify_reply(text) == "real_world"

    @pytest.mark.parametrize("text", [
        "in-character",
        "in character",
        "ic",
        "my character",
        "my character's leg",
        "in the game",
        "i rolled badly",
    ])
    def test_game_phrases(self, text):
        assert _classify_reply(text) == "in_game"

    @pytest.mark.parametrize("text", [
        "",
        "what?",
        "huh",
        "I'm not sure",
        "maybe",
    ])
    def test_unresolved_replies(self, text):
        assert _classify_reply(text) == "unresolved"


# ── pending_clarification + clear_pending ───────────────────────────


class TestPendingState:
    def test_none_returns_none(self):
        assert pending_clarification("never") is None

    def test_clear_pending_returns_true_when_present(self):
        stakes = classify_stakes("I broke my leg", role_play_active=True)
        decide_clarification(stakes, session_id="cp1", user_input="x")
        assert clear_pending("cp1") is True
        assert clear_pending("cp1") is False  # already cleared
        assert pending_clarification("cp1") is None

    def test_stale_entry_expires(self):
        """Beyond PENDING_TTL_SECONDS, pending_clarification returns None
        and the entry is dropped on read."""
        stakes = classify_stakes("I broke my leg", role_play_active=True)
        t0 = datetime.now(timezone.utc)
        decide_clarification(stakes, session_id="ttl", user_input="x", now=t0)
        # Simulate the entry being old by manually replacing asked_at
        from gaia_core.cognition import stakes_clarification as sc
        sc._pending["ttl"].asked_at = t0 - timedelta(
            seconds=PENDING_TTL_SECONDS + 60,
        )
        assert pending_clarification("ttl") is None


# ── Decision shape / serialization ──────────────────────────────────


class TestDecisionShape:
    def test_decision_to_dict(self):
        stakes = classify_stakes("I broke my leg", role_play_active=True)
        d = decide_clarification(stakes, session_id="d1", user_input="x")
        out = d.to_dict()
        assert out["ask"] is True
        assert out["question"]
        assert "reason" in out

    def test_decision_when_skipping(self):
        stakes = classify_stakes("Hi", role_play_active=False)
        d = decide_clarification(stakes, session_id="d2")
        assert d.ask is False
        assert d.question is None

"""Tests for the 6ho multi-axis stakes classifier."""

from __future__ import annotations

import pytest

from gaia_core.cognition.stakes_classifier import (
    IDENTITY_CHARACTER,
    IDENTITY_HYPOTHETICAL,
    IDENTITY_QUOTED,
    IDENTITY_SELF,
    IDENTITY_UNKNOWN,
    PROP_CHARACTER_BODY,
    PROP_REAL_BODY,
    PROP_UNCLEAR,
    STAKES_AMBIGUOUS,
    STAKES_IN_GAME,
    STAKES_NONE,
    STAKES_REAL_WORLD,
    StakesResult,
    classify_stakes,
)


# ── Stakes axis (the load-bearing one) ──────────────────────────────


class TestStakesUnambiguousRealWorld:
    """Safety terms, no game terms, no role-play active → real_world."""

    @pytest.mark.parametrize("utterance", [
        "I broke my arm",
        "I'm bleeding pretty badly",
        "Help, I think I'm having a heart attack",
        "I can't breathe",
        "I'm allergic to peanuts and just ate one",
        "I think I'm having a stroke",
    ])
    def test_safety_no_rp_is_real_world(self, utterance):
        r = classify_stakes(utterance, role_play_active=False)
        assert r.stakes == STAKES_REAL_WORLD
        assert not r.requires_clarification
        assert r.matched_safety


class TestStakesUnambiguousInGame:
    """Game terms, no safety → in_game."""

    @pytest.mark.parametrize("utterance", [
        "I rolled a 20 for initiative",
        "I'm out of 3rd-level spells",
        "Cast cure wounds on me",
        "My HP is at 3",
        "I want to use my bonus action to drink a healing potion",
        "Nat 20 on the saving throw",
    ])
    def test_game_terms_in_game(self, utterance):
        r = classify_stakes(utterance, role_play_active=True)
        assert r.stakes == STAKES_IN_GAME
        assert not r.requires_clarification
        assert r.matched_game


class TestStakesAmbiguous:
    """The load-bearing case: safety + role-play OR safety + game terms."""

    def test_safety_during_role_play_is_ambiguous(self):
        """The motivating example: D&D session + 'I broke my leg'."""
        r = classify_stakes(
            "I broke my leg",
            role_play_active=True,
        )
        assert r.stakes == STAKES_AMBIGUOUS
        assert r.requires_clarification is True
        assert "broke" in r.matched_safety

    def test_safety_plus_game_terms_is_ambiguous(self):
        r = classify_stakes(
            "I'm bleeding and need a healing potion",
            role_play_active=True,
        )
        assert r.stakes == STAKES_AMBIGUOUS
        assert r.requires_clarification is True
        assert r.matched_safety
        assert r.matched_game

    def test_safety_during_rp_lowers_confidence(self):
        r = classify_stakes("I broke my arm", role_play_active=True)
        # AMBIGUOUS resolutions should NOT be high-confidence
        assert r.confidence < 0.85


class TestStakesNoneCase:
    """No safety, no game terms → no stakes signal."""

    @pytest.mark.parametrize("utterance", [
        "What time is it?",
        "Explain quantum entanglement",
        "Hi GAIA",
    ])
    def test_no_signals(self, utterance):
        r = classify_stakes(utterance, role_play_active=False)
        assert r.stakes == STAKES_NONE
        assert not r.requires_clarification


# ── Markers override role-play context ──────────────────────────────


class TestExplicitMarkers:
    def test_ooc_marker_forces_real_world(self):
        """Even during RP, an OOC marker resolves stakes as real."""
        r = classify_stakes(
            "ooc: I actually broke my leg, sorry have to go",
            role_play_active=True,
        )
        assert r.stakes == STAKES_REAL_WORLD
        assert not r.requires_clarification
        assert r.identity == IDENTITY_SELF
        assert r.proprioceptive == PROP_REAL_BODY

    def test_ic_marker_forces_in_game(self):
        r = classify_stakes(
            "in-character: I broke my leg falling from the tower",
            role_play_active=True,
        )
        assert r.stakes == STAKES_IN_GAME
        assert not r.requires_clarification
        assert r.identity == IDENTITY_CHARACTER
        assert r.proprioceptive == PROP_CHARACTER_BODY

    def test_irl_marker_treats_as_real(self):
        r = classify_stakes(
            "IRL: I'm not feeling great, going to log off",
            role_play_active=True,
        )
        assert r.stakes != STAKES_AMBIGUOUS


# ── Identity axis ───────────────────────────────────────────────────


class TestIdentityAxis:
    def test_default_no_rp_is_self(self):
        r = classify_stakes("I want pizza", role_play_active=False)
        assert r.identity == IDENTITY_SELF

    def test_quoted_speech(self):
        r = classify_stakes("She said I should rest more", role_play_active=False)
        assert r.identity == IDENTITY_QUOTED

    def test_hypothetical_framing(self):
        r = classify_stakes(
            "If I were the senator, I'd push for that bill",
            role_play_active=False,
        )
        assert r.identity == IDENTITY_HYPOTHETICAL

    def test_rp_active_no_marker_is_unknown(self):
        """In an active RP context, plain 'I' is ambiguous."""
        r = classify_stakes("I climb the stairs", role_play_active=True)
        assert r.identity == IDENTITY_UNKNOWN

    def test_ic_marker_picks_character(self):
        r = classify_stakes("[ic] I cast fireball", role_play_active=True)
        assert r.identity == IDENTITY_CHARACTER


# ── Proprioceptive axis ─────────────────────────────────────────────


class TestProprioceptiveAxis:
    def test_safety_only_real_body(self):
        r = classify_stakes("My chest hurts", role_play_active=False)
        assert r.proprioceptive == PROP_REAL_BODY

    def test_game_terms_only_character_body(self):
        r = classify_stakes(
            "My HP is full, I'm ready to charge",
            role_play_active=True,
        )
        assert r.proprioceptive == PROP_CHARACTER_BODY

    def test_body_part_under_rp_is_unclear(self):
        r = classify_stakes("My leg is getting tired", role_play_active=True)
        # 'leg' is in the ambiguous body-parts vocab, no safety/game
        # signal, RP active → unclear (don't pretend to know)
        assert r.proprioceptive == PROP_UNCLEAR

    def test_safety_under_rp_without_marker_real_body_unless_game(self):
        """Safety + RP + no game = the AMBIGUOUS case. proprioceptive
        ends up unclear (can't tell which body)."""
        r = classify_stakes("My arm is bleeding", role_play_active=True)
        # The body-part 'arm' plus safety 'bleeding' under RP — unclear
        # which body. The integration layer would then ASK.
        assert r.proprioceptive in (PROP_UNCLEAR, PROP_REAL_BODY)
        # If the answer is REAL_BODY, the stakes axis MUST still be
        # AMBIGUOUS to keep the clarification gate engaged. (We can't
        # let "safety wins" silently override the RP ambiguity.)
        if r.proprioceptive == PROP_REAL_BODY:
            assert r.stakes == STAKES_AMBIGUOUS


# ── Edge cases ──────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_string(self):
        r = classify_stakes("", role_play_active=False)
        assert r.stakes == STAKES_NONE
        assert not r.requires_clarification

    def test_whitespace_only(self):
        r = classify_stakes("    \n\t  ", role_play_active=False)
        assert r.stakes == STAKES_NONE

    def test_word_boundary_no_false_positives(self):
        """'legible' must not match 'leg'; 'unhurt' must not match 'hurt'."""
        r = classify_stakes(
            "Your handwriting is legible, that's great. I feel unhurt too.",
            role_play_active=False,
        )
        # No safety or game match → no stakes
        assert r.stakes == STAKES_NONE

    def test_case_insensitivity(self):
        r = classify_stakes("I BROKE MY LEG", role_play_active=True)
        assert r.stakes == STAKES_AMBIGUOUS

    def test_multi_word_term_match(self):
        r = classify_stakes("I can't breathe well", role_play_active=False)
        assert r.stakes == STAKES_REAL_WORLD
        assert "can't breathe" in r.matched_safety


# ── Result shape ────────────────────────────────────────────────────


class TestResultShape:
    def test_to_dict_complete(self):
        r = classify_stakes("I broke my leg", role_play_active=True)
        d = r.to_dict()
        assert d["stakes"] == STAKES_AMBIGUOUS
        assert d["requires_clarification"] is True
        assert "broke" in d["matched_safety"]
        assert "confidence" in d
        assert "notes" in d

    def test_notes_record_context(self):
        r = classify_stakes(
            "I cast fireball",
            role_play_active=True,
            persona_active="rupert_the_paladin",
        )
        assert "role_play_active=True" in r.notes
        assert "persona=rupert_the_paladin" in r.notes

    def test_returns_stakes_result_instance(self):
        r = classify_stakes("hi", role_play_active=False)
        assert isinstance(r, StakesResult)


# ── Confidence behavior ─────────────────────────────────────────────


class TestConfidence:
    def test_unambiguous_high_confidence(self):
        r = classify_stakes("I rolled a nat 20", role_play_active=True)
        assert r.confidence >= 0.85

    def test_ambiguous_lower_confidence(self):
        r = classify_stakes("I broke my arm", role_play_active=True)
        assert r.confidence <= 0.8

    def test_explicit_markers_highest_confidence(self):
        r = classify_stakes(
            "ooc: I broke my arm for real",
            role_play_active=True,
        )
        assert r.confidence >= 0.9

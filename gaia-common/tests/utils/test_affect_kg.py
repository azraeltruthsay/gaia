"""Tests for AffectKG (GAIA_Project-usv, Phase 1 data layer).

Locks in:
  - Predicate constructor canonicalization
  - Multiple simultaneous feelings (the bug that motivated the prefixed-
    predicate + sentinel-object design — no contradictions between
    different emotions held at the same time)
  - Intensity update path closes the prior triple and inserts new
  - Context overlay activation / deactivation / idempotency
  - Inheritance: overlay triples shadow actuality
  - flatten_current_affect returns the right shape and respects decay
  - Theory of mind lives in belief_of worlds and doesn't leak into self
  - Decay halflife behavior pins so Stage 7 can replace the kernel
"""

from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def kg(tmp_path):
    from gaia_common.utils.knowledge_graph import KnowledgeGraph
    return KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite"))


@pytest.fixture
def affect(kg):
    from gaia_common.utils.affect_kg import AffectKG
    return AffectKG(kg)


# ── Predicate canonicalization ──────────────────────────────────────

class TestPredicates:
    def test_feels_lowercases_and_snake_cases(self):
        from gaia_common.utils.affect_predicates import pred_feels
        assert pred_feels("Irritation") == "feels_irritation"
        assert pred_feels("HUNGER FOR NOVELTY") == "feels_hunger_for_novelty"

    def test_invalid_suffix_raises(self):
        from gaia_common.utils.affect_predicates import pred_trait
        with pytest.raises(ValueError):
            pred_trait("")
        with pytest.raises(ValueError):
            pred_trait("   ")

    def test_context_world_name_idempotent(self):
        from gaia_common.utils.affect_predicates import context_world_name
        assert context_world_name("dnd session") == "ctx_dnd_session"
        # Already prefixed — don't double-prefix
        assert context_world_name("ctx_dnd_session") == "ctx_dnd_session"


# ── Simultaneous feelings (the motivating case) ─────────────────────

class TestSimultaneousFeelings:
    def test_multiple_emotions_coexist(self, affect):
        """The key requirement: holding irritation AND curiosity at once."""
        affect.record_feeling("irritation", 0.3)
        affect.record_feeling("curiosity", 0.8)
        affect.record_feeling("calm", 0.4)
        snap = affect.flatten_current_affect()
        assert "irritation" in snap["feels"]
        assert "curiosity" in snap["feels"]
        assert "calm" in snap["feels"]
        # Sanity: values preserved (within decay tolerance — but new
        # triples are fresh so decay is ~0).
        assert snap["feels"]["irritation"] > 0.25
        assert snap["feels"]["curiosity"] > 0.75
        assert snap["feels"]["calm"] > 0.35

    def test_traits_drives_attention_coexist(self, affect):
        affect.record_trait("curiosity", 0.9)
        affect.record_trait("warmth", 0.7)
        affect.record_drive("hunger_for_novelty", 0.6)
        affect.record_curious_about("consistency_detector", 0.85)
        affect.record_tired_of("dnd_session", 0.4)

        snap = affect.flatten_current_affect()
        assert snap["traits"]["curiosity"] > 0.85
        assert snap["traits"]["warmth"] > 0.65
        assert snap["drives"]["hunger_for_novelty"] > 0.55
        assert snap["curious_about"]["consistency_detector"] > 0.8
        assert snap["tired_of"]["dnd_session"] > 0.35


# ── Update path ─────────────────────────────────────────────────────

class TestUpdatePath:
    def test_update_replaces_open_triple(self, affect):
        affect.record_feeling("irritation", 0.2)
        affect.record_feeling("irritation", 0.7)  # update
        snap = affect.flatten_current_affect()
        # Only the new value is in the open set.
        assert snap["feels"]["irritation"] > 0.65
        assert snap["feels"]["irritation"] < 0.75

    def test_history_preserved_with_include_closed(self, affect):
        affect.record_feeling("irritation", 0.2)
        affect.record_feeling("irritation", 0.7)
        snap_all = affect.flatten_current_affect(include_closed=True)
        # Closed triples participate — the max is reported.
        assert snap_all["feels"]["irritation"] >= 0.65


# ── Context overlays ────────────────────────────────────────────────

class TestContexts:
    def test_activate_creates_ephemeral_world(self, affect, kg):
        w = affect.activate_context("dnd_session", ttl_seconds=300)
        assert w == "ctx_dnd_session"
        meta = kg.get_world(w)
        assert meta is not None
        assert meta["modality"] == "context"
        assert meta["lifecycle"] == "ephemeral"
        assert meta["expires_at"] is not None

    def test_activate_idempotent(self, affect, kg):
        w1 = affect.activate_context("dnd_session", ttl_seconds=300)
        w2 = affect.activate_context("dnd_session", ttl_seconds=600)
        assert w1 == w2
        # Did not double-create
        meta = kg.get_world(w1)
        assert meta is not None

    def test_deactivate_removes(self, affect, kg):
        affect.activate_context("dnd_session", ttl_seconds=300)
        assert affect.deactivate_context("dnd_session") is True
        assert kg.get_world("ctx_dnd_session") is None

    def test_deactivate_unknown_returns_false(self, affect):
        assert affect.deactivate_context("never_active") is False


# ── Inheritance: overlay shadows actuality ──────────────────────────

class TestInheritance:
    def test_overlay_trait_shadows_actuality(self, affect, kg):
        # Base actuality persona
        affect.record_trait("playfulness", 0.3)
        # DnD session boosts playfulness
        affect.activate_context("dnd_session", ttl_seconds=600)
        affect.record_trait("playfulness", 0.9, world="ctx_dnd_session")

        # Without context: base value
        snap_base = affect.flatten_current_affect()
        assert snap_base["traits"]["playfulness"] < 0.4

        # With context: overlay value shadows
        snap_dnd = affect.flatten_current_affect(active_context="ctx_dnd_session")
        assert snap_dnd["traits"]["playfulness"] > 0.85
        assert snap_dnd["active_context"] == "ctx_dnd_session"

    def test_overlay_doesnt_leak_to_actuality(self, affect):
        affect.activate_context("private_mode", ttl_seconds=300)
        affect.record_feeling("anxious", 0.8, world="ctx_private_mode")
        # Queried without active_context, the overlay feeling is invisible
        snap_base = affect.flatten_current_affect()
        assert "anxious" not in snap_base["feels"]


# ── Theory of mind ──────────────────────────────────────────────────

class TestTheoryOfMind:
    def test_belief_lives_in_belief_of_world(self, affect, kg):
        affect.record_belief_about("azrael", "current_mood", "focused", 0.8)
        # The belief world was auto-created
        meta = kg.get_world("belief_of_azrael")
        assert meta is not None
        assert meta["modality"] == "belief_of"

    def test_belief_doesnt_leak_to_self(self, affect):
        affect.record_belief_about("azrael", "current_mood", "frustrated", 0.7)
        # GAIA's own affect snapshot does NOT include Azrael's mood.
        snap = affect.flatten_current_affect()
        assert "frustrated" not in snap["feels"]
        assert all(
            "azrael" not in t for t in snap["traits"]
        ), f"Azrael bled into self.traits: {snap['traits']}"

    def test_belief_about_returns_attribute_values(self, affect):
        affect.record_belief_about("azrael", "current_mood", "focused", 0.8)
        affect.record_belief_about(
            "azrael", "curious_about_topic", "audio_pipeline", 0.7,
        )
        beliefs = affect.belief_about("azrael")
        assert "current_mood" in beliefs
        assert beliefs["current_mood"]["value"] == "focused"
        assert beliefs["current_mood"]["confidence"] > 0.75
        assert "curious_about_topic" in beliefs
        assert beliefs["curious_about_topic"]["value"] == "audio_pipeline"

    def test_belief_about_unknown_person(self, affect):
        assert affect.belief_about("nobody_real") == {}


# ── Decay ───────────────────────────────────────────────────────────

class TestDecay:
    def test_feeling_decays_over_a_halflife(self, affect):
        from gaia_common.utils.affect_predicates import HALFLIFE_BY_PREFIX, PREFIX_FEELS
        hl = HALFLIFE_BY_PREFIX[PREFIX_FEELS]
        affect.record_feeling("irritation", 0.8)
        # Snapshot one half-life in the future — value should ~halve
        future = datetime.now(timezone.utc) + timedelta(seconds=hl)
        snap = affect.flatten_current_affect(now=future)
        assert 0.35 < snap["feels"]["irritation"] < 0.45

    def test_traits_decay_slower_than_feelings(self, affect):
        from gaia_common.utils.affect_predicates import HALFLIFE_BY_PREFIX, PREFIX_FEELS
        affect.record_feeling("irritation", 0.8)
        affect.record_trait("curiosity", 0.8)
        # Halflife for feelings is 12h; trait halflife is 7 days.
        future = datetime.now(timezone.utc) + timedelta(seconds=HALFLIFE_BY_PREFIX[PREFIX_FEELS])
        snap = affect.flatten_current_affect(now=future)
        assert snap["traits"]["curiosity"] > snap["feels"]["irritation"]

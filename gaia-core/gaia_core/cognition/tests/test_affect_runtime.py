"""Tests for affect_runtime (GAIA_Project-usv Phase 2).

Pins behavior of the runtime affect surface:
  - graceful no-op when AffectKG init fails
  - prompt lines rendered with the right format and threshold
  - inference modulation rules fire on the right traits/feelings
  - render_into_identity_lines mutates the list, never raises
"""

import pytest


@pytest.fixture
def affect_with_kg(tmp_path):
    """Build a real AffectKG on a tmp_path KG and inject it into the runtime."""
    from gaia_common.utils.knowledge_graph import KnowledgeGraph
    from gaia_common.utils.affect_kg import AffectKG
    from gaia_core.cognition import affect_runtime

    kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite"))
    affect = AffectKG(kg)
    affect_runtime.reset_for_tests(affect)
    yield affect
    affect_runtime.reset_for_tests(None)


# ── Snapshot / safety ──────────────────────────────────────────────

class TestSnapshotSafety:
    def test_empty_kg_returns_empty_snapshot(self, affect_with_kg):
        from gaia_core.cognition.affect_runtime import current_affect_snapshot
        snap = current_affect_snapshot()
        assert snap["feels"] == {}
        assert snap["drives"] == {}
        assert snap["traits"] == {}

    def test_no_kg_configured_returns_safe_empty(self):
        from gaia_core.cognition import affect_runtime
        affect_runtime.reset_for_tests(None)
        # Force _get_affect_kg to fail-fast — _affect_kg stays None,
        # second call uses the cached None; but the lazy init would
        # try to build a real KG. We patch _get_affect_kg to None.
        import gaia_core.cognition.affect_runtime as ar
        orig_get = ar._get_affect_kg
        ar._get_affect_kg = lambda: None
        try:
            snap = ar.current_affect_snapshot()
            assert snap["feels"] == {}
            assert snap["traits"] == {}
            # affect_state_lines on the empty snapshot is empty
            assert ar.affect_state_lines(snap) == []
        finally:
            ar._get_affect_kg = orig_get


# ── Prompt lines ───────────────────────────────────────────────────

class TestPromptLines:
    def test_no_axis_above_threshold_returns_no_lines(self, affect_with_kg):
        from gaia_core.cognition.affect_runtime import (
            affect_state_lines, current_affect_snapshot,
        )
        # Record a feeling below the prompt threshold (0.15)
        affect_with_kg.record_feeling("mild_curiosity", 0.10)
        lines = affect_state_lines(current_affect_snapshot())
        assert lines == []

    def test_feels_line_format(self, affect_with_kg):
        from gaia_core.cognition.affect_runtime import (
            affect_state_lines, current_affect_snapshot,
        )
        affect_with_kg.record_feeling("irritation", 0.6)
        affect_with_kg.record_feeling("curiosity", 0.8)
        lines = affect_state_lines(current_affect_snapshot())
        # One feels line (drives/focus/aversion empty)
        feels_lines = [l for l in lines if "feels" in l]
        assert len(feels_lines) == 1
        # Both emotions present, formatted as name=value
        assert "curiosity=" in feels_lines[0]
        assert "irritation=" in feels_lines[0]

    def test_drives_focus_aversion_separate_lines(self, affect_with_kg):
        from gaia_core.cognition.affect_runtime import (
            affect_state_lines, current_affect_snapshot,
        )
        affect_with_kg.record_drive("hunger_for_novelty", 0.7)
        affect_with_kg.record_curious_about("consistency_detector", 0.85)
        affect_with_kg.record_tired_of("dnd_session", 0.4)
        lines = affect_state_lines(current_affect_snapshot())
        # One line each per non-empty axis
        assert any("drives" in l and "hunger_for_novelty" in l for l in lines)
        assert any("focus" in l and "consistency_detector" in l for l in lines)
        assert any("aversion" in l and "dnd_session" in l for l in lines)

    def test_top_k_cap(self, affect_with_kg):
        from gaia_core.cognition.affect_runtime import (
            affect_state_lines, current_affect_snapshot,
        )
        for i in range(8):
            affect_with_kg.record_curious_about(f"topic_{i}", 0.3 + i * 0.05)
        lines = affect_state_lines(current_affect_snapshot())
        focus_line = [l for l in lines if "focus" in l][0]
        # Capped at 4 — count `=` signs
        assert focus_line.count("=") == 4


# ── Inference modulation ──────────────────────────────────────────

class TestInferenceParams:
    def test_default_modulation_is_neutral(self, affect_with_kg):
        from gaia_core.cognition.affect_runtime import (
            affect_inference_params, current_affect_snapshot,
        )
        params = affect_inference_params(current_affect_snapshot())
        assert params["temperature_delta"] == 0.0
        assert params["max_tokens_multiplier"] == 1.0
        assert params["escalate_to_prime"] is False
        assert params["style_hint"] is None
        assert params["reasons"] == []

    def test_high_caution_plus_logic_escalates(self, affect_with_kg):
        from gaia_core.cognition.affect_runtime import (
            affect_inference_params, current_affect_snapshot,
        )
        affect_with_kg.record_trait("caution", 0.8)
        affect_with_kg.record_trait("logic_priority", 0.85)
        params = affect_inference_params(current_affect_snapshot())
        assert params["escalate_to_prime"] is True
        assert any("escalate" in r for r in params["reasons"])

    def test_irritation_caps_temperature(self, affect_with_kg):
        from gaia_core.cognition.affect_runtime import (
            affect_inference_params, current_affect_snapshot,
        )
        affect_with_kg.record_feeling("irritation", 0.75)
        params = affect_inference_params(current_affect_snapshot())
        assert params["temperature_delta"] < 0
        assert params["style_hint"] == "measured"

    def test_curiosity_expands_tokens(self, affect_with_kg):
        from gaia_core.cognition.affect_runtime import (
            affect_inference_params, current_affect_snapshot,
        )
        affect_with_kg.record_feeling("curiosity", 0.8)
        params = affect_inference_params(current_affect_snapshot())
        assert params["max_tokens_multiplier"] > 1.0
        assert params["style_hint"] == "exploratory"

    def test_fatigue_contracts_and_overrides_style(self, affect_with_kg):
        from gaia_core.cognition.affect_runtime import (
            affect_inference_params, current_affect_snapshot,
        )
        # Curiosity would say "exploratory" but fatigue should win.
        affect_with_kg.record_feeling("curiosity", 0.7)
        affect_with_kg.record_feeling("fatigue", 0.7)
        params = affect_inference_params(current_affect_snapshot())
        assert params["style_hint"] == "terse"
        # Fatigue ×0.8 then prior ×1.3 ⇒ 1.04 — caller will floor
        # at some sane minimum; we just check direction.
        assert params["max_tokens_multiplier"] < 1.3


# ── Render-into-identity-lines (the prompt builder hook) ──────────

class TestRenderIntoIdentityLines:
    def test_appends_when_state_present(self, affect_with_kg):
        from gaia_core.cognition.affect_runtime import render_into_identity_lines
        affect_with_kg.record_feeling("irritation", 0.55)
        lines: list[str] = ["You are GAIA.", "Traits: curiosity: 0.9"]
        render_into_identity_lines(lines)
        assert len(lines) >= 3  # at least one affect line appended
        assert any("irritation" in l for l in lines)
        # Static identity lines preserved unchanged at positions 0-1
        assert lines[0] == "You are GAIA."
        assert lines[1].startswith("Traits:")

    def test_noop_when_empty_kg(self, affect_with_kg):
        from gaia_core.cognition.affect_runtime import render_into_identity_lines
        lines: list[str] = ["You are GAIA."]
        before = list(lines)
        render_into_identity_lines(lines)
        assert lines == before

    def test_never_raises(self):
        """Even with broken/missing KG, the renderer must not raise."""
        from gaia_core.cognition import affect_runtime
        affect_runtime.reset_for_tests(None)
        # Force broken init
        orig_get = affect_runtime._get_affect_kg
        affect_runtime._get_affect_kg = lambda: None
        try:
            lines: list[str] = ["base"]
            affect_runtime.render_into_identity_lines(lines)
            assert lines == ["base"]
        finally:
            affect_runtime._get_affect_kg = orig_get

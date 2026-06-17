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


# ── Phase 3: sampler modulation ─────────────────────────────────────


class TestApplyAffectModulation:
    def test_neutral_affect_passes_baseline_through(self, affect_with_kg):
        from gaia_core.cognition.affect_runtime import apply_affect_modulation
        t, m, dbg = apply_affect_modulation(0.7, 1024)
        assert t == 0.7
        assert m == 1024
        assert dbg["reasons"] == []

    def test_curiosity_expands_max_tokens(self, affect_with_kg):
        from gaia_core.cognition.affect_runtime import apply_affect_modulation
        affect_with_kg.record_feeling("curiosity", 0.8)
        _, m, dbg = apply_affect_modulation(0.7, 1000)
        assert m > 1000
        assert any("exploratory" in r for r in dbg["reasons"])

    def test_irritation_lowers_temperature(self, affect_with_kg):
        from gaia_core.cognition.affect_runtime import apply_affect_modulation
        affect_with_kg.record_feeling("irritation", 0.8)
        t, _, dbg = apply_affect_modulation(0.7, 1000)
        assert t < 0.7
        assert dbg["style_hint"] == "measured"

    def test_temperature_delta_is_bounded(self, affect_with_kg):
        """Even with maximum irritation, temp can't dive below 0.0."""
        from gaia_core.cognition.affect_runtime import apply_affect_modulation
        affect_with_kg.record_feeling("irritation", 1.0)
        affect_with_kg.record_feeling("fatigue", 1.0)
        t, m, _ = apply_affect_modulation(0.1, 100)
        assert t >= 0.0
        # max_tokens never falls below the floor
        assert m >= 64

    def test_failure_returns_baseline(self):
        """Broken kg → baseline values pass through, no exception."""
        from gaia_core.cognition import affect_runtime
        affect_runtime.reset_for_tests(None)
        orig = affect_runtime._get_affect_kg
        affect_runtime._get_affect_kg = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            t, m, _ = affect_runtime.apply_affect_modulation(0.5, 500)
            assert t == 0.5
            assert m == 500
        finally:
            affect_runtime._get_affect_kg = orig


# ── Phase 3: context detection ──────────────────────────────────────


class TestDetectContexts:
    def test_no_input_returns_empty(self):
        from gaia_core.cognition.affect_runtime import detect_contexts
        assert detect_contexts("") == []
        assert detect_contexts(None) == []

    def test_dnd_command_detected(self):
        from gaia_core.cognition.affect_runtime import detect_contexts
        assert "dnd_session" in detect_contexts("/roll 1d20+3")

    def test_debug_keywords_detected(self):
        from gaia_core.cognition.affect_runtime import detect_contexts
        assert "coding_debug" in detect_contexts(
            "Here's the traceback I'm seeing..."
        )

    def test_research_keywords_detected(self):
        from gaia_core.cognition.affect_runtime import detect_contexts
        assert "research_mode" in detect_contexts(
            "Can you survey the literature on consistency models?"
        )

    def test_no_match_returns_empty(self):
        from gaia_core.cognition.affect_runtime import detect_contexts
        assert detect_contexts("Hey, how are you?") == []

    def test_multiple_contexts_can_fire(self):
        from gaia_core.cognition.affect_runtime import detect_contexts
        hits = detect_contexts(
            "Let's debug this code that handles d&d encounter logic"
        )
        assert "coding_debug" in hits
        assert "dnd_session" in hits


class TestActivateDetectedContexts:
    def test_creates_worlds_for_each_match(self, affect_with_kg):
        from gaia_core.cognition.affect_runtime import activate_detected_contexts
        activated = activate_detected_contexts(
            "Help me with this traceback from the debug session"
        )
        assert any("coding_debug" in w for w in activated)
        # World actually exists in the KG
        assert affect_with_kg.kg.get_world("ctx_coding_debug") is not None

    def test_no_kg_noops(self):
        from gaia_core.cognition import affect_runtime
        affect_runtime.reset_for_tests(None)
        orig = affect_runtime._get_affect_kg
        affect_runtime._get_affect_kg = lambda: None
        try:
            result = affect_runtime.activate_detected_contexts("/roll 1d20")
            assert result == []
        finally:
            affect_runtime._get_affect_kg = orig

    def test_idempotent(self, affect_with_kg):
        from gaia_core.cognition.affect_runtime import activate_detected_contexts
        a1 = activate_detected_contexts("/roll 1d20")
        a2 = activate_detected_contexts("/roll 1d20")
        assert a1 == a2


# ── A4: number-free "Inner weather" felt-fact for casual/social mode ──────────

def test_affect_felt_line_is_number_free_and_grammatical():
    from gaia_core.cognition.affect_runtime import affect_felt_line
    line = affect_felt_line({
        "feels": {"curious": 0.55},
        "curious_about": {"the engine work": 0.7},
        "tired_of": {"docs triage": 0.45},
    })
    assert line == "a quiet curiosity, keenly drawn toward the engine work, a little worn"
    assert not any(c.isdigit() for c in line)  # no raw stats leak into the felt fact


def test_affect_felt_line_normalizes_adjective_feels():
    """Adjective feel-words ('curious', 'frustrated') normalize to nouns so the
    'a quiet ___' article stays grammatical; nouns pass through."""
    from gaia_core.cognition.affect_runtime import affect_felt_line
    assert affect_felt_line({"feels": {"frustrated": 0.5}}) == "a quiet frustration"
    assert affect_felt_line({"feels": {"irritation": 0.7}}) == "a strong irritation"


def test_affect_felt_line_empty_when_calm():
    from gaia_core.cognition.affect_runtime import affect_felt_line
    assert affect_felt_line({}) == ""


def test_render_felt_mode_emits_inner_weather_not_stats():
    """felt=True renders the declarative 'Inner weather:' fact; felt=False keeps
    the mechanical 'Current Affect (...)' stat lines (task mode)."""
    from gaia_core.cognition import affect_runtime
    snap = {"feels": {"curious": 0.55}}
    orig = affect_runtime.current_affect_snapshot
    affect_runtime.current_affect_snapshot = lambda *a, **k: snap
    try:
        felt_lines = []
        affect_runtime.render_into_identity_lines(felt_lines, felt=True)
        assert felt_lines == ["Inner weather: a quiet curiosity."]
        assert not any(ch.isdigit() for ln in felt_lines for ch in ln)

        stat_lines = []
        affect_runtime.render_into_identity_lines(stat_lines, felt=False)
        assert any(ln.startswith("Current Affect") for ln in stat_lines)
    finally:
        affect_runtime.current_affect_snapshot = orig

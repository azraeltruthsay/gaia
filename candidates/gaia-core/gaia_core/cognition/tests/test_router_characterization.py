"""
Characterization tests for the NLU router (NeuralRouter) — GAIA_Project-21j.

These PIN THE CURRENT routing behaviour BEFORE the Nano tier is retired, so the
removal rewrite (make routing Core-direct, drop the Nano triage/tiebreak) is
verifiable against captured behaviour rather than guesswork.

IMPORTANT: the assertions marked "NANO→CORE on retire" document where NANO is
currently produced. When 21j lands, those expectations FLIP to CORE — and the
diff to this file is the precise, reviewed behaviour change. Do not "fix" them
to CORE before the router code actually changes.
"""
import pytest

from gaia_core.cognition.nlu.router import (
    NeuralRouter, TargetEngine, RouterResult, THRESHOLD_PRIME, THRESHOLD_LITE,
)
from gaia_core.config import get_config


@pytest.fixture
def router():
    # No model_pool / embed_model → nano-triage + embed stages are skipped, so
    # route() is deterministic over the heuristic/score paths.
    return NeuralRouter(get_config())


# ── Pinned constants (the routing boundaries) ───────────────────────────────

def test_thresholds_are_pinned():
    assert THRESHOLD_PRIME == 0.8
    assert THRESHOLD_LITE == 0.3
    # TargetEngine still has a NANO member today (removed by 21j).
    assert TargetEngine.NANO.value == "nano"


# ── _score_to_engine: the score→tier map (pure, static) ─────────────────────

def test_score_to_engine_high_complexity_is_prime():
    assert NeuralRouter._score_to_engine(0.95, "other", "design a distributed system") == TargetEngine.PRIME


def test_score_to_engine_high_complexity_recitation_stays_core():
    # recitation markers keep high-score requests on Core, not Prime
    assert NeuralRouter._score_to_engine(0.95, "other", "recite the poem for me") == TargetEngine.CORE


def test_score_to_engine_low_complexity_is_nano_TODAY():
    # NANO→CORE on retire: simple/low-score queries currently route to NANO
    # (which proxies to Core via the socat shim). After 21j this is CORE.
    assert NeuralRouter._score_to_engine(0.1, "greeting", "hi there") == TargetEngine.NANO


def test_score_to_engine_ambiguous_zone_is_core():
    assert NeuralRouter._score_to_engine(0.5, "other", "tell me about france") == TargetEngine.CORE


# ── _resolve_prime_target: Core-vs-Prime for complex requests (pure) ────────

def test_resolve_prime_recitation_stays_core():
    assert NeuralRouter._resolve_prime_target("please recite this poem") == TargetEngine.CORE


def test_resolve_prime_escalation_marker_goes_prime():
    assert NeuralRouter._resolve_prime_target("debug this stack trace in my code") == TargetEngine.PRIME


def test_resolve_prime_very_long_input_goes_prime():
    assert NeuralRouter._resolve_prime_target("word " * 130) == TargetEngine.PRIME


def test_resolve_prime_plain_default_is_core():
    assert NeuralRouter._resolve_prime_target("how are you today") == TargetEngine.CORE


# ── route(): end-to-end shape + determinism (no embed/pool) ─────────────────

def test_route_returns_router_result_with_valid_target(router):
    r = router.route("hello", source="api")
    assert isinstance(r, RouterResult)
    assert r.target in (TargetEngine.NANO, TargetEngine.CORE, TargetEngine.PRIME)
    assert 0.0 <= r.score <= 1.0


def test_route_complex_code_request_does_not_go_nano(router):
    # A clearly-complex request must never route to the reflex tier.
    r = router.route("write a python function with a recursive algorithm and benchmark it", source="api")
    assert r.target != TargetEngine.NANO


def test_route_is_deterministic_for_same_input(router):
    a = router.route("what time is it", source="api")
    b = router.route("what time is it", source="api")
    assert a.target == b.target and a.score == b.score


# ── NANO production surface inventory (the 21j change map) ───────────────────

def test_nano_production_surface_is_documented():
    """Inventory of where NANO is currently produced — every spot 21j must flip.

    If this list changes, the router's nano surface changed; update 21j's map.
    """
    # 1. score < THRESHOLD_LITE  -> NANO   (_score_to_engine)
    assert NeuralRouter._score_to_engine(THRESHOLD_LITE - 0.01, "x", "hi") == TargetEngine.NANO
    # 2. score >= THRESHOLD_LITE -> not NANO
    assert NeuralRouter._score_to_engine(THRESHOLD_LITE, "x", "hi") != TargetEngine.NANO


# ── Audio triage: tier decision matrix (pure, static) ───────────────────────

def _audio(duration):
    return [{"duration_seconds": duration}]


def test_audio_short_reflex_source_is_nano_TODAY():
    # NANO→CORE on retire: short audio from a reflex source currently → NANO.
    tgt, _ = NeuralRouter._triage_audio("go", "wake_word", _audio(2.0))
    assert tgt == TargetEngine.NANO


def test_audio_short_brief_text_is_nano_TODAY():
    # NANO→CORE on retire: short audio + ≤6 words → NANO.
    tgt, _ = NeuralRouter._triage_audio("turn on the light", "api", _audio(3.0))
    assert tgt == TargetEngine.NANO


def test_audio_technical_markers_go_prime():
    tgt, _ = NeuralRouter._triage_audio("debug this docker deploy pipeline", "api", _audio(30.0))
    assert tgt == TargetEngine.PRIME


def test_audio_very_long_goes_prime():
    tgt, _ = NeuralRouter._triage_audio("a long talk", "api", _audio(700.0))
    assert tgt == TargetEngine.PRIME


def test_audio_standard_is_core():
    tgt, _ = NeuralRouter._triage_audio("here is a normal voice message about my day", "api", _audio(20.0))
    assert tgt == TargetEngine.CORE

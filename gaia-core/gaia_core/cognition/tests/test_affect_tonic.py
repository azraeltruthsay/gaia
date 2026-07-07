"""
Tests for the l11 additions: note_engagement (ordinary-chat write hook) and
appraise_tonic (heartbeat baseline floors). The goal under test: the affect
organ populates during ordinary, fully-grounded conversation — not only on
failures/samvega — while floors stay raise-only and self-directed questions
never become curiosity topics.
"""

import time

import pytest

from gaia_core.cognition import affect_appraiser
from gaia_core.cognition import affect_runtime
from gaia_common.utils.knowledge_graph import KnowledgeGraph
from gaia_common.utils.affect_kg import AffectKG

# Force enable affect appraisal for tests
affect_appraiser.appraisal_enabled = lambda: True


@pytest.fixture
def mock_affect_kg(tmp_path):
    """Clean AffectKG on sqlite; resets appraiser module state around each test."""
    kg_path = tmp_path / "test_affect_tonic.sqlite"
    kg = KnowledgeGraph(db_path=str(kg_path))
    af = AffectKG(kg)
    affect_runtime.reset_for_tests(af)
    affect_appraiser._recent_outcomes.clear()
    affect_appraiser._last_turn_ts = 0.0
    yield af
    affect_runtime.reset_for_tests(None)
    affect_appraiser._recent_outcomes.clear()
    affect_appraiser._last_turn_ts = 0.0


def _curious_topics(af):
    return set((af.flatten_current_affect().get("curious_about") or {}).keys())


def test_engagement_outward_question_writes_weak_curiosity(mock_affect_kg):
    affect_appraiser.note_engagement("what is the tallest mountain in washington?")
    snap = mock_affect_kg.flatten_current_affect()
    topics = snap.get("curious_about") or {}
    # KG normalizes topic keys with underscores
    assert any("tallest_mountain" in t for t in topics), topics
    # Weak write: below knowledge_gap's 0.55 so a real gap still dominates
    val = next(v for t, v in topics.items() if "tallest_mountain" in t)
    assert 0.2 <= val <= 0.45


def test_engagement_self_directed_question_writes_nothing(mock_affect_kg):
    affect_appraiser.note_engagement("how are you today?")
    affect_appraiser.note_engagement("do you like jazz?")
    # Live regression 2026-07-07: this wrote itself as a curiosity topic and
    # the felt-line then reported her drawn toward being asked what's on her
    # mind. Any second-person reference must mark the question self-directed.
    affect_appraiser.note_engagement("Anything on your mind lately, GAIA?")
    affect_appraiser.note_engagement("what was your favorite part of today?")
    assert _curious_topics(mock_affect_kg) == set()


def test_engagement_imperative_writes_nothing(mock_affect_kg):
    affect_appraiser.note_engagement("list the files in the sandbox")
    assert _curious_topics(mock_affect_kg) == set()


def test_engagement_stamps_turn_recency(mock_affect_kg):
    before = time.time()
    affect_appraiser.note_engagement("hello there")
    assert affect_appraiser._last_turn_ts >= before


def test_engagement_type_safety():
    affect_appraiser.note_engagement(None)
    affect_appraiser.note_engagement({"not": "a string"})


def test_tonic_competence_floor_from_error_rate(mock_affect_kg):
    # 10 outcomes, half failures → floor = min(0.40, 0.55 * 0.5) = 0.275
    affect_appraiser._recent_outcomes.extend([True, False] * 5)
    affect_appraiser.appraise_tonic()
    drives = mock_affect_kg.flatten_current_affect().get("drives") or {}
    assert drives.get("competence", 0.0) >= 0.25


def test_tonic_is_raise_only(mock_affect_kg):
    mock_affect_kg.record_drive("competence", 0.80, source="test:event")
    affect_appraiser._recent_outcomes.extend([True, False] * 5)  # floor 0.275
    affect_appraiser.appraise_tonic()
    drives = mock_affect_kg.flatten_current_affect().get("drives") or {}
    assert drives.get("competence", 0.0) >= 0.75  # not dragged down to the floor


def test_tonic_silent_on_all_success(mock_affect_kg):
    affect_appraiser._recent_outcomes.extend([True] * 10)
    affect_appraiser.appraise_tonic()
    drives = mock_affect_kg.flatten_current_affect().get("drives") or {}
    assert drives.get("competence", 0.0) == 0.0


def test_tonic_novelty_from_idleness(mock_affect_kg):
    affect_appraiser._last_turn_ts = time.time() - 3 * 3600  # 3h idle → 0.15
    affect_appraiser.appraise_tonic()
    drives = mock_affect_kg.flatten_current_affect().get("drives") or {}
    assert 0.10 <= drives.get("novelty", 0.0) <= 0.35


def test_tonic_no_novelty_during_active_conversation(mock_affect_kg):
    affect_appraiser.note_engagement("hey")
    affect_appraiser.appraise_tonic()
    drives = mock_affect_kg.flatten_current_affect().get("drives") or {}
    assert drives.get("novelty", 0.0) == 0.0


def test_felt_line_renders_after_ordinary_chat(mock_affect_kg):
    """End-to-end for the l11 goal: an ordinary grounded conversation plus one
    tonic pass leaves a non-empty felt-line for the next 'how are you'."""
    affect_appraiser.note_engagement("what's new in the seattle transit expansion?")
    affect_appraiser._recent_outcomes.extend([True, False, False, True])
    affect_appraiser.appraise_tonic()
    line = affect_runtime.affect_felt_line()
    assert line, "felt-line should be non-empty after ordinary-chat appraisal"
    assert "seattle transit expansion" in line or "pull to get this right" in line

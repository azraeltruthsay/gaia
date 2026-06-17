"""
Stress tests for affect appraiser and decay math (Milestone 4 Challenger).
Tests for type-safety gaps, NaN propagation, and mathematical boundaries.
"""

import pytest
from datetime import datetime, timezone, timedelta

from gaia_core.cognition import affect_appraiser
from gaia_core.cognition import affect_runtime
from gaia_common.utils.recency import decay, age_seconds
from gaia_common.utils.knowledge_graph import KnowledgeGraph
from gaia_common.utils.affect_kg import AffectKG

# Force enable affect appraisal for tests
affect_appraiser.appraisal_enabled = lambda: True

@pytest.fixture
def mock_affect_kg(tmp_path):
    """Fixture to provide a clean AffectKG sqlite database."""
    kg_path = tmp_path / "test_stress_affect.sqlite"
    kg = KnowledgeGraph(db_path=str(kg_path))
    af = AffectKG(kg)
    affect_runtime.reset_for_tests(af)
    yield af
    affect_runtime.reset_for_tests(None)


def test_note_samvega_type_safety():
    """Test if note_samvega handles non-float weight or invalid root_cause safely without raising."""
    affect_appraiser.note_samvega(weight=None, root_cause="test")
    affect_appraiser.note_samvega(weight=[1.0], root_cause="test")


def test_note_task_outcome_type_safety():
    """Test if note_task_outcome handles non-string label safely without raising."""
    affect_appraiser.note_task_outcome(success=True, label=1234)


def test_note_knowledge_gap_type_safety():
    """Test if note_knowledge_gap handles non-string topic safely without raising."""
    affect_appraiser.note_knowledge_gap(topic={"what": "is this"})


def test_nan_propagation(mock_affect_kg):
    """Test if NaN values are clamped and do not propagate to cause affect corruption."""
    # Write NaN to curiosity
    mock_affect_kg.record_feeling("curiosity", float("nan"))
    
    # Flatten snapshot
    snap = mock_affect_kg.flatten_current_affect()
    curiosity_val = snap.get("feels", {}).get("curiosity", 0.0)
    
    # Asserting that NaN was clamped
    assert curiosity_val == 0.0, f"NaN should have been clamped to 0.0, got {curiosity_val}"


def test_age_seconds_future_time():
    """Test that future times return 0.0 age (no decay)."""
    now = datetime.now(timezone.utc)
    future = now + timedelta(hours=5)
    age = age_seconds(future, now=now)
    assert age == 0.0, f"Future time should result in 0.0 age, got {age}"


def test_decay_extreme_age():
    """Test decay with extreme age values."""
    # Extreme age should decay to 0.0 without underflow exceptions
    d = decay(age=1e20, fact_type="affect_drive")
    assert d == 0.0

    # Age 0 should yield 1.0 (no decay)
    d_zero = decay(age=0.0, fact_type="affect_drive")
    assert d_zero == 1.0


def test_decay_zero_halflife():
    """Test that a zero or negative half-life does not raise ZeroDivisionError in decay."""
    import gaia_common.utils.recency as recency
    orig_hl = recency.halflife_seconds
    recency.halflife_seconds = lambda ft: 0.0
    
    try:
        val = decay(age=10.0, fact_type="affect_drive")
        assert val in (0.0, 1.0), f"Expected 0.0 or 1.0, got {val}"
    finally:
        recency.halflife_seconds = orig_hl

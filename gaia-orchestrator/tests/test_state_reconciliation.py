"""Tests for stale handoff reconciliation on startup."""

import asyncio
import json
import pytest
from pathlib import Path
from datetime import datetime, timezone

from gaia_orchestrator.state import StateManager
from gaia_orchestrator.models.schemas import (
    HandoffStatus,
    HandoffPhase,
    HandoffType,
    GPUOwner,
)


@pytest.fixture
def state_dir(tmp_path):
    return tmp_path / "orchestrator"


@pytest.mark.asyncio
async def test_reconcile_stale_handoff(state_dir, monkeypatch):
    """A non-terminal active handoff should be marked FAILED on startup."""
    # Pre-seed state with a stale handoff stuck in RELEASING_GPU
    state_dir.mkdir(parents=True)
    state_file = state_dir / "state.json"
    stale_state = {
        "gpu": {"owner": "gaia-core", "lease_id": None, "reason": None, "acquired_at": None, "queue": []},
        "containers": {"live": {}, "candidate": {}},
        "active_handoff": {
            "handoff_id": "stale-123",
            "handoff_type": "prime_to_study",
            "phase": "releasing_gpu",
            "started_at": "2026-02-19T10:00:00+00:00",
            "completed_at": None,
            "source": "gaia-core",
            "destination": "gaia-study",
            "error": None,
            "progress_pct": 30,
        },
        "handoff_history": [],
        "last_updated": "2026-02-19T10:00:00+00:00",
    }
    state_file.write_text(json.dumps(stale_state))

    # Patch config to use our temp dir
    monkeypatch.setattr(
        "gaia_orchestrator.state.get_config",
        lambda: type("C", (), {"state_dir": state_dir, "state_file": "state.json"})(),
    )

    sm = StateManager(state_dir=state_dir)
    await sm.initialize()

    # active_handoff should now be None (reconciled)
    assert sm.state.active_handoff is None

    # Should be in history, marked FAILED
    assert len(sm.state.handoff_history) == 1
    reconciled = sm.state.handoff_history[0]
    assert reconciled.handoff_id == "stale-123"
    assert reconciled.phase == HandoffPhase.FAILED
    assert "startup reconciliation" in reconciled.error
    assert reconciled.completed_at is not None


@pytest.mark.asyncio
async def test_no_reconciliation_for_completed_handoff(state_dir, monkeypatch):
    """A completed handoff should NOT be reconciled."""
    state_dir.mkdir(parents=True)
    state_file = state_dir / "state.json"
    completed_state = {
        "gpu": {"owner": "none", "lease_id": None, "reason": None, "acquired_at": None, "queue": []},
        "containers": {"live": {}, "candidate": {}},
        "active_handoff": {
            "handoff_id": "done-456",
            "handoff_type": "study_to_prime",
            "phase": "completed",
            "started_at": "2026-02-19T10:00:00+00:00",
            "completed_at": "2026-02-19T10:05:00+00:00",
            "source": "gaia-study",
            "destination": "gaia-core",
            "error": None,
            "progress_pct": 100,
        },
        "handoff_history": [],
        "last_updated": "2026-02-19T10:05:00+00:00",
    }
    state_file.write_text(json.dumps(completed_state))

    monkeypatch.setattr(
        "gaia_orchestrator.state.get_config",
        lambda: type("C", (), {"state_dir": state_dir, "state_file": "state.json"})(),
    )

    sm = StateManager(state_dir=state_dir)
    await sm.initialize()

    # active_handoff should still be present (terminal phase, not reconciled)
    assert sm.state.active_handoff is not None
    assert sm.state.active_handoff.handoff_id == "done-456"
    assert len(sm.state.handoff_history) == 0


@pytest.mark.asyncio
async def test_no_reconciliation_when_no_handoff(state_dir, monkeypatch):
    """No active handoff means nothing to reconcile."""
    state_dir.mkdir(parents=True)
    state_file = state_dir / "state.json"
    clean_state = {
        "gpu": {"owner": "none", "lease_id": None, "reason": None, "acquired_at": None, "queue": []},
        "containers": {"live": {}, "candidate": {}},
        "active_handoff": None,
        "handoff_history": [],
        "last_updated": "2026-02-19T10:00:00+00:00",
    }
    state_file.write_text(json.dumps(clean_state))

    monkeypatch.setattr(
        "gaia_orchestrator.state.get_config",
        lambda: type("C", (), {"state_dir": state_dir, "state_file": "state.json"})(),
    )

    sm = StateManager(state_dir=state_dir)
    await sm.initialize()

    assert sm.state.active_handoff is None
    assert len(sm.state.handoff_history) == 0

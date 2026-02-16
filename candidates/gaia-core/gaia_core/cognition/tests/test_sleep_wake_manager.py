"""
Unit tests for the GAIA sleep/wake state machine.

Tests the 6-state + 2-phase lifecycle:
    ACTIVE → DROWSY → ASLEEP → DREAMING / DISTRACTED / OFFLINE
    Internal phases: _FINISHING_TASK, _WAKING
"""

import pytest
from unittest.mock import MagicMock
from pathlib import Path

from gaia_core.cognition.sleep_wake_manager import (
    GaiaState,
    SleepWakeManager,
    _TransientPhase,
    CANNED_DREAMING,
    CANNED_DISTRACTED,
)


@pytest.fixture
def mock_config(tmp_path):
    config = MagicMock()
    config.SLEEP_IDLE_THRESHOLD_MINUTES = 5
    config.SLEEP_CHECKPOINT_DIR = str(tmp_path / "sleep_state")
    config.SHARED_DIR = str(tmp_path)
    return config


@pytest.fixture
def manager(mock_config):
    return SleepWakeManager(mock_config)


# ── State initialisation ──────────────────────────────────────────────

class TestInitialState:
    def test_starts_active(self, manager):
        assert manager.get_state() == GaiaState.ACTIVE

    def test_no_pending_wake(self, manager):
        assert manager.wake_signal_pending is False

    def test_prime_not_available(self, manager):
        assert manager.prime_available is False

    def test_phase_none(self, manager):
        assert manager._phase == _TransientPhase.NONE


# ── Drowsy threshold ──────────────────────────────────────────────────

class TestDrowsyThreshold:
    def test_below_threshold(self, manager):
        assert manager.should_transition_to_drowsy(3.0) is False

    def test_at_threshold(self, manager):
        assert manager.should_transition_to_drowsy(5.0) is True

    def test_above_threshold(self, manager):
        assert manager.should_transition_to_drowsy(10.0) is True

    def test_not_when_asleep(self, manager):
        manager.state = GaiaState.ASLEEP
        assert manager.should_transition_to_drowsy(10.0) is False

    def test_not_when_dreaming(self, manager):
        manager.state = GaiaState.DREAMING
        assert manager.should_transition_to_drowsy(10.0) is False

    def test_not_when_distracted(self, manager):
        manager.state = GaiaState.DISTRACTED
        assert manager.should_transition_to_drowsy(10.0) is False


# ── ACTIVE → DROWSY → ASLEEP transition ──────────────────────────────

class TestInitiateDrowsy:
    def test_happy_path(self, manager):
        """ACTIVE → DROWSY (checkpoint) → ASLEEP."""
        result = manager.initiate_drowsy()
        assert result is True
        assert manager.get_state() == GaiaState.ASLEEP

    def test_checkpoint_written(self, manager, mock_config, tmp_path):
        """Checkpoint file should exist after initiate_drowsy."""
        manager.initiate_drowsy()
        checkpoint_dir = Path(mock_config.SHARED_DIR) / "sleep_state"
        prime_md = checkpoint_dir / "prime.md"
        assert prime_md.exists()

    def test_rejects_from_asleep(self, manager):
        manager.state = GaiaState.ASLEEP
        result = manager.initiate_drowsy()
        assert result is False
        assert manager.get_state() == GaiaState.ASLEEP

    def test_rejects_from_dreaming(self, manager):
        manager.state = GaiaState.DREAMING
        result = manager.initiate_drowsy()
        assert result is False

    def test_rejects_from_distracted(self, manager):
        manager.state = GaiaState.DISTRACTED
        result = manager.initiate_drowsy()
        assert result is False

    def test_cancels_on_wake_during_checkpoint(self, manager):
        """If a wake signal arrives during the DROWSY phase, sleep is cancelled."""
        original_create = manager.checkpoint_manager.create_checkpoint

        def inject_wake(*args, **kwargs):
            result = original_create(*args, **kwargs)
            manager.wake_signal_pending = True
            return result

        manager.checkpoint_manager.create_checkpoint = inject_wake
        result = manager.initiate_drowsy()
        assert result is False
        assert manager.get_state() == GaiaState.ACTIVE
        assert manager.wake_signal_pending is False


# ── Wake signal handling ──────────────────────────────────────────────

class TestReceiveWakeSignal:
    def test_wake_while_active(self, manager):
        manager.receive_wake_signal()
        assert manager.wake_signal_pending is False
        assert manager.get_state() == GaiaState.ACTIVE

    def test_wake_while_asleep_transitions_to_waking(self, manager):
        manager.state = GaiaState.ASLEEP
        manager.receive_wake_signal()
        assert manager.get_state() == GaiaState.ASLEEP  # state stays ASLEEP
        assert manager._phase == _TransientPhase.WAKING  # phase changes

    def test_wake_while_asleep_non_interruptible(self, manager):
        manager.state = GaiaState.ASLEEP
        manager.current_task = {"task_id": "curate_conversations", "interruptible": False}
        manager.receive_wake_signal()
        assert manager.get_state() == GaiaState.ASLEEP
        assert manager._phase == _TransientPhase.FINISHING_TASK

    def test_wake_while_drowsy_sets_flag(self, manager):
        manager.state = GaiaState.DROWSY
        manager.receive_wake_signal()
        assert manager.wake_signal_pending is True

    def test_wake_while_dreaming_defers(self, manager):
        manager.state = GaiaState.DREAMING
        manager.receive_wake_signal()
        assert manager.wake_signal_pending is True
        assert manager.get_state() == GaiaState.DREAMING  # stays in dreaming

    def test_wake_while_distracted_notes(self, manager):
        manager.state = GaiaState.DISTRACTED
        manager.receive_wake_signal()
        assert manager.wake_signal_pending is True
        assert manager.get_state() == GaiaState.DISTRACTED


# ── Waking / context restoration ─────────────────────────────────────

class TestCompleteWake:
    def test_restores_from_checkpoint(self, manager):
        """Full cycle: sleep, write checkpoint, wake, restore context."""
        manager.initiate_drowsy()
        assert manager.get_state() == GaiaState.ASLEEP

        manager.receive_wake_signal()
        assert manager._phase == _TransientPhase.WAKING

        result = manager.complete_wake()
        assert result["checkpoint_loaded"] is True
        assert "SLEEP RESTORATION CONTEXT" in result.get("context", "")
        assert manager.get_state() == GaiaState.ACTIVE
        assert manager.prime_available is True

    def test_complete_wake_no_checkpoint(self, manager, mock_config, tmp_path):
        """When no checkpoint exists, wake still completes cleanly."""
        manager.state = GaiaState.ASLEEP
        manager._phase = _TransientPhase.WAKING
        result = manager.complete_wake()
        assert result["checkpoint_loaded"] is False
        assert manager.get_state() == GaiaState.ACTIVE

    def test_complete_wake_wrong_state(self, manager):
        result = manager.complete_wake()
        assert result["checkpoint_loaded"] is False
        assert manager.get_state() == GaiaState.ACTIVE


# ── Status ────────────────────────────────────────────────────────────

class TestStatus:
    def test_status_fields(self, manager):
        status = manager.get_status()
        assert status["state"] == "active"
        assert status["phase"] == "none"
        assert "wake_signal_pending" in status
        assert "prime_available" in status
        assert "seconds_in_state" in status
        assert "last_state_change" in status

    def test_status_reflects_state_change(self, manager):
        manager.initiate_drowsy()
        status = manager.get_status()
        assert status["state"] == "asleep"

    def test_status_includes_dreaming_handoff_id(self, manager):
        manager.state = GaiaState.ASLEEP
        manager.enter_dreaming("hid-123")
        status = manager.get_status()
        assert status["state"] == "dreaming"
        assert status["dreaming_handoff_id"] == "hid-123"


# ── Checkpoint formatting ─────────────────────────────────────────────

class TestFormatCheckpoint:
    def test_empty_checkpoint(self):
        assert SleepWakeManager._format_checkpoint_as_review("") == ""
        assert SleepWakeManager._format_checkpoint_as_review(None) == ""

    def test_review_framing(self):
        result = SleepWakeManager._format_checkpoint_as_review("test content")
        assert "[SLEEP RESTORATION CONTEXT" in result
        assert "Internal Review Only" in result
        assert "Do not respond to them directly" in result
        assert "test content" in result


# ── Transition to waking (internal phase) ────────────────────────────

class TestTransitionToWaking:
    def test_from_asleep(self, manager):
        manager.state = GaiaState.ASLEEP
        manager.transition_to_waking()
        assert manager.get_state() == GaiaState.ASLEEP  # state stays ASLEEP
        assert manager._phase == _TransientPhase.WAKING

    def test_rejects_from_active(self, manager):
        manager.transition_to_waking()
        assert manager.get_state() == GaiaState.ACTIVE
        assert manager._phase == _TransientPhase.NONE


# ── Canned responses ─────────────────────────────────────────────────

class TestCannedResponses:
    def test_no_canned_when_active(self, manager):
        assert manager.get_canned_response() is None

    def test_no_canned_when_asleep(self, manager):
        manager.state = GaiaState.ASLEEP
        assert manager.get_canned_response() is None

    def test_canned_when_dreaming(self, manager):
        manager.state = GaiaState.DREAMING
        assert manager.get_canned_response() == CANNED_DREAMING

    def test_canned_when_distracted(self, manager):
        manager.state = GaiaState.DISTRACTED
        assert manager.get_canned_response() == CANNED_DISTRACTED


# ── DREAMING transitions ─────────────────────────────────────────────

class TestDreamingTransition:
    def test_enter_dreaming_from_asleep(self, manager):
        manager.state = GaiaState.ASLEEP
        ok = manager.enter_dreaming("handoff-abc")
        assert ok is True
        assert manager.get_state() == GaiaState.DREAMING
        assert manager.dreaming_handoff_id == "handoff-abc"

    def test_enter_dreaming_rejects_from_active(self, manager):
        ok = manager.enter_dreaming("handoff-xyz")
        assert ok is False
        assert manager.get_state() == GaiaState.ACTIVE

    def test_exit_dreaming_to_asleep(self, manager):
        manager.state = GaiaState.ASLEEP
        manager.enter_dreaming("handoff-abc")
        ok = manager.exit_dreaming()
        assert ok is True
        assert manager.get_state() == GaiaState.ASLEEP
        assert manager.dreaming_handoff_id is None

    def test_exit_dreaming_triggers_pending_wake(self, manager):
        manager.state = GaiaState.ASLEEP
        manager.enter_dreaming("handoff-abc")
        manager.receive_wake_signal()  # sets pending
        manager.exit_dreaming()
        assert manager.get_state() == GaiaState.ASLEEP
        assert manager._phase == _TransientPhase.WAKING

    def test_exit_dreaming_rejects_from_active(self, manager):
        ok = manager.exit_dreaming()
        assert ok is False


# ── DISTRACTED transitions ───────────────────────────────────────────

class TestDistractedTransition:
    def test_enter_distracted_from_asleep(self, manager):
        manager.state = GaiaState.ASLEEP
        ok = manager.enter_distracted()
        assert ok is True
        assert manager.get_state() == GaiaState.DISTRACTED

    def test_enter_distracted_rejects_from_active(self, manager):
        ok = manager.enter_distracted()
        assert ok is False
        assert manager.get_state() == GaiaState.ACTIVE

    def test_exit_distracted_to_asleep(self, manager):
        manager.state = GaiaState.ASLEEP
        manager.enter_distracted()
        ok = manager.exit_distracted()
        assert ok is True
        assert manager.get_state() == GaiaState.ASLEEP

    def test_exit_distracted_triggers_pending_wake(self, manager):
        manager.state = GaiaState.ASLEEP
        manager.enter_distracted()
        manager.receive_wake_signal()
        manager.exit_distracted()
        assert manager.get_state() == GaiaState.ASLEEP
        assert manager._phase == _TransientPhase.WAKING

    def test_exit_distracted_rejects_from_active(self, manager):
        ok = manager.exit_distracted()
        assert ok is False


# ── OFFLINE transition ───────────────────────────────────────────────

class TestOffline:
    def test_offline_from_active(self, manager):
        manager.initiate_offline()
        assert manager.get_state() == GaiaState.OFFLINE
        assert manager._phase == _TransientPhase.NONE

    def test_offline_from_asleep(self, manager):
        manager.state = GaiaState.ASLEEP
        manager.initiate_offline()
        assert manager.get_state() == GaiaState.OFFLINE

    def test_offline_from_dreaming(self, manager):
        manager.state = GaiaState.DREAMING
        manager.initiate_offline()
        assert manager.get_state() == GaiaState.OFFLINE

    def test_offline_from_distracted(self, manager):
        manager.state = GaiaState.DISTRACTED
        manager.initiate_offline()
        assert manager.get_state() == GaiaState.OFFLINE

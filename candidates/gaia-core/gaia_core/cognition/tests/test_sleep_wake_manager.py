"""
Unit tests for the GAIA sleep/wake state machine.

Tests the 5-state lifecycle:
    AWAKE → DROWSY → SLEEPING → FINISHING_TASK/WAKING → AWAKE
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from gaia_core.cognition.sleep_wake_manager import GaiaState, SleepWakeManager


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
    def test_starts_awake(self, manager):
        assert manager.get_state() == GaiaState.AWAKE

    def test_no_pending_wake(self, manager):
        assert manager.wake_signal_pending is False

    def test_prime_not_available(self, manager):
        assert manager.prime_available is False


# ── Drowsy threshold ──────────────────────────────────────────────────

class TestDrowsyThreshold:
    def test_below_threshold(self, manager):
        assert manager.should_transition_to_drowsy(3.0) is False

    def test_at_threshold(self, manager):
        assert manager.should_transition_to_drowsy(5.0) is True

    def test_above_threshold(self, manager):
        assert manager.should_transition_to_drowsy(10.0) is True

    def test_not_when_sleeping(self, manager):
        manager.state = GaiaState.SLEEPING
        assert manager.should_transition_to_drowsy(10.0) is False


# ── AWAKE → DROWSY → SLEEPING transition ─────────────────────────────

class TestInitiateDrowsy:
    def test_happy_path(self, manager):
        """AWAKE → DROWSY (checkpoint) → SLEEPING."""
        result = manager.initiate_drowsy()
        assert result is True
        assert manager.get_state() == GaiaState.SLEEPING

    def test_checkpoint_written(self, manager, mock_config, tmp_path):
        """Checkpoint file should exist after initiate_drowsy."""
        manager.initiate_drowsy()
        checkpoint_dir = Path(mock_config.SHARED_DIR) / "sleep_state"
        prime_md = checkpoint_dir / "prime.md"
        assert prime_md.exists()

    def test_rejects_from_sleeping(self, manager):
        manager.state = GaiaState.SLEEPING
        result = manager.initiate_drowsy()
        assert result is False
        assert manager.get_state() == GaiaState.SLEEPING

    def test_rejects_from_waking(self, manager):
        manager.state = GaiaState.WAKING
        result = manager.initiate_drowsy()
        assert result is False

    def test_cancels_on_wake_during_checkpoint(self, manager):
        """If a wake signal arrives during the DROWSY phase, sleep is cancelled."""
        # Simulate wake signal arriving between checkpoint write and state check
        original_create = manager.checkpoint_manager.create_checkpoint

        def inject_wake(*args, **kwargs):
            result = original_create(*args, **kwargs)
            manager.wake_signal_pending = True
            return result

        manager.checkpoint_manager.create_checkpoint = inject_wake
        result = manager.initiate_drowsy()
        assert result is False
        assert manager.get_state() == GaiaState.AWAKE
        assert manager.wake_signal_pending is False


# ── Wake signal handling ──────────────────────────────────────────────

class TestReceiveWakeSignal:
    def test_wake_while_awake(self, manager):
        manager.receive_wake_signal()
        # Should immediately clear
        assert manager.wake_signal_pending is False
        assert manager.get_state() == GaiaState.AWAKE

    def test_wake_while_sleeping_transitions_to_waking(self, manager):
        manager.state = GaiaState.SLEEPING
        manager.receive_wake_signal()
        assert manager.get_state() == GaiaState.WAKING

    def test_wake_while_sleeping_non_interruptible(self, manager):
        manager.state = GaiaState.SLEEPING
        manager.current_task = {"task_id": "curate_conversations", "interruptible": False}
        manager.receive_wake_signal()
        assert manager.get_state() == GaiaState.FINISHING_TASK

    def test_wake_while_drowsy_sets_flag(self, manager):
        manager.state = GaiaState.DROWSY
        manager.receive_wake_signal()
        assert manager.wake_signal_pending is True


# ── Waking / context restoration ─────────────────────────────────────

class TestCompleteWake:
    def test_restores_from_checkpoint(self, manager):
        """Full cycle: sleep, write checkpoint, wake, restore context."""
        manager.initiate_drowsy()
        assert manager.get_state() == GaiaState.SLEEPING

        manager.receive_wake_signal()
        assert manager.get_state() == GaiaState.WAKING

        result = manager.complete_wake()
        assert result["checkpoint_loaded"] is True
        assert "SLEEP RESTORATION CONTEXT" in result.get("context", "")
        assert manager.get_state() == GaiaState.AWAKE
        assert manager.prime_available is True

    def test_complete_wake_no_checkpoint(self, manager, mock_config, tmp_path):
        """When no checkpoint exists, wake still completes cleanly."""
        manager.state = GaiaState.WAKING
        result = manager.complete_wake()
        assert result["checkpoint_loaded"] is False
        assert manager.get_state() == GaiaState.AWAKE

    def test_complete_wake_wrong_state(self, manager):
        result = manager.complete_wake()
        assert result["checkpoint_loaded"] is False
        assert manager.get_state() == GaiaState.AWAKE


# ── Status ────────────────────────────────────────────────────────────

class TestStatus:
    def test_status_fields(self, manager):
        status = manager.get_status()
        assert status["state"] == "awake"
        assert "wake_signal_pending" in status
        assert "prime_available" in status
        assert "seconds_in_state" in status
        assert "last_state_change" in status

    def test_status_reflects_state_change(self, manager):
        manager.initiate_drowsy()
        status = manager.get_status()
        assert status["state"] == "sleeping"


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


# ── Transition to waking ──────────────────────────────────────────────

class TestTransitionToWaking:
    def test_from_sleeping(self, manager):
        manager.state = GaiaState.SLEEPING
        manager.transition_to_waking()
        assert manager.get_state() == GaiaState.WAKING

    def test_from_finishing_task(self, manager):
        manager.state = GaiaState.FINISHING_TASK
        manager.transition_to_waking()
        assert manager.get_state() == GaiaState.WAKING

    def test_rejects_from_awake(self, manager):
        manager.transition_to_waking()
        assert manager.get_state() == GaiaState.AWAKE

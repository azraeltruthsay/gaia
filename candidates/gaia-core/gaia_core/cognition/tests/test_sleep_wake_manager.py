"""
Unit tests for the GAIA sleep/wake state machine.

Tests the 6-state + 2-phase lifecycle:
    ACTIVE → DROWSY → ASLEEP → DREAMING / DISTRACTED / OFFLINE
    Internal phases: _FINISHING_TASK, _WAKING

Also covers:
- Checkpoint rotation order (rotate before create)
- Post-wake consumed sentinel
- LLM-generated checkpoint (Phase 2)
"""

import pytest
from unittest.mock import MagicMock, patch
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

    def test_rotation_before_create(self, manager):
        """Rotation should happen BEFORE create so backups capture previous state."""
        call_order = []
        original_rotate = manager.checkpoint_manager.rotate_checkpoints
        original_create = manager.checkpoint_manager.create_checkpoint

        def track_rotate(*a, **kw):
            call_order.append("rotate")
            return original_rotate(*a, **kw)

        def track_create(*a, **kw):
            call_order.append("create")
            return original_create(*a, **kw)

        manager.checkpoint_manager.rotate_checkpoints = track_rotate
        manager.checkpoint_manager.create_checkpoint = track_create
        manager.initiate_drowsy()
        assert call_order == ["rotate", "create"]

    def test_previous_checkpoint_preserved_in_backup(self, manager, mock_config):
        """When sleeping twice, the backup should contain the FIRST checkpoint."""
        manager.initiate_drowsy()
        checkpoint_dir = Path(mock_config.SHARED_DIR) / "sleep_state"
        first_content = (checkpoint_dir / "prime.md").read_text(encoding="utf-8")

        # Wake up
        manager.receive_wake_signal()
        manager.complete_wake()

        # Sleep again
        manager.initiate_drowsy()
        backup_content = (checkpoint_dir / "prime_previous.md").read_text(encoding="utf-8")
        assert backup_content == first_content

    def test_consumed_sentinel_cleared_on_new_checkpoint(self, manager, mock_config):
        """Creating a new checkpoint should clear the consumed sentinel."""
        checkpoint_dir = Path(mock_config.SHARED_DIR) / "sleep_state"
        sentinel = checkpoint_dir / ".prime_consumed"

        # First sleep + wake (creates consumed sentinel)
        manager.initiate_drowsy()
        manager.receive_wake_signal()
        manager.complete_wake()
        assert sentinel.exists()

        # Second sleep should clear the sentinel
        manager.initiate_drowsy()
        assert not sentinel.exists()


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

    def test_complete_wake_marks_consumed(self, manager, mock_config):
        """After wake, the consumed sentinel should exist."""
        manager.initiate_drowsy()
        manager.receive_wake_signal()
        manager.complete_wake()

        sentinel = Path(mock_config.SHARED_DIR) / "sleep_state" / ".prime_consumed"
        assert sentinel.exists()

    def test_complete_wake_no_consumed_when_no_checkpoint(self, manager, mock_config):
        """If there's no checkpoint, no consumed sentinel should be created."""
        manager.state = GaiaState.ASLEEP
        manager._phase = _TransientPhase.WAKING
        manager.complete_wake()

        sentinel = Path(mock_config.SHARED_DIR) / "sleep_state" / ".prime_consumed"
        assert not sentinel.exists()


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


# ── LLM-generated checkpoint (Phase 2) ──────────────────────────────

class TestLLMCheckpoint:
    def test_llm_checkpoint_with_model_pool(self, mock_config):
        """When model_pool provides a lite model, checkpoint uses LLM generation."""
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "I was thinking about artisanal AI..."}}]
        }
        mock_pool = MagicMock()
        mock_pool.get_model_for_role.return_value = mock_llm

        mgr = SleepWakeManager(mock_config, model_pool=mock_pool)
        mgr.initiate_drowsy()

        checkpoint_file = Path(mock_config.SHARED_DIR) / "sleep_state" / "prime.md"
        content = checkpoint_file.read_text(encoding="utf-8")

        assert "LLM introspection" in content
        assert "artisanal AI" in content
        mock_pool.get_model_for_role.assert_called_with("lite")
        mock_llm.create_chat_completion.assert_called_once()

    def test_llm_fallback_on_no_lite_model(self, mock_config):
        """Falls back to template when lite model is None."""
        mock_pool = MagicMock()
        mock_pool.get_model_for_role.return_value = None

        mgr = SleepWakeManager(mock_config, model_pool=mock_pool)
        mgr.initiate_drowsy()

        checkpoint_file = Path(mock_config.SHARED_DIR) / "sleep_state" / "prime.md"
        content = checkpoint_file.read_text(encoding="utf-8")

        assert "static template" in content

    def test_llm_fallback_on_exception(self, mock_config):
        """Falls back to template when LLM call raises."""
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.side_effect = RuntimeError("model crashed")
        mock_pool = MagicMock()
        mock_pool.get_model_for_role.return_value = mock_llm

        mgr = SleepWakeManager(mock_config, model_pool=mock_pool)
        result = mgr.initiate_drowsy()
        assert result is True  # should still enter ASLEEP

        checkpoint_file = Path(mock_config.SHARED_DIR) / "sleep_state" / "prime.md"
        content = checkpoint_file.read_text(encoding="utf-8")
        assert "static template" in content

    def test_template_without_model_pool(self, manager, mock_config):
        """Without model_pool, checkpoint uses static template."""
        manager.initiate_drowsy()

        checkpoint_file = Path(mock_config.SHARED_DIR) / "sleep_state" / "prime.md"
        content = checkpoint_file.read_text(encoding="utf-8")
        assert "static template" in content


# ── Checkpoint consumed sentinel ─────────────────────────────────────

class TestCheckpointConsumed:
    def test_is_consumed_false_initially(self, manager):
        assert manager.checkpoint_manager.is_consumed() is False

    def test_mark_consumed_creates_sentinel(self, manager, mock_config):
        manager.checkpoint_manager.mark_consumed()
        sentinel = Path(mock_config.SHARED_DIR) / "sleep_state" / ".prime_consumed"
        assert sentinel.exists()
        assert manager.checkpoint_manager.is_consumed() is True

    def test_create_checkpoint_clears_consumed(self, manager, mock_config):
        """New checkpoint creation should clear consumed flag."""
        manager.checkpoint_manager.mark_consumed()
        assert manager.checkpoint_manager.is_consumed() is True

        manager.checkpoint_manager.create_checkpoint()
        assert manager.checkpoint_manager.is_consumed() is False

"""Tests for the sleep-interference gates (GAIA_Project-r2kn, beads-planning-2l9).

  1. SleepTaskScheduler.get_next_task() must yield nothing while a wake
     signal is pending — starting a task just to interrupt it seconds
     later swallows the queued wake (execute_task clears the event).

  2. SleepWakeManager.should_transition_to_drowsy() must refuse to park
     while an engine reports in-flight inference, and must reset the
     idle clock so the recheck happens a full window later.
"""

import pytest
from unittest.mock import MagicMock

from gaia_core.cognition.sleep_task_scheduler import SleepTaskScheduler
from gaia_core.cognition.sleep_wake_manager import GaiaState, SleepWakeManager


@pytest.fixture
def mock_config(tmp_path):
    config = MagicMock()
    config.SLEEP_CYCLE = {"idle_threshold_minutes": 5}
    config.SLEEP_CHECKPOINT_DIR = str(tmp_path / "sleep_state")
    config.SHARED_DIR = str(tmp_path)
    return config


class TestSchedulerWakeGate:
    def test_no_task_while_wake_pending(self, mock_config):
        scheduler = SleepTaskScheduler(mock_config)
        assert scheduler.get_next_task() is not None
        scheduler.signal_wake()
        assert scheduler.get_next_task() is None

    def test_tasks_resume_after_wake_cleared(self, mock_config):
        scheduler = SleepTaskScheduler(mock_config)
        scheduler.signal_wake()
        assert scheduler.get_next_task() is None
        scheduler._wake_event.clear()
        assert scheduler.get_next_task() is not None


class TestDrowsyInferenceGate:
    @pytest.fixture
    def manager(self, mock_config):
        mgr = SleepWakeManager(mock_config, idle_monitor=MagicMock())
        mgr.state = GaiaState.ACTIVE
        return mgr

    def test_busy_engine_blocks_drowsy(self, manager, monkeypatch):
        monkeypatch.setattr(manager, "_inference_in_flight", lambda: True)
        assert manager.should_transition_to_drowsy(10.0) is False

    def test_busy_engine_resets_idle_clock(self, manager, monkeypatch):
        monkeypatch.setattr(manager, "_inference_in_flight", lambda: False)
        assert manager.should_transition_to_drowsy(10.0) is True
        manager.idle_monitor.mark_active.assert_not_called()

        monkeypatch.setattr(manager, "_inference_in_flight", lambda: True)
        assert manager.should_transition_to_drowsy(10.0) is False
        manager.idle_monitor.mark_active.assert_called_once()

    def test_probe_not_reached_below_threshold(self, manager, monkeypatch):
        """The engine probe is HTTP — it must only run after every cheap
        gate has passed, never on the ordinary 10s poll while active."""
        def boom():
            raise AssertionError("probe fired below idle threshold")
        monkeypatch.setattr(manager, "_inference_in_flight", boom)
        assert manager.should_transition_to_drowsy(1.0) is False

    def test_unreachable_engines_do_not_block(self, manager, monkeypatch):
        """_inference_in_flight itself: unreachable endpoints mean no
        in-flight inference (best-effort semantics, no exception leak)."""
        monkeypatch.setattr(
            SleepWakeManager, "_ENGINE_PROBE_ENDPOINTS",
            (("core", "TEST_UNUSED_ENV", "http://127.0.0.1:1"),),
        )
        assert manager._inference_in_flight() is False

"""A handled wake must not leave the scheduler's wake event set
(2026-11-18): a wake during a long sleep task left it set, get_next_task
started nothing while it was, and she slept 40 minutes running no task."""
import threading
from unittest.mock import MagicMock

from gaia_core.cognition import sleep_wake_manager as swm
from gaia_core.cognition.sleep_task_scheduler import SleepTaskScheduler
from gaia_core.cognition.sleep_wake_manager import GaiaState, SleepWakeManager


class _Scheduler:
    """The real wake-event methods, without registering the real tasks."""
    def __init__(self):
        self._wake_event = threading.Event()
    signal_wake = SleepTaskScheduler.signal_wake
    clear_wake = SleepTaskScheduler.clear_wake


def _manager(monkeypatch):
    monkeypatch.setattr(swm, "is_maintenance_active", lambda: False)
    m = SleepWakeManager.__new__(SleepWakeManager)
    m.state, m._phase = GaiaState.ACTIVE, swm._TransientPhase.NONE
    m.current_task, m.wake_signal_pending, m.idle_monitor = None, False, None
    m._timeline, m._task_scheduler = None, _Scheduler()
    m.checkpoint_manager, m._council_notes, m.model_pool = MagicMock(), MagicMock(), None
    m._emit_state_change = lambda *a, **k: None
    return m


def test_falling_asleep_clears_a_stale_wake(monkeypatch):
    m = _manager(monkeypatch)
    m._task_scheduler.signal_wake()              # a message while awake
    m.wake_signal_pending = False                # handled by the turn
    assert m.initiate_drowsy() is True
    assert not m._task_scheduler._wake_event.is_set()


def test_a_wake_still_pending_is_kept(monkeypatch):
    m = _manager(monkeypatch)
    m._task_scheduler.signal_wake()
    m.wake_signal_pending = True
    m._clear_scheduler_wake()
    assert m._task_scheduler._wake_event.is_set()


def test_waking_clears_it(monkeypatch):
    m = _manager(monkeypatch)
    m.state, m._phase = GaiaState.ASLEEP, swm._TransientPhase.WAKING
    m._task_scheduler.signal_wake()
    m._check_lmcache_status = lambda: {}
    m.complete_wake()
    assert m.state == GaiaState.ACTIVE
    assert not m._task_scheduler._wake_event.is_set()

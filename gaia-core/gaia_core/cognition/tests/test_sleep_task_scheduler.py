"""Unit tests for SleepTaskScheduler."""

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from gaia_core.cognition.sleep_task_scheduler import SleepTask, SleepTaskScheduler


class FakeConfig:
    SLEEP_ENABLED = True
    SLEEP_IDLE_THRESHOLD_MINUTES = 5
    SLEEP_CHECKPOINT_DIR = "/tmp/test_sleep"


@pytest.fixture
def config():
    return FakeConfig()


@pytest.fixture
def scheduler(config):
    """Scheduler with default tasks registered."""
    return SleepTaskScheduler(config)


@pytest.fixture
def bare_scheduler(config):
    """Scheduler with NO tasks (for testing registration)."""
    s = SleepTaskScheduler.__new__(SleepTaskScheduler)
    s.config = config
    s.model_pool = None
    s.agent_core = None
    s._tasks = []
    return s


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------


class TestRegistration:
    def test_default_tasks_registered(self, scheduler):
        assert len(scheduler._tasks) == 3

    def test_default_task_ids(self, scheduler):
        ids = {t.task_id for t in scheduler._tasks}
        assert ids == {"conversation_curation", "thought_seed_review", "initiative_cycle"}

    def test_register_custom_task(self, bare_scheduler):
        task = SleepTask(
            task_id="custom",
            task_type="custom_type",
            priority=5,
            interruptible=False,
            estimated_duration_seconds=10,
            handler=lambda: None,
        )
        bare_scheduler.register_task(task)
        assert len(bare_scheduler._tasks) == 1
        assert bare_scheduler._tasks[0].task_id == "custom"


# ------------------------------------------------------------------
# Scheduling (priority + LRU)
# ------------------------------------------------------------------


class TestScheduling:
    def test_priority_ordering(self, bare_scheduler):
        """Lower priority number = higher priority (runs first)."""
        bare_scheduler.register_task(SleepTask(
            task_id="low", task_type="l", priority=3, interruptible=True,
            estimated_duration_seconds=10, handler=lambda: None,
        ))
        bare_scheduler.register_task(SleepTask(
            task_id="high", task_type="h", priority=1, interruptible=True,
            estimated_duration_seconds=10, handler=lambda: None,
        ))
        next_task = bare_scheduler.get_next_task()
        assert next_task.task_id == "high"

    def test_lru_within_same_priority(self, bare_scheduler):
        """Among tasks of equal priority, the least-recently-run is picked."""
        old = datetime(2020, 1, 1, tzinfo=timezone.utc)
        recent = datetime(2025, 1, 1, tzinfo=timezone.utc)

        bare_scheduler.register_task(SleepTask(
            task_id="recent", task_type="r", priority=1, interruptible=True,
            estimated_duration_seconds=10, handler=lambda: None, last_run=recent,
        ))
        bare_scheduler.register_task(SleepTask(
            task_id="old", task_type="o", priority=1, interruptible=True,
            estimated_duration_seconds=10, handler=lambda: None, last_run=old,
        ))
        next_task = bare_scheduler.get_next_task()
        assert next_task.task_id == "old"

    def test_never_run_beats_recently_run(self, bare_scheduler):
        """A task that has never run (last_run=None) should be chosen first."""
        recent = datetime(2025, 1, 1, tzinfo=timezone.utc)

        bare_scheduler.register_task(SleepTask(
            task_id="ran", task_type="r", priority=1, interruptible=True,
            estimated_duration_seconds=10, handler=lambda: None, last_run=recent,
        ))
        bare_scheduler.register_task(SleepTask(
            task_id="never", task_type="n", priority=1, interruptible=True,
            estimated_duration_seconds=10, handler=lambda: None, last_run=None,
        ))
        next_task = bare_scheduler.get_next_task()
        assert next_task.task_id == "never"

    def test_empty_scheduler_returns_none(self, bare_scheduler):
        assert bare_scheduler.get_next_task() is None


# ------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------


class TestExecution:
    def test_successful_execution(self, bare_scheduler):
        handler = MagicMock()
        task = SleepTask(
            task_id="ok", task_type="t", priority=1, interruptible=True,
            estimated_duration_seconds=1, handler=handler,
        )
        bare_scheduler.register_task(task)

        result = bare_scheduler.execute_task(task)

        assert result is True
        handler.assert_called_once()
        assert task.run_count == 1
        assert task.last_run is not None
        assert task.last_error is None

    def test_failed_execution(self, bare_scheduler):
        handler = MagicMock(side_effect=RuntimeError("boom"))
        task = SleepTask(
            task_id="fail", task_type="t", priority=1, interruptible=True,
            estimated_duration_seconds=1, handler=handler,
        )
        bare_scheduler.register_task(task)

        result = bare_scheduler.execute_task(task)

        assert result is False
        assert task.last_error == "boom"
        assert task.last_run is not None

    def test_run_count_increments(self, bare_scheduler):
        handler = MagicMock()
        task = SleepTask(
            task_id="multi", task_type="t", priority=1, interruptible=True,
            estimated_duration_seconds=1, handler=handler,
        )
        bare_scheduler.register_task(task)

        bare_scheduler.execute_task(task)
        bare_scheduler.execute_task(task)
        bare_scheduler.execute_task(task)

        assert task.run_count == 3

    def test_failed_task_does_not_crash_scheduler(self, bare_scheduler):
        """A failing handler must not propagate — scheduler stays alive."""
        handler = MagicMock(side_effect=Exception("oops"))
        task = SleepTask(
            task_id="crash", task_type="t", priority=1, interruptible=True,
            estimated_duration_seconds=1, handler=handler,
        )
        bare_scheduler.register_task(task)

        # Should not raise
        result = bare_scheduler.execute_task(task)
        assert result is False


# ------------------------------------------------------------------
# Status
# ------------------------------------------------------------------


class TestStatus:
    def test_status_shape(self, scheduler):
        status = scheduler.get_status()
        assert isinstance(status, list)
        assert len(status) == 3

        for entry in status:
            assert "task_id" in entry
            assert "task_type" in entry
            assert "priority" in entry
            assert "interruptible" in entry
            assert "run_count" in entry
            assert "last_run" in entry
            assert "last_error" in entry

    def test_status_reflects_execution(self, bare_scheduler):
        handler = MagicMock()
        task = SleepTask(
            task_id="tracked", task_type="t", priority=1, interruptible=True,
            estimated_duration_seconds=1, handler=handler,
        )
        bare_scheduler.register_task(task)
        bare_scheduler.execute_task(task)

        status = bare_scheduler.get_status()
        assert status[0]["run_count"] == 1
        assert status[0]["last_run"] is not None
        assert status[0]["last_error"] is None

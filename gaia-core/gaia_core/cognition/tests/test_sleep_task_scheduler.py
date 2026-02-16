"""Unit tests for SleepTaskScheduler."""

import time
from datetime import datetime, timezone
from pathlib import Path
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
        assert len(scheduler._tasks) == 4

    def test_default_task_ids(self, scheduler):
        ids = {t.task_id for t in scheduler._tasks}
        assert ids == {"conversation_curation", "thought_seed_review", "initiative_cycle", "blueprint_validation"}

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
        assert len(status) == 4

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


# ------------------------------------------------------------------
# Blueprint Validation
# ------------------------------------------------------------------


class TestBlueprintValidation:
    def test_task_registered(self, scheduler):
        """blueprint_validation must be in the default task list."""
        ids = {t.task_id for t in scheduler._tasks}
        assert "blueprint_validation" in ids

    def test_task_priority(self, scheduler):
        task = next(t for t in scheduler._tasks if t.task_id == "blueprint_validation")
        assert task.priority == 3

    def test_extract_enums(self, tmp_path):
        """Extract enum members from a source file."""
        source = tmp_path / "states.py"
        source.write_text(
            "from enum import Enum\n\n"
            "class GaiaState(str, Enum):\n"
            "    ACTIVE = 'active'\n"
            "    DROWSY = 'drowsy'\n"
            "    ASLEEP = 'asleep'\n"
        )
        facts = SleepTaskScheduler._extract_facts(
            ["states.py"], [tmp_path],
        )
        assert "GaiaState.ACTIVE" in facts["enums"]
        assert "GaiaState.DROWSY" in facts["enums"]
        assert "GaiaState.ASLEEP" in facts["enums"]

    def test_extract_endpoints(self, tmp_path):
        """Extract router endpoints from a source file."""
        source = tmp_path / "endpoints.py"
        source.write_text(
            'from fastapi import APIRouter\n'
            'router = APIRouter(prefix="/sleep")\n\n'
            '@router.post("/wake")\n'
            'async def wake(): pass\n\n'
            '@router.get("/status")\n'
            'async def status(): pass\n'
        )
        facts = SleepTaskScheduler._extract_facts(
            ["endpoints.py"], [tmp_path],
        )
        assert "POST /wake" in facts["endpoints"]
        assert "GET /status" in facts["endpoints"]

    def test_extract_constants(self, tmp_path):
        """Extract top-level UPPER_CASE constants."""
        source = tmp_path / "consts.py"
        source.write_text(
            'CANNED_DREAMING = "I am studying"\n'
            'CANNED_DISTRACTED = "I am busy"\n'
            'some_var = 123\n'
        )
        facts = SleepTaskScheduler._extract_facts(
            ["consts.py"], [tmp_path],
        )
        assert "CANNED_DREAMING" in facts["constants"]
        assert "CANNED_DISTRACTED" in facts["constants"]
        # lowercase should NOT be extracted
        assert "some_var" not in facts["constants"]

    def test_detects_stale_enum(self, tmp_path):
        """A blueprint missing a known enum value should be flagged."""
        # Source has ACTIVE, DROWSY, ASLEEP
        source = tmp_path / "states.py"
        source.write_text(
            "from enum import Enum\n\n"
            "class GaiaState(str, Enum):\n"
            "    ACTIVE = 'active'\n"
            "    DROWSY = 'drowsy'\n"
            "    ASLEEP = 'asleep'\n"
        )
        # Blueprint only mentions ACTIVE — missing DROWSY and ASLEEP
        bp_text = "# Blueprint\nGaiaState has ACTIVE state.\n"

        facts = SleepTaskScheduler._extract_facts(["states.py"], [tmp_path])
        missing = SleepTaskScheduler._check_facts(facts, bp_text)

        assert any("DROWSY" in m for m in missing)
        assert any("ASLEEP" in m for m in missing)
        assert not any("ACTIVE" in m for m in missing)

    def test_append_update_notes(self, tmp_path):
        """_append_update_notes should add a timestamped section."""
        bp_path = tmp_path / "TEST_BP.md"
        bp_path.write_text("# Test Blueprint\n\nSome content.\n")

        SleepTaskScheduler._append_update_notes(
            bp_path,
            bp_path.read_text(),
            ["enum:GaiaState.MISSING", "constant:NEW_CONST"],
        )

        updated = bp_path.read_text()
        assert "## Recent Implementation Updates" in updated
        assert "`enum:GaiaState.MISSING`" in updated
        assert "`constant:NEW_CONST`" in updated
        assert "blueprint_validation sleep task" in updated

    def test_no_false_positives_when_current(self, tmp_path):
        """A blueprint mentioning all facts should produce no mismatches."""
        source = tmp_path / "states.py"
        source.write_text(
            "from enum import Enum\n\n"
            "class GaiaState(str, Enum):\n"
            "    ACTIVE = 'active'\n"
        )
        bp_text = "# Blueprint\nThe GaiaState enum includes ACTIVE.\n"

        facts = SleepTaskScheduler._extract_facts(["states.py"], [tmp_path])
        missing = SleepTaskScheduler._check_facts(facts, bp_text)

        assert missing == []

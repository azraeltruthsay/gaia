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
    s._timeline = None
    s._tasks = []
    return s


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------


class TestRegistration:
    def test_default_tasks_registered(self, scheduler):
        assert len(scheduler._tasks) == 5

    def test_default_task_ids(self, scheduler):
        ids = {t.task_id for t in scheduler._tasks}
        assert ids == {
            "conversation_curation", "blueprint_validation", "code_evolution",
            "code_review", "wiki_doc_regen",
        }

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
        assert len(status) == 5

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


# ------------------------------------------------------------------
# Wiki Doc Regen
# ------------------------------------------------------------------


SAMPLE_BLUEPRINT_YAML = """\
id: test-svc
version: "0.1"
role: "The Tester"
service_status: live

runtime:
  port: 9999
  base_image: python:3.11-slim
  gpu: false
  startup_cmd: "python -m test_svc"
  health_check: "curl -f http://localhost:9999"
  dockerfile: test-svc/Dockerfile

interfaces:
  - id: hello
    direction: inbound
    description: "A greeting endpoint."
    status: active
    transport:
      type: http_rest
      path: /hello
      method: GET

  - id: notify_peer
    direction: outbound
    description: "Notify a peer service."
    status: active
    transport:
      type: http_rest
      path: /notify
      method: POST
      target_service: peer-svc

dependencies:
  services:
    - id: peer-svc
      role: notification
      required: false
      fallback: null
  volumes:
    - name: data-vol
      access: rw
      purpose: "Persistent data"
      mount_path: /data
  external_apis: []

failure_modes:
  - condition: "Peer unavailable"
    response: "Notification skipped gracefully"
    severity: degraded
    auto_recovers: true

intent:
  purpose: "A test service for unit testing wiki doc regen."
  design_decisions:
    - "Keep it simple"
    - "No GPU required"
  open_questions:
    - "Is this enough?"
"""


class TestWikiDocRegen:
    def test_task_registered_with_correct_priority(self, scheduler):
        task = next(
            (t for t in scheduler._tasks if t.task_id == "wiki_doc_regen"),
            None,
        )
        assert task is not None
        assert task.priority == 5
        assert task.task_type == "DOC_GENERATION"
        assert task.interruptible is True

    def test_render_service_wiki_page_sections(self):
        """Rendered page should contain all expected section headers."""
        import yaml

        data = yaml.safe_load(SAMPLE_BLUEPRINT_YAML)
        page = SleepTaskScheduler._render_service_wiki_page("test-svc", data)

        assert "# test-svc" in page
        assert "**Role:** The Tester" in page
        assert "## Purpose" in page
        assert "## Design Decisions" in page
        assert "## Runtime" in page
        assert "## Inbound Endpoints" in page
        assert "## Outbound Connections" in page
        assert "## Service Dependencies" in page
        assert "## Volume Mounts" in page
        assert "## Failure Modes" in page
        assert "Auto-generated from" in page

    def test_render_service_wiki_page_endpoint_data(self):
        """Endpoint tables should contain actual data from the blueprint."""
        import yaml

        data = yaml.safe_load(SAMPLE_BLUEPRINT_YAML)
        page = SleepTaskScheduler._render_service_wiki_page("test-svc", data)

        # Inbound
        assert "`/hello`" in page
        assert "GET" in page
        assert "A greeting endpoint." in page

        # Outbound
        assert "notify_peer" in page
        assert "http_rest" in page

    def test_render_service_wiki_page_failure_admonitions(self):
        """Failure modes should render as MkDocs admonitions."""
        import yaml

        data = yaml.safe_load(SAMPLE_BLUEPRINT_YAML)
        page = SleepTaskScheduler._render_service_wiki_page("test-svc", data)

        assert '!!! warning "Peer unavailable"' in page
        assert "**Severity:** degraded" in page

    def test_render_wiki_index(self):
        """Index page should have a table with service links."""
        rows = [
            {"service_id": "svc-a", "role": "Role A", "port": 1111, "gpu": False, "status": "live"},
            {"service_id": "svc-b", "role": "Role B", "port": 2222, "gpu": True, "status": "candidate"},
        ]
        index = SleepTaskScheduler._render_wiki_index(rows)

        assert "# Auto-Generated Service Map" in index
        assert "[svc-a](svc-a.md)" in index
        assert "[svc-b](svc-b.md)" in index
        assert "Role A" in index
        assert "Role B" in index

    def test_index_row_from_data(self):
        """_index_row_from_data extracts the correct fields."""
        import yaml

        data = yaml.safe_load(SAMPLE_BLUEPRINT_YAML)
        row = SleepTaskScheduler._index_row_from_data("test-svc", data)

        assert row["service_id"] == "test-svc"
        assert row["role"] == "The Tester"
        assert row["port"] == 9999
        assert row["gpu"] is False
        assert row["status"] == "live"

    def test_atomic_write_no_residue(self, tmp_path):
        """_atomic_write should leave no .tmp file on success."""
        target = tmp_path / "output.md"
        SleepTaskScheduler._atomic_write(target, "hello world")

        assert target.read_text() == "hello world"
        assert not (tmp_path / "output.tmp").exists()

    def test_atomic_write_content_correct(self, tmp_path):
        """Written content should match input exactly."""
        target = tmp_path / "test.md"
        content = "# Header\n\nSome **bold** content.\n"
        SleepTaskScheduler._atomic_write(target, content)
        assert target.read_text(encoding="utf-8") == content

    def test_malformed_yaml_skipped(self, tmp_path):
        """Malformed YAML files should be skipped without crashing."""
        bp_dir = tmp_path / "blueprints"
        bp_dir.mkdir()
        out_dir = tmp_path / "wiki_auto"
        out_dir.mkdir()

        bad_file = bp_dir / "broken.yaml"
        bad_file.write_text("{{{{invalid yaml: [[[")

        # Patch paths and run handler
        scheduler = SleepTaskScheduler.__new__(SleepTaskScheduler)
        scheduler.config = FakeConfig()
        scheduler.model_pool = None
        scheduler.agent_core = None
        scheduler._timeline = None
        scheduler._tasks = []

        with patch.object(
            SleepTaskScheduler, "_BLUEPRINTS_DIR", str(bp_dir)
        ), patch.object(
            SleepTaskScheduler, "_WIKI_AUTO_DIR", str(out_dir)
        ), patch.object(
            SleepTaskScheduler, "_REGEN_MANIFEST", str(out_dir / "_manifest.json")
        ):
            # Should not raise
            scheduler._run_wiki_doc_regen()

        # No output file for the broken blueprint
        assert not (out_dir / "broken.md").exists()

    def test_full_regen_cycle(self, tmp_path):
        """End-to-end: YAML blueprint → markdown page + index."""
        import yaml

        bp_dir = tmp_path / "blueprints"
        bp_dir.mkdir()
        out_dir = tmp_path / "wiki_auto"
        out_dir.mkdir()

        # Write a valid blueprint
        (bp_dir / "test-svc.yaml").write_text(SAMPLE_BLUEPRINT_YAML)

        scheduler = SleepTaskScheduler.__new__(SleepTaskScheduler)
        scheduler.config = FakeConfig()
        scheduler.model_pool = None
        scheduler.agent_core = None
        scheduler._timeline = None
        scheduler._tasks = []

        with patch.object(
            SleepTaskScheduler, "_BLUEPRINTS_DIR", str(bp_dir)
        ), patch.object(
            SleepTaskScheduler, "_WIKI_AUTO_DIR", str(out_dir)
        ), patch.object(
            SleepTaskScheduler, "_REGEN_MANIFEST", str(out_dir / "_manifest.json")
        ):
            scheduler._run_wiki_doc_regen()

        # Service page generated
        svc_page = out_dir / "test-svc.md"
        assert svc_page.exists()
        content = svc_page.read_text()
        assert "# test-svc" in content

        # Index page generated
        index_page = out_dir / "index.md"
        assert index_page.exists()
        assert "test-svc" in index_page.read_text()

        # Manifest written
        manifest_path = out_dir / "_manifest.json"
        assert manifest_path.exists()

    def test_incremental_skip(self, tmp_path):
        """Second run with unchanged blueprints should skip regeneration."""
        import json
        import yaml

        bp_dir = tmp_path / "blueprints"
        bp_dir.mkdir()
        out_dir = tmp_path / "wiki_auto"
        out_dir.mkdir()

        bp_file = bp_dir / "test-svc.yaml"
        bp_file.write_text(SAMPLE_BLUEPRINT_YAML)

        scheduler = SleepTaskScheduler.__new__(SleepTaskScheduler)
        scheduler.config = FakeConfig()
        scheduler.model_pool = None
        scheduler.agent_core = None
        scheduler._timeline = None
        scheduler._tasks = []

        patches = [
            patch.object(SleepTaskScheduler, "_BLUEPRINTS_DIR", str(bp_dir)),
            patch.object(SleepTaskScheduler, "_WIKI_AUTO_DIR", str(out_dir)),
            patch.object(SleepTaskScheduler, "_REGEN_MANIFEST", str(out_dir / "_manifest.json")),
        ]
        for p in patches:
            p.start()

        try:
            # First run — generates files
            scheduler._run_wiki_doc_regen()
            first_content = (out_dir / "test-svc.md").read_text()

            # Second run — should skip (same mtime)
            scheduler._run_wiki_doc_regen()
            second_content = (out_dir / "test-svc.md").read_text()

            # Content unchanged (timestamp embedded, but file not rewritten)
            assert first_content == second_content
        finally:
            for p in patches:
                p.stop()

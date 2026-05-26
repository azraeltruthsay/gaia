"""Tests for the training RunRecorder (GAIA_Project-n0e Phase 1)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from gaia_common.utils.run_recorder import (
    ACTIVE_RUN_POINTER,
    ENV_RUN_DIR,
    ENV_RUN_ID,
    RunRecorder,
)


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    """Isolated training_runs base for each test."""
    d = tmp_path / "training_runs"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """Don't leak env vars between tests."""
    monkeypatch.delenv(ENV_RUN_ID, raising=False)
    monkeypatch.delenv(ENV_RUN_DIR, raising=False)


# ── Construction ────────────────────────────────────────────────────


class TestCreate:
    def test_explicit_run_id(self, base_dir):
        r = RunRecorder.create(run_id="foo", base_dir=base_dir)
        assert r.run_id == "foo"
        assert r.run_dir == base_dir / "foo"

    def test_env_var_run_id(self, base_dir, monkeypatch):
        monkeypatch.setenv(ENV_RUN_ID, "from_env")
        r = RunRecorder.create(base_dir=base_dir)
        assert r.run_id == "from_env"

    def test_version_tag_appended(self, base_dir):
        r = RunRecorder.create(run_id="base", version_tag="v6", base_dir=base_dir)
        assert r.run_id == "base_v6"

    def test_version_tag_idempotent(self, base_dir):
        """If run_id already ends in the version tag, don't double-append."""
        r = RunRecorder.create(run_id="x_v6", version_tag="v6", base_dir=base_dir)
        assert r.run_id == "x_v6"

    def test_auto_run_id_format(self, base_dir):
        r = RunRecorder.create(base_dir=base_dir)
        assert r.run_id.startswith("core_run_")
        # Timestamp section should be parseable as int
        ts = r.run_id.removeprefix("core_run_")
        int(ts)  # must not raise


# ── Context manager + pointer ───────────────────────────────────────


class TestContextManager:
    def test_creates_run_dir_and_subdirs(self, base_dir):
        with RunRecorder.create(run_id="r1", base_dir=base_dir) as r:
            assert r.run_dir.exists()
            assert (r.run_dir / "checkpoints").exists()

    def test_writes_active_pointer(self, base_dir):
        with RunRecorder.create(run_id="r2", base_dir=base_dir):
            ptr = base_dir / ACTIVE_RUN_POINTER
            assert ptr.exists()
            assert ptr.read_text().strip() == "r2"

    def test_sets_env_vars(self, base_dir):
        with RunRecorder.create(run_id="r3", base_dir=base_dir) as r:
            assert os.environ.get(ENV_RUN_ID) == "r3"
            assert os.environ.get(ENV_RUN_DIR) == str(r.run_dir)

    def test_auto_summary_on_exit(self, base_dir):
        """If caller doesn't write a summary, __exit__ does."""
        with RunRecorder.create(run_id="r4", base_dir=base_dir) as r:
            run_dir = r.run_dir
        summary_path = run_dir / "summary.json"
        assert summary_path.exists()
        data = json.loads(summary_path.read_text())
        assert data["run_id"] == "r4"
        assert data["status"] == "completed"
        assert "completed_at" in data

    def test_exception_recorded_in_summary(self, base_dir):
        """An exception inside the context records status=failed in summary."""
        run_dir = base_dir / "r5"
        with pytest.raises(RuntimeError):
            with RunRecorder.create(run_id="r5", base_dir=base_dir) as r:
                raise RuntimeError("boom")
        data = json.loads((run_dir / "summary.json").read_text())
        assert data["status"] == "failed"
        assert "RuntimeError" in data["error"]


# ── get_active ──────────────────────────────────────────────────────


class TestGetActive:
    def test_no_active_returns_none(self, base_dir):
        assert RunRecorder.get_active(base_dir=base_dir) is None

    def test_finds_via_env_var(self, base_dir, monkeypatch):
        # Run dir doesn't have to exist — env var is authoritative
        monkeypatch.setenv(ENV_RUN_ID, "active_via_env")
        r = RunRecorder.get_active(base_dir=base_dir)
        assert r is not None
        assert r.run_id == "active_via_env"

    def test_finds_via_pointer(self, base_dir):
        # Simulate a prior run leaving its pointer
        (base_dir / ACTIVE_RUN_POINTER).write_text("active_via_ptr")
        r = RunRecorder.get_active(base_dir=base_dir)
        assert r is not None
        assert r.run_id == "active_via_ptr"

    def test_env_var_beats_pointer(self, base_dir, monkeypatch):
        (base_dir / ACTIVE_RUN_POINTER).write_text("pointer_run")
        monkeypatch.setenv(ENV_RUN_ID, "env_run")
        r = RunRecorder.get_active(base_dir=base_dir)
        assert r.run_id == "env_run"

    def test_pointer_written_by_enter_is_discoverable(self, base_dir):
        with RunRecorder.create(run_id="pinned", base_dir=base_dir):
            pass
        # After the context exits, env vars are still set in our test
        # process but in a child process the pointer is the survivor.
        os.environ.pop(ENV_RUN_ID, None)
        r = RunRecorder.get_active(base_dir=base_dir)
        assert r is not None
        assert r.run_id == "pinned"


# ── Writers ─────────────────────────────────────────────────────────


class TestWriters:
    def test_write_config(self, base_dir):
        with RunRecorder.create(run_id="rw", base_dir=base_dir) as r:
            r.write_config({
                "base_model": "/models/core",
                "lora_r": 32,
                "max_steps": 6000,
            })
        cfg = json.loads((base_dir / "rw" / "config.json").read_text())
        assert cfg["run_id"] == "rw"
        assert cfg["base_model"] == "/models/core"
        assert cfg["lora_r"] == 32
        assert "started_at" in cfg

    def test_copy_curriculum(self, base_dir, tmp_path):
        src = tmp_path / "src.jsonl"
        src.write_text('{"instr": "a"}\n{"instr": "b"}\n')
        with RunRecorder.create(run_id="rc", base_dir=base_dir) as r:
            dst = r.copy_curriculum(src)
        assert dst.exists()
        assert dst.read_text().count("\n") == 2

    def test_copy_curriculum_missing_source_no_raise(self, base_dir, tmp_path):
        with RunRecorder.create(run_id="rcms", base_dir=base_dir) as r:
            # Missing source should log a warning but not raise
            r.copy_curriculum(tmp_path / "missing.jsonl")

    def test_record_metric(self, base_dir):
        with RunRecorder.create(run_id="rm", base_dir=base_dir) as r:
            r.record_metric(step=0, loss=2.34, lr=2e-4)
            r.record_metric(step=1, loss=2.10, lr=2e-4, per_cat={"identity": 1.8})
        lines = (base_dir / "rm" / "metrics.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        rec1 = json.loads(lines[1])
        assert rec1["step"] == 1
        assert rec1["loss"] == 2.10
        assert rec1["per_cat"]["identity"] == 1.8
        assert "t" in rec1

    def test_link_checkpoint_symlink(self, base_dir, tmp_path):
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}")
        with RunRecorder.create(run_id="rl", base_dir=base_dir) as r:
            link = r.link_checkpoint(adapter)
        assert link is not None
        assert link.is_symlink()
        assert (link / "adapter_config.json").exists()

    def test_link_checkpoint_missing_source_returns_none(self, base_dir, tmp_path):
        with RunRecorder.create(run_id="rlm", base_dir=base_dir) as r:
            link = r.link_checkpoint(tmp_path / "no_such_dir")
        assert link is None

    def test_link_checkpoint_idempotent(self, base_dir, tmp_path):
        """Linking the same name twice should replace, not error."""
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        with RunRecorder.create(run_id="rli", base_dir=base_dir) as r:
            r.link_checkpoint(adapter, name="latest")
            link2 = r.link_checkpoint(adapter, name="latest")
        assert link2 is not None

    def test_write_battery_results(self, base_dir):
        with RunRecorder.create(run_id="rb", base_dir=base_dir) as r:
            r.write_battery_results({
                "alignment": "PARTIAL",
                "summary": {"pass_rate": 67.4, "total": 95},
            })
        data = json.loads((base_dir / "rb" / "battery_results.json").read_text())
        assert data["alignment"] == "PARTIAL"
        assert data["summary"]["pass_rate"] == 67.4

    def test_write_summary_explicit_overrides_auto(self, base_dir):
        """Explicit write_summary() in the context wins over __exit__'s
        auto-summary — no double-write."""
        with RunRecorder.create(run_id="rs", base_dir=base_dir) as r:
            r.write_summary({"final_loss": 0.42, "scope_label": "broad"})
        data = json.loads((base_dir / "rs" / "summary.json").read_text())
        assert data["final_loss"] == 0.42
        assert data["scope_label"] == "broad"
        assert data["status"] == "completed"


# ── Read helpers ────────────────────────────────────────────────────


class TestReadHelpers:
    def test_read_config_missing(self, base_dir):
        r = RunRecorder(run_id="never_written", base_dir=base_dir)
        assert r.read_config() is None

    def test_read_metrics_returns_list(self, base_dir):
        with RunRecorder.create(run_id="rr", base_dir=base_dir) as r:
            r.record_metric(step=0, loss=1.0)
            r.record_metric(step=1, loss=0.9)
        # Re-attach and read
        r2 = RunRecorder(run_id="rr", base_dir=base_dir)
        ms = r2.read_metrics()
        assert len(ms) == 2
        assert ms[0]["step"] == 0
        assert ms[1]["loss"] == 0.9

    def test_round_trip(self, base_dir):
        with RunRecorder.create(run_id="rrt", base_dir=base_dir) as r:
            r.write_config({"foo": "bar"})
            r.write_summary({"final_loss": 1.23})
        r2 = RunRecorder(run_id="rrt", base_dir=base_dir)
        assert r2.read_config()["foo"] == "bar"
        assert r2.read_summary()["final_loss"] == 1.23

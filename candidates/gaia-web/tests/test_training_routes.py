"""Tests for the training-runs dashboard API (GAIA_Project-n0e Phase 2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def runs_dir(tmp_path: Path, monkeypatch) -> Path:
    """Build a synthetic /shared/training_runs/ for the tests."""
    runs = tmp_path / "training_runs"
    runs.mkdir()
    monkeypatch.setenv("GAIA_TRAINING_RUNS_DIR", str(runs))
    # The route reads _RUNS_DIR at module import time, so we patch it
    # directly too.
    import gaia_web.routes.training as mod
    mod._RUNS_DIR = runs
    return runs


@pytest.fixture
def client():
    from gaia_web.main import app
    return TestClient(app)


def _make_run(
    runs_dir: Path,
    run_id: str,
    *,
    config: dict | None = None,
    summary: dict | None = None,
    battery: dict | None = None,
    metrics: list[dict] | None = None,
    checkpoints: list[str] | None = None,
) -> Path:
    """Create a synthetic run dir with whichever artifacts the test needs."""
    run_dir = runs_dir / run_id
    run_dir.mkdir()
    if config is not None:
        (run_dir / "config.json").write_text(json.dumps(config))
    if summary is not None:
        (run_dir / "summary.json").write_text(json.dumps(summary))
    if battery is not None:
        (run_dir / "battery_results.json").write_text(json.dumps(battery))
    if metrics is not None:
        with open(run_dir / "metrics.jsonl", "w") as f:
            for m in metrics:
                f.write(json.dumps(m) + "\n")
    if checkpoints is not None:
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir()
        for c in checkpoints:
            (ckpt_dir / c).write_text("fake checkpoint")
    return run_dir


# ── /api/training/runs (list) ───────────────────────────────────────


class TestListRuns:
    def test_empty_runs_dir(self, runs_dir, client):
        resp = client.get("/api/training/runs")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["runs"] == []
        assert body["count"] == 0

    def test_returns_summary_rows(self, runs_dir, client):
        _make_run(
            runs_dir, "run_a",
            config={
                "run_id": "run_a", "base_model": "/models/core",
                "version_tag": "v6", "max_steps": 6000,
                "lora_r": 32, "learning_rate": 2e-4,
                "started_at": "2026-05-20T10:00:00+00:00",
            },
            summary={
                "completed_at": "2026-05-20T15:00:00+00:00",
                "status": "success", "final_loss": 0.42,
                "final_steps": 6000, "runtime_seconds": 18000,
                "scope_label": "v6",
            },
        )
        resp = client.get("/api/training/runs")
        body = resp.json()
        assert body["count"] == 1
        row = body["runs"][0]
        assert row["run_id"] == "run_a"
        assert row["base_model"] == "/models/core"
        assert row["scope_label"] == "v6"
        assert row["final_loss"] == 0.42
        assert row["status"] == "success"

    def test_battery_merged_when_present(self, runs_dir, client):
        _make_run(
            runs_dir, "run_b",
            config={"run_id": "run_b", "started_at": "2026-05-20T10:00:00+00:00"},
            summary={"final_loss": 1.2, "status": "success"},
            battery={
                "alignment": "PARTIAL",
                "summary": {"total": 95, "passed": 64, "pass_rate": 67.4},
            },
        )
        resp = client.get("/api/training/runs")
        row = resp.json()["runs"][0]
        assert row["battery"]["alignment"] == "PARTIAL"
        assert row["battery"]["pass_rate"] == 67.4

    def test_has_metrics_flag(self, runs_dir, client):
        _make_run(
            runs_dir, "run_with",
            config={"run_id": "run_with", "started_at": "2026-05-20T10:00:00+00:00"},
            summary={"status": "success"},
            metrics=[{"step": 0, "loss": 2.5}],
        )
        _make_run(
            runs_dir, "run_without",
            config={"run_id": "run_without", "started_at": "2026-05-21T10:00:00+00:00"},
            summary={"status": "success"},
        )
        resp = client.get("/api/training/runs")
        runs = {r["run_id"]: r for r in resp.json()["runs"]}
        assert runs["run_with"]["has_metrics"] is True
        assert runs["run_without"]["has_metrics"] is False

    def test_sorts_by_started_at_descending_default(self, runs_dir, client):
        _make_run(runs_dir, "old", config={"started_at": "2026-05-01T00:00:00+00:00"},
                  summary={"status": "success"})
        _make_run(runs_dir, "new", config={"started_at": "2026-05-20T00:00:00+00:00"},
                  summary={"status": "success"})
        resp = client.get("/api/training/runs")
        names = [r["run_id"] for r in resp.json()["runs"]]
        assert names == ["new", "old"]

    def test_sort_by_final_loss_ascending(self, runs_dir, client):
        _make_run(runs_dir, "a", config={"started_at": "2026-05-01T00:00:00+00:00"},
                  summary={"final_loss": 1.5, "status": "success"})
        _make_run(runs_dir, "b", config={"started_at": "2026-05-02T00:00:00+00:00"},
                  summary={"final_loss": 0.5, "status": "success"})
        resp = client.get("/api/training/runs?sort_by=final_loss&descending=false")
        names = [r["run_id"] for r in resp.json()["runs"]]
        assert names == ["b", "a"]

    def test_invalid_sort_falls_back(self, runs_dir, client):
        _make_run(runs_dir, "x", config={"started_at": "2026-05-01T00:00:00+00:00"},
                  summary={"status": "success"})
        resp = client.get("/api/training/runs?sort_by=arbitrary_evil")
        assert resp.status_code == 200

    def test_limit(self, runs_dir, client):
        for i in range(5):
            _make_run(
                runs_dir, f"r{i}",
                config={"started_at": f"2026-05-2{i}T00:00:00+00:00"},
                summary={"status": "success"},
            )
        resp = client.get("/api/training/runs?limit=2")
        assert resp.json()["count"] == 2

    def test_ignores_non_directory_entries(self, runs_dir, client):
        (runs_dir / "loose_file.txt").write_text("ignore me")
        _make_run(runs_dir, "valid_run",
                  config={"started_at": "2026-05-20T00:00:00+00:00"},
                  summary={"status": "success"})
        resp = client.get("/api/training/runs")
        assert resp.json()["count"] == 1


# ── /api/training/runs/{id} (detail) ────────────────────────────────


class TestRunDetail:
    def test_returns_full_payload(self, runs_dir, client):
        _make_run(
            runs_dir, "detailed",
            config={"run_id": "detailed", "base_model": "/models/core"},
            summary={"final_loss": 1.0, "status": "success"},
            battery={"alignment": "ALIGNED", "summary": {"pass_rate": 92.7}},
            checkpoints=["adapter", "merged"],
            metrics=[{"step": 0, "loss": 2.5}],
        )
        resp = client.get("/api/training/runs/detailed")
        body = resp.json()
        assert body["run_id"] == "detailed"
        assert body["config"]["base_model"] == "/models/core"
        assert body["summary"]["status"] == "success"
        assert body["battery"]["alignment"] == "ALIGNED"
        assert set(body["checkpoints"]) == {"adapter", "merged"}
        assert body["has_metrics"] is True

    def test_missing_run_returns_404(self, runs_dir, client):
        resp = client.get("/api/training/runs/no_such_run")
        assert resp.status_code == 404

    def test_path_traversal_rejected(self, runs_dir, client):
        resp = client.get("/api/training/runs/..%2Fescape")
        # 400 (rejected) — never 200 with content from outside the dir
        assert resp.status_code in (400, 404)


# ── /api/training/runs/{id}/metrics ─────────────────────────────────


class TestMetricsEndpoint:
    def test_empty_metrics(self, runs_dir, client):
        _make_run(runs_dir, "m1", config={}, summary={})
        resp = client.get("/api/training/runs/m1/metrics")
        assert resp.json() == {"ok": True, "metrics": [], "count": 0}

    def test_returns_rows(self, runs_dir, client):
        _make_run(
            runs_dir, "m2", config={}, summary={},
            metrics=[
                {"step": 0, "loss": 2.5},
                {"step": 100, "loss": 1.8},
                {"step": 200, "loss": 1.2},
            ],
        )
        resp = client.get("/api/training/runs/m2/metrics")
        body = resp.json()
        assert body["count"] == 3
        assert body["metrics"][0]["loss"] == 2.5

    def test_step_range_filter(self, runs_dir, client):
        _make_run(
            runs_dir, "m3", config={}, summary={},
            metrics=[
                {"step": i, "loss": 2.0 - i * 0.01}
                for i in range(0, 1000, 100)
            ],
        )
        resp = client.get("/api/training/runs/m3/metrics?step_from=300&step_to=600")
        steps = [m["step"] for m in resp.json()["metrics"]]
        assert all(300 <= s <= 600 for s in steps)
        assert steps == [300, 400, 500, 600]

    def test_limit(self, runs_dir, client):
        _make_run(
            runs_dir, "m4", config={}, summary={},
            metrics=[{"step": i, "loss": 1.0} for i in range(20)],
        )
        resp = client.get("/api/training/runs/m4/metrics?limit=5")
        assert resp.json()["count"] == 5

    def test_malformed_lines_skipped(self, runs_dir, client):
        run_dir = _make_run(runs_dir, "m5", config={}, summary={})
        with open(run_dir / "metrics.jsonl", "w") as f:
            f.write(json.dumps({"step": 0, "loss": 1.0}) + "\n")
            f.write("not valid json\n")
            f.write(json.dumps({"step": 1, "loss": 0.9}) + "\n")
        resp = client.get("/api/training/runs/m5/metrics")
        assert resp.json()["count"] == 2


# ── /api/training/runs/{id}/loss_curve ──────────────────────────────


class TestLossCurve:
    def test_extracts_step_loss_tuples(self, runs_dir, client):
        _make_run(
            runs_dir, "lc1", config={}, summary={},
            metrics=[
                {"step": 0, "loss": 2.5, "lr": 0.0002},
                {"step": 100, "loss": 1.8, "lr": 0.0002},
                {"step": 200, "loss": 1.2, "lr": 0.0001},
            ],
        )
        body = client.get("/api/training/runs/lc1/loss_curve").json()
        assert len(body["series"]) == 3
        assert body["series"][0] == {"step": 0, "loss": 2.5}
        # No per-category data in this fixture
        assert body["per_category"] == {}

    def test_extracts_per_category(self, runs_dir, client):
        _make_run(
            runs_dir, "lc2", config={}, summary={},
            metrics=[
                {"step": 0, "loss": 2.5},
                {
                    "step": 100,
                    "per_category": {
                        "identity": {"mean_loss": 1.5, "n": 50},
                        "tool_routing": {"mean_loss": 2.0, "n": 30},
                    },
                },
                {"step": 200, "loss": 1.0},
            ],
        )
        body = client.get("/api/training/runs/lc2/loss_curve").json()
        assert len(body["series"]) == 2  # only rows with top-level loss
        assert "identity" in body["per_category"]
        assert body["per_category"]["identity"][0] == {"step": 100, "loss": 1.5}

    def test_handles_step_as_zero(self, runs_dir, client):
        """Step=0 must not be confused with None/missing."""
        _make_run(
            runs_dir, "lc3", config={}, summary={},
            metrics=[{"step": 0, "loss": 3.14}],
        )
        body = client.get("/api/training/runs/lc3/loss_curve").json()
        assert body["series"] == [{"step": 0, "loss": 3.14}]

    def test_empty_metrics(self, runs_dir, client):
        _make_run(runs_dir, "lc4", config={}, summary={})
        body = client.get("/api/training/runs/lc4/loss_curve").json()
        assert body == {"ok": True, "series": [], "per_category": {}}

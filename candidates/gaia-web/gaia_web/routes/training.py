"""Training runs dashboard (GAIA_Project-n0e Phase 2).

Read-only HTTP surface over /shared/training_runs/ — the structured
records the RunRecorder writes (Phase 1, commit 39a392b). The frontend
consumes these endpoints to list past runs, surface loss curves, and
link to battery results.

Endpoints:

  GET /api/training/runs
      List all runs. Returns a flat array of summary dicts merged from
      each run's config.json + summary.json + (if present)
      battery_results.json. Lightweight — no metrics payload.

  GET /api/training/runs/{run_id}
      Full detail for one run: config + summary + battery + the
      checkpoint dir listing.

  GET /api/training/runs/{run_id}/metrics
      Streams the run's metrics.jsonl as a JSON array. May be large
      (5K-50K rows for a real run); the frontend should treat this
      as a separate fetch from the detail call.

  GET /api/training/runs/{run_id}/loss_curve
      Pre-aggregated loss series for the chart. Same data as /metrics
      but stripped to just (step, loss) tuples + per-category if
      present. Lighter payload, easier for the canvas to render.

Sourced from /shared/training_runs/<run_id>/ on disk; no DB.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

logger = logging.getLogger("GAIA.Web.Training")

router = APIRouter(prefix="/api/training", tags=["training"])


_RUNS_DIR = Path(os.getenv("GAIA_TRAINING_RUNS_DIR", "/shared/training_runs"))


# ── Helpers ─────────────────────────────────────────────────────────


def _safe_run_id(run_id: str) -> str:
    """Reject path traversal; allow only safe characters in run_id."""
    import re
    if not run_id or not re.match(r"\A[a-zA-Z0-9_.-]+\Z", run_id):
        raise HTTPException(
            status_code=400, detail=f"Invalid run_id: {run_id!r}",
        )
    return run_id


def _run_dir(run_id: str) -> Path:
    rid = _safe_run_id(run_id)
    p = _RUNS_DIR / rid
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return p


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Failed to parse %s: %s", path, e)
        return None


def _list_checkpoints(run_dir: Path) -> list[str]:
    """Names of files/symlinks under the checkpoints/ subdir."""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []
    try:
        return sorted(p.name for p in ckpt_dir.iterdir())
    except OSError:
        return []


def _build_summary_row(run_dir: Path) -> dict:
    """Merge config.json + summary.json + battery_results.json into a
    single row suitable for the runs list view. Missing fields just
    omit; the frontend should be defensive."""
    config = _read_json(run_dir / "config.json") or {}
    summary = _read_json(run_dir / "summary.json") or {}
    battery = _read_json(run_dir / "battery_results.json") or {}

    row = {
        "run_id": config.get("run_id") or summary.get("run_id") or run_dir.name,
        # Config-side
        "base_model": config.get("base_model"),
        "scope_label": config.get("version_tag") or summary.get("scope_label"),
        "max_steps": config.get("max_steps"),
        "lora_r": config.get("lora_r"),
        "learning_rate": config.get("learning_rate"),
        "started_at": config.get("started_at"),
        # Summary-side
        "completed_at": summary.get("completed_at"),
        "status": summary.get("status"),
        "final_loss": summary.get("final_loss"),
        "final_steps": summary.get("final_steps"),
        "runtime_seconds": summary.get("runtime_seconds"),
        "adapter_dir": summary.get("adapter_dir") or config.get("adapter_dir"),
        "merged_dir": summary.get("merged_dir") or config.get("merged_dir"),
        # Battery-side (compact)
        "battery": (
            {
                "alignment": battery.get("alignment"),
                "pass_rate": (battery.get("summary") or {}).get("pass_rate"),
                "total": (battery.get("summary") or {}).get("total"),
                "passed": (battery.get("summary") or {}).get("passed"),
            }
            if battery else None
        ),
        # Has metrics?
        "has_metrics": (run_dir / "metrics.jsonl").exists(),
        "checkpoint_count": len(_list_checkpoints(run_dir)),
    }
    return row


# ── Endpoints ───────────────────────────────────────────────────────


@router.get("/runs")
async def list_runs(
    sort_by: str = "started_at",
    descending: bool = True,
    limit: Optional[int] = None,
) -> dict:
    """List all training runs from /shared/training_runs/.

    Each row carries a compact summary (config + summary + battery
    pass-rate). Sorted by `sort_by` (default started_at) — supported
    columns: started_at, completed_at, final_loss, status, scope_label,
    run_id.
    """
    if not _RUNS_DIR.exists():
        return {"ok": True, "runs": [], "count": 0,
                "warning": f"runs dir absent: {_RUNS_DIR}"}

    runs: list[dict] = []
    try:
        for entry in sorted(_RUNS_DIR.iterdir()):
            if not entry.is_dir():
                continue
            try:
                runs.append(_build_summary_row(entry))
            except Exception as e:
                logger.debug("Skipping malformed run dir %s: %s", entry, e)
                continue
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))

    allowed_sorts = {
        "started_at", "completed_at", "final_loss",
        "status", "scope_label", "run_id", "final_steps",
    }
    if sort_by not in allowed_sorts:
        sort_by = "started_at"

    sentinel = object()

    def _key(r):
        v = r.get(sort_by, sentinel)
        return (v is sentinel, v if v is not sentinel else "")

    runs.sort(key=_key, reverse=descending)

    if limit is not None and limit >= 0:
        runs = runs[:limit]
    return {"ok": True, "runs": runs, "count": len(runs)}


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    """Full detail for one run: config + summary + battery + checkpoints.

    Does NOT include the metrics payload — fetch that via
    /runs/{run_id}/metrics or /runs/{run_id}/loss_curve.
    """
    run_dir = _run_dir(run_id)
    config = _read_json(run_dir / "config.json") or {}
    summary = _read_json(run_dir / "summary.json") or {}
    battery = _read_json(run_dir / "battery_results.json")
    return {
        "ok": True,
        "run_id": run_id,
        "config": config,
        "summary": summary,
        "battery": battery,
        "checkpoints": _list_checkpoints(run_dir),
        "has_metrics": (run_dir / "metrics.jsonl").exists(),
        "has_curriculum": (run_dir / "curriculum.jsonl").exists(),
    }


@router.get("/runs/{run_id}/metrics")
async def get_run_metrics(
    run_id: str,
    limit: Optional[int] = None,
    step_from: Optional[int] = None,
    step_to: Optional[int] = None,
) -> dict:
    """Return the full metrics.jsonl as a JSON array.

    Optional filters: limit (cap row count), step_from/step_to (inclusive
    step range). Large payloads — the frontend should fetch this lazily
    and consider step-range pagination for very long runs.
    """
    run_dir = _run_dir(run_id)
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return {"ok": True, "metrics": [], "count": 0}

    rows: list[dict] = []
    try:
        with open(metrics_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = rec.get("step")
                if step_from is not None and isinstance(step, (int, float)) and step < step_from:
                    continue
                if step_to is not None and isinstance(step, (int, float)) and step > step_to:
                    continue
                rows.append(rec)
                if limit is not None and limit >= 0 and len(rows) >= limit:
                    break
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"ok": True, "metrics": rows, "count": len(rows)}


@router.get("/runs/{run_id}/loss_curve")
async def get_loss_curve(run_id: str) -> dict:
    """Lightweight (step, loss) series for chart rendering.

    Also surfaces per-category losses when the RunRecorder logged them
    (compute_loss callback in train_core_multimodal.py). Output shape:

      {
        "ok": true,
        "series": [{"step": 0, "loss": 2.34}, ...],
        "per_category": {
          "identity": [{"step": 100, "loss": 1.8}, ...],
          ...
        }
      }
    """
    run_dir = _run_dir(run_id)
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return {"ok": True, "series": [], "per_category": {}}

    series: list[dict] = []
    per_cat: dict[str, list[dict]] = {}
    try:
        with open(metrics_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = rec.get("step")
                if not isinstance(step, (int, float)):
                    continue
                # Top-level loss (bulk-trainer callback writes "loss")
                loss = rec.get("loss")
                if isinstance(loss, (int, float)):
                    series.append({"step": step, "loss": loss})
                # Per-category sub-dict (compute_loss callback writes
                # per_category = {cat: {mean_loss, n}, ...})
                cat_block = rec.get("per_category")
                if isinstance(cat_block, dict):
                    for cat, stats in cat_block.items():
                        if isinstance(stats, dict):
                            mean = stats.get("mean_loss")
                            if isinstance(mean, (int, float)):
                                per_cat.setdefault(cat, []).append(
                                    {"step": step, "loss": mean}
                                )
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "ok": True,
        "series": series,
        "per_category": per_cat,
    }

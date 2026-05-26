"""Structured training-run recorder (GAIA_Project-n0e Phase 1).

Replaces ad-hoc log files with a per-run directory under
`/shared/training_runs/<run_id>/` containing:

  config.json          — base model, target_modules, lr, steps, curriculum
  curriculum.jsonl     — copy/snapshot of the text curriculum
  metrics.jsonl        — per-step loss, grad_norm, per-cat losses, lr
  checkpoints/         — symlinks to LoRA adapter directories
  battery_results.json — post-training cognitive battery (if run)
  summary.json         — final loss, runtime, status, exit code

Usage:

  with RunRecorder.create(version_tag="core2x_v6") as run:
      run.write_config({...})
      run.copy_curriculum("/path/to/text.jsonl")
      for step in range(steps):
          ...
          run.record_metric(step=step, loss=loss, lr=lr)
      run.link_checkpoint("/models/lora_adapters/foo")
      run.write_summary({"final_loss": 0.42, "runtime_s": 3600})

Downstream tools (cognitive_test_battery, dashboard, future analytics)
discover the active run via GAIA_TRAIN_RUN_ID env var or the
`/shared/training_runs/current_run.txt` pointer that __enter__ writes.

This is a reusable replacement for the ad-hoc RunRecorder code that
was inlined in scripts/train_core_multimodal.py. Same on-disk schema,
proper module home.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("GAIA.RunRecorder")


DEFAULT_BASE_DIR = Path(os.environ.get("GAIA_TRAINING_RUNS_DIR", "/shared/training_runs"))
ACTIVE_RUN_POINTER = "current_run.txt"
ENV_RUN_ID = "GAIA_TRAIN_RUN_ID"
ENV_RUN_DIR = "GAIA_TRAIN_RUN_DIR"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunRecorder:
    """Per-training-run on-disk record. Use as a context manager."""

    def __init__(
        self,
        run_id: str,
        base_dir: Optional[Path] = None,
    ) -> None:
        self.run_id = run_id
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_BASE_DIR
        self.run_dir = self.base_dir / run_id
        self._t0_monotonic: Optional[float] = None
        self._status: str = "starting"
        self._summary_written = False

    # ── Construction helpers ──────────────────────────────────────────

    @classmethod
    def create(
        cls,
        *,
        version_tag: str = "",
        run_id: Optional[str] = None,
        base_dir: Optional[Path] = None,
    ) -> "RunRecorder":
        """Create a recorder with a freshly-derived run_id.

        run_id resolution order:
          1. explicit `run_id` arg
          2. GAIA_TRAIN_RUN_ID env var
          3. f"core_run_{int(time.time())}" + optional version_tag suffix
        """
        rid = (
            run_id
            or os.environ.get(ENV_RUN_ID)
            or f"core_run_{int(time.time())}"
        )
        if version_tag and not rid.endswith(version_tag):
            rid = f"{rid}_{version_tag}"
        return cls(run_id=rid, base_dir=base_dir)

    @classmethod
    def get_active(cls, base_dir: Optional[Path] = None) -> Optional["RunRecorder"]:
        """Find the currently-active run via env var or pointer file.

        Returns None if no active run is discoverable. Used by downstream
        tools (battery, dashboard) to locate where to write/read.
        """
        base = Path(base_dir) if base_dir else DEFAULT_BASE_DIR

        # 1. Explicit env var
        rid = os.environ.get(ENV_RUN_ID)
        if rid:
            run_dir = base / rid
            if run_dir.exists():
                return cls(run_id=rid, base_dir=base)
            # Env var set but dir missing — still return a recorder so
            # the caller can write into a fresh dir if it wants.
            return cls(run_id=rid, base_dir=base)

        # 2. Pointer file written by __enter__
        ptr = base / ACTIVE_RUN_POINTER
        if ptr.exists():
            try:
                rid = ptr.read_text().strip()
                if rid:
                    return cls(run_id=rid, base_dir=base)
            except OSError:
                pass
        return None

    # ── Context manager ───────────────────────────────────────────────

    def __enter__(self) -> "RunRecorder":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        # Atomic-ish pointer: write to .tmp then rename.
        ptr = self.base_dir / ACTIVE_RUN_POINTER
        ptr_tmp = self.base_dir / (ACTIVE_RUN_POINTER + ".tmp")
        try:
            ptr_tmp.write_text(self.run_id)
            ptr_tmp.replace(ptr)
        except OSError as e:
            logger.warning("Could not write active-run pointer: %s", e)
        self._t0_monotonic = time.monotonic()
        self._status = "running"
        # Expose env for child processes (e.g. cognitive_test_battery
        # invoked from this training run's shell context).
        os.environ[ENV_RUN_ID] = self.run_id
        os.environ[ENV_RUN_DIR] = str(self.run_dir)
        logger.info("RunRecorder started: %s → %s", self.run_id, self.run_dir)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Auto-write a summary if the caller didn't.
        if not self._summary_written:
            extra: dict = {}
            if exc_type is not None:
                extra["status"] = "failed"
                extra["error"] = f"{exc_type.__name__}: {exc_val}"
            try:
                self.write_summary(extra)
            except Exception as e:
                logger.warning("Auto-summary write failed: %s", e)
        # Don't clear the pointer — the run dir persists; the next run
        # will overwrite the pointer on its own enter.

    # ── Writers ───────────────────────────────────────────────────────

    def write_config(self, config: dict) -> Path:
        """Persist run config. Adds run_id + started_at if not present."""
        merged = {
            "run_id": self.run_id,
            "started_at": _utcnow_iso(),
            **config,
        }
        path = self.run_dir / "config.json"
        with open(path, "w") as f:
            json.dump(merged, f, indent=2, default=str)
        return path

    def copy_curriculum(self, src: Path, *, name: str = "curriculum.jsonl") -> Path:
        """Snapshot the curriculum file into the run dir so future runs
        and analysis aren't broken by curriculum-source edits."""
        src = Path(src)
        if not src.exists():
            logger.warning("Curriculum source missing: %s", src)
            return self.run_dir / name
        dst = self.run_dir / name
        shutil.copy2(src, dst)
        return dst

    def record_metric(self, **fields: Any) -> None:
        """Append one metric line (JSONL) to metrics.jsonl.

        Caller passes step, loss, lr, grad_norm, per-cat losses, etc.
        Anything JSON-serializable. A timestamp is added automatically.
        """
        record = {"t": _utcnow_iso(), **fields}
        path = self.run_dir / "metrics.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def link_checkpoint(
        self,
        adapter_path: Path,
        *,
        name: Optional[str] = None,
    ) -> Optional[Path]:
        """Symlink (or copy if symlink fails) an adapter into checkpoints/.

        Returns the link path or None on failure. Name defaults to the
        adapter directory's basename.
        """
        adapter_path = Path(adapter_path)
        if not adapter_path.exists():
            logger.warning("Checkpoint source missing: %s", adapter_path)
            return None
        link_name = name or adapter_path.name
        link_path = self.run_dir / "checkpoints" / link_name
        # Idempotent: remove existing link/file if present
        if link_path.exists() or link_path.is_symlink():
            try:
                link_path.unlink()
            except OSError:
                pass
        try:
            link_path.symlink_to(adapter_path.resolve())
            return link_path
        except OSError as e:
            logger.info("symlink failed (%s) — falling back to copy", e)
            try:
                if adapter_path.is_dir():
                    shutil.copytree(adapter_path, link_path)
                else:
                    shutil.copy2(adapter_path, link_path)
                return link_path
            except OSError as e2:
                logger.warning("Checkpoint copy failed: %s", e2)
                return None

    def write_battery_results(self, results: dict) -> Path:
        """Persist a cognitive_test_battery payload into the run dir.

        Called from cognitive_test_battery when GAIA_TRAIN_RUN_ID is set,
        so the training run record links directly to its evaluation.
        """
        path = self.run_dir / "battery_results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        return path

    def write_summary(self, extra: Optional[dict] = None) -> Path:
        """Write the final run summary. Idempotent — safe to call once
        the caller has the final numbers, or rely on __exit__."""
        runtime_s = (
            time.monotonic() - self._t0_monotonic
            if self._t0_monotonic is not None else 0.0
        )
        summary = {
            "run_id": self.run_id,
            "completed_at": _utcnow_iso(),
            "status": (extra or {}).get("status", self._status if self._status != "running" else "completed"),
            "runtime_seconds": round(runtime_s, 1),
        }
        if extra:
            for k, v in extra.items():
                summary.setdefault(k, v)
                # extra wins over defaults
                summary[k] = v
        path = self.run_dir / "summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        self._summary_written = True
        self._status = summary.get("status", "completed")
        return path

    # ── Read helpers ──────────────────────────────────────────────────

    def read_config(self) -> Optional[dict]:
        p = self.run_dir / "config.json"
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def read_summary(self) -> Optional[dict]:
        p = self.run_dir / "summary.json"
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def read_metrics(self) -> list[dict]:
        p = self.run_dir / "metrics.jsonl"
        if not p.exists():
            return []
        out = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

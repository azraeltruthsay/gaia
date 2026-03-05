"""
CandidateCheckpointManager — Rollback safety net for the autonomous resilience pipeline.

Before any self-modification attempt the resilience drill calls `snapshot()` to record
the current git HEAD SHA.  If the LLM-generated fix fails health checks, `restore()`
posts to the orchestrator, which runs `git checkout <sha> -- candidates/` and restarts
the affected containers.

Git is the stable-state mechanism: the checkout is atomic and always succeeds regardless
of what the LLM wrote to those files.  This guarantee must hold *before* any forward-
looking fix logic is implemented.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime

import requests

logger = logging.getLogger("GAIA.CandidateCheckpoint")

# ── Internal health URLs (docker network, reachable from within gaia-core-candidate) ──
_HEALTH_URLS: dict[str, str] = {
    "core":         "http://gaia-core-candidate:6415/health",
    "web":          "http://gaia-web-candidate:6414/health",
    "mcp":          "http://gaia-mcp-candidate:8765/health",
    "study":        "http://gaia-study-candidate:8766/health",
    "orchestrator": "http://gaia-orchestrator-candidate:6410/health",
    "audio":        "http://gaia-audio-candidate:8080/health",
    "prime":        "http://gaia-prime-candidate:7777/health",
}

# ── Orchestrator endpoint (candidate orchestrator, on the same docker network) ──
_ORCHESTRATOR_URL = "http://gaia-orchestrator-candidate:6410"


@dataclass(frozen=True)
class CandidateSnapshot:
    """Immutable record of the candidate stack state at a point in time."""

    sha: str
    timestamp: datetime
    services: tuple[str, ...]  # which logical services are covered

    def __str__(self) -> str:
        return f"CandidateSnapshot(sha={self.sha[:8]}, services={list(self.services)}, ts={self.timestamp.isoformat()})"


class CandidateCheckpointManager:
    """
    Snapshot / health-check / restore primitives for the autonomous resilience pipeline.

    Usage::

        mgr = CandidateCheckpointManager()
        snap = mgr.snapshot(["core", "mcp"])
        try:
            apply_llm_fix(...)
            restart_containers(...)
            if not mgr.is_healthy(["core", "mcp"], timeout=30):
                mgr.restore(snap)           # guaranteed rollback
                emit_samvega(...)
        except Exception:
            mgr.restore(snap)
            raise
    """

    def __init__(self, orchestrator_url: str = _ORCHESTRATOR_URL) -> None:
        self._orchestrator_url = orchestrator_url.rstrip("/")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def snapshot(self, services: list[str]) -> CandidateSnapshot:
        """
        Record the current git HEAD SHA as the stable state for the given services.

        Calls GET /candidate/snapshot on the orchestrator, which runs
        `git rev-parse HEAD` on the host repo.
        """
        self._validate_services(services)
        try:
            resp = requests.get(
                f"{self._orchestrator_url}/candidate/snapshot",
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            snap = CandidateSnapshot(
                sha=data["sha"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                services=tuple(services),
            )
            logger.info("Snapshot taken: %s", snap)
            return snap
        except Exception as exc:
            logger.error("snapshot() failed: %s", exc)
            raise RuntimeError(f"Could not take candidate snapshot: {exc}") from exc

    def is_healthy(
        self,
        services: list[str],
        timeout: int = 30,
        poll_interval: float = 2.0,
    ) -> bool:
        """
        Poll health endpoints for the given services until all return HTTP 200
        or `timeout` seconds elapse.

        Returns True if all healthy, False on timeout.
        """
        self._validate_services(services)
        deadline = time.monotonic() + timeout
        pending = set(services)

        while time.monotonic() < deadline and pending:
            still_pending: set[str] = set()
            for svc in pending:
                url = _HEALTH_URLS.get(svc)
                if url is None:
                    logger.warning("No health URL for service '%s' — skipping", svc)
                    continue
                try:
                    r = requests.get(url, timeout=3)
                    if r.status_code == 200:
                        logger.debug("%s healthy", svc)
                    else:
                        still_pending.add(svc)
                except requests.RequestException:
                    still_pending.add(svc)
            pending = still_pending
            if pending:
                time.sleep(poll_interval)

        if pending:
            logger.warning("is_healthy timed out — still unhealthy: %s", pending)
            return False
        return True

    def restore(self, snapshot: CandidateSnapshot, health_timeout: int = 45) -> bool:
        """
        Restore all candidates/ files to `snapshot.sha` and restart affected containers.

        Posts to POST /candidate/rollback on the orchestrator, which:
          1. Runs `git checkout <sha> -- candidates/` (atomic, always succeeds)
          2. Restarts the affected containers via the Docker SDK

        Returns True if the restored stack passes health checks afterwards.
        """
        logger.warning("RESTORE initiated from %s", snapshot)
        try:
            resp = requests.post(
                f"{self._orchestrator_url}/candidate/rollback",
                json={
                    "sha": snapshot.sha,
                    "services": list(snapshot.services),
                },
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
            if result.get("errors"):
                logger.warning("Rollback completed with errors: %s", result["errors"])
            else:
                logger.info("Rollback complete: %s", result)
        except Exception as exc:
            logger.error("restore() POST failed: %s — files may be in unknown state", exc)
            return False

        # Verify the restored stack is healthy
        healthy = self.is_healthy(list(snapshot.services), timeout=health_timeout)
        if healthy:
            logger.info("Candidate stack restored and healthy ✓")
        else:
            logger.error(
                "Candidate stack restored but health check still failing — "
                "manual investigation required"
            )
        return healthy

    # ─────────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_services(services: list[str]) -> None:
        unknown = set(services) - set(_HEALTH_URLS)
        if unknown:
            raise ValueError(f"Unknown candidate services: {unknown}")

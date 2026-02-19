"""
Health Watchdog for GAIA Orchestrator — HA-aware edition.

Background asyncio task that monitors both live and candidate service health.
Tracks consecutive failures, derives an HA status, and broadcasts state changes
via NotificationManager.

HA Status States:
  active           — live + candidate healthy, failover ready
  degraded         — live healthy but candidate unhealthy (failover unavailable)
  failover_active  — live down, candidate is handling traffic
  failed           — both live and candidate down

Does NOT take remediation action — Docker handles restarts via
``restart: unless-stopped``.  Alerting is informational only.
"""

import asyncio
import logging
import os
import subprocess
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import httpx

logger = logging.getLogger("GAIA.Orchestrator.HealthWatchdog")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class HAStatus(str, Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILOVER_ACTIVE = "failover_active"
    FAILED = "failed"


# Services to monitor: name -> health URL
_LIVE_SERVICES: Dict[str, str] = {
    "gaia-core": os.environ.get("ORCHESTRATOR_CORE_URL", "http://gaia-core:6415") + "/health",
    "gaia-prime": os.environ.get("ORCHESTRATOR_PRIME_URL", "http://gaia-prime:7777") + "/health",
}

_CANDIDATE_SERVICES: Dict[str, str] = {
    "gaia-core-candidate": "http://gaia-core-candidate:6415/health",
    "gaia-mcp-candidate": "http://gaia-mcp-candidate:8765/health",
}

# Polling interval (seconds)
POLL_INTERVAL = 30

# Consecutive failures before declaring a service unhealthy
FAILURE_THRESHOLD = 2

# Maintenance flag — when set, candidate monitoring is informational only
_MAINTENANCE_FLAG = Path(os.environ.get("SHARED_DIR", "/shared")) / "ha_maintenance"

# Session sync script path (inside container, project root is /gaia/GAIA_Project)
_SYNC_SCRIPT = Path("/gaia/GAIA_Project/scripts/ha_sync.sh")


class HealthWatchdog:
    """Monitors live + candidate service health, derives HA status."""

    def __init__(self, notification_manager=None) -> None:
        self._notification_manager = notification_manager
        self._task: Optional[asyncio.Task] = None

        # Per-service tracking
        self._live_healthy: Dict[str, bool] = {}
        self._candidate_healthy: Dict[str, bool] = {}
        self._consecutive_failures: Dict[str, int] = {}

        # HA state
        self._ha_status: HAStatus = HAStatus.DEGRADED  # assume degraded until first poll
        self._candidates_enabled = False  # set True once we see candidate services

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the watchdog background task."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._poll_loop(), name="health-watchdog")
        logger.info(
            "Health watchdog started (live=%s, candidate=%s, interval=%ds)",
            list(_LIVE_SERVICES.keys()),
            list(_CANDIDATE_SERVICES.keys()),
            POLL_INTERVAL,
        )

    async def stop(self) -> None:
        """Stop the watchdog background task."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("Health watchdog stopped")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Main polling loop — runs until cancelled."""
        while True:
            # Poll live services
            for name, url in _LIVE_SERVICES.items():
                await self._poll_service(name, url, self._live_healthy)

            # Poll candidate services (best-effort — they may not be running)
            for name, url in _CANDIDATE_SERVICES.items():
                await self._poll_service(name, url, self._candidate_healthy)

            # Derive and broadcast HA status
            await self._evaluate_ha_status()

            # Session sync (live → candidate) if HA is active and not in maintenance
            if self._ha_status == HAStatus.ACTIVE and not self._is_maintenance_mode():
                await self._run_session_sync()

            await asyncio.sleep(POLL_INTERVAL)

    async def _poll_service(
        self, name: str, url: str, registry: Dict[str, bool],
    ) -> None:
        """Check a single service and update its health state."""
        is_healthy = await self._check_health(name, url)
        prev_healthy = registry.get(name)

        if is_healthy:
            self._consecutive_failures[name] = 0
        else:
            self._consecutive_failures[name] = self._consecutive_failures.get(name, 0) + 1

        # Apply failure threshold: only declare unhealthy after N consecutive failures
        effective_healthy = is_healthy or (
            self._consecutive_failures.get(name, 0) < FAILURE_THRESHOLD
        )

        if prev_healthy is not None and effective_healthy != prev_healthy:
            new_state = "healthy" if effective_healthy else "unhealthy"
            old_state = "healthy" if prev_healthy else "unhealthy"
            logger.warning("Service %s changed: %s -> %s", name, old_state, new_state)
            await self._broadcast_state_change(name, old_state, new_state)
        elif prev_healthy is None:
            state = "healthy" if effective_healthy else "unhealthy"
            logger.info("Initial health check: %s is %s", name, state)

        registry[name] = effective_healthy

    async def _check_health(self, name: str, url: str) -> bool:
        """GET /health and return True if status 200."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                return resp.status_code == 200
        except Exception as exc:
            logger.debug("Health check failed for %s: %s", name, exc)
            return False

    # ------------------------------------------------------------------
    # HA Status
    # ------------------------------------------------------------------

    async def _evaluate_ha_status(self) -> None:
        """Derive HA status from live + candidate health, broadcast on change."""
        live_core_ok = self._live_healthy.get("gaia-core", False)
        candidate_core_ok = self._candidate_healthy.get("gaia-core-candidate", False)

        # Track whether candidates are enabled (ever seen healthy)
        if candidate_core_ok:
            self._candidates_enabled = True

        # Determine new HA status
        if not self._candidates_enabled or self._is_maintenance_mode():
            # No HA if candidates never came up or maintenance mode
            new_status = HAStatus.ACTIVE if live_core_ok else HAStatus.FAILED
        elif live_core_ok and candidate_core_ok:
            new_status = HAStatus.ACTIVE
        elif live_core_ok and not candidate_core_ok:
            new_status = HAStatus.DEGRADED
        elif not live_core_ok and candidate_core_ok:
            new_status = HAStatus.FAILOVER_ACTIVE
        else:
            new_status = HAStatus.FAILED

        if new_status != self._ha_status:
            logger.warning(
                "HA status changed: %s -> %s", self._ha_status.value, new_status.value,
            )
            await self._broadcast_ha_change(self._ha_status, new_status)
            self._ha_status = new_status

    # ------------------------------------------------------------------
    # Session sync
    # ------------------------------------------------------------------

    async def _run_session_sync(self) -> None:
        """Run incremental session sync (live → candidate) in background."""
        if not _SYNC_SCRIPT.exists():
            return

        try:
            proc = await asyncio.create_subprocess_exec(
                "bash", str(_SYNC_SCRIPT), "--incremental",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode != 0:
                logger.warning(
                    "Session sync failed (rc=%d): %s",
                    proc.returncode,
                    stderr.decode().strip()[:200],
                )
            else:
                logger.debug("Session sync completed")
        except asyncio.TimeoutError:
            logger.warning("Session sync timed out (>30s)")
        except Exception:
            logger.debug("Session sync unavailable", exc_info=True)

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    async def _broadcast_state_change(
        self, service_name: str, old_state: str, new_state: str,
    ) -> None:
        """Broadcast a service health change notification."""
        if self._notification_manager is None:
            return
        try:
            from .models.schemas import Notification, NotificationType

            notification = Notification(
                notification_type=NotificationType.SERVICE_HEALTH_CHANGE,
                title=f"{service_name} {'Recovered' if new_state == 'healthy' else 'Down'}",
                message=f"{service_name} changed from {old_state} to {new_state}",
                data={
                    "service": service_name,
                    "old_state": old_state,
                    "new_state": new_state,
                },
            )
            await self._notification_manager.broadcast(notification)
        except Exception:
            logger.warning("Failed to broadcast health change notification", exc_info=True)

    async def _broadcast_ha_change(
        self, old_status: HAStatus, new_status: HAStatus,
    ) -> None:
        """Broadcast an HA status change notification."""
        if self._notification_manager is None:
            return
        try:
            from .models.schemas import Notification, NotificationType

            severity = "critical" if new_status == HAStatus.FAILED else (
                "warning" if new_status in (HAStatus.DEGRADED, HAStatus.FAILOVER_ACTIVE)
                else "info"
            )

            notification = Notification(
                notification_type=NotificationType.HA_STATUS_CHANGE,
                title=f"HA: {new_status.value}",
                message=f"HA status changed from {old_status.value} to {new_status.value}",
                data={
                    "old_status": old_status.value,
                    "new_status": new_status.value,
                    "severity": severity,
                },
            )
            await self._notification_manager.broadcast(notification)
        except Exception:
            logger.warning("Failed to broadcast HA status notification", exc_info=True)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _is_maintenance_mode() -> bool:
        """Check if HA maintenance mode is active."""
        return _MAINTENANCE_FLAG.exists()

    def get_status(self) -> Dict:
        """Return current known health + HA status."""
        return {
            "ha_status": self._ha_status.value,
            "maintenance_mode": self._is_maintenance_mode(),
            "live": {
                name: ("healthy" if healthy else "unhealthy")
                for name, healthy in self._live_healthy.items()
            },
            "candidate": {
                name: ("healthy" if healthy else "unhealthy")
                for name, healthy in self._candidate_healthy.items()
            },
            "consecutive_failures": dict(self._consecutive_failures),
        }

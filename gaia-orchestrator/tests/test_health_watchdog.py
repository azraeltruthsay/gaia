"""Tests for the HA-aware health watchdog."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from gaia_orchestrator.health_watchdog import (
    HealthWatchdog,
    HAStatus,
    FAILURE_THRESHOLD,
    _LIVE_SERVICES,
    _CANDIDATE_SERVICES,
)


@pytest.fixture
def watchdog():
    return HealthWatchdog(notification_manager=None)


@pytest.mark.asyncio
async def test_initial_ha_status_is_degraded(watchdog):
    """HA status starts as DEGRADED until first poll confirms health."""
    assert watchdog._ha_status == HAStatus.DEGRADED


@pytest.mark.asyncio
async def test_check_health_returns_true_on_200(watchdog):
    """Healthy service returns True."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get.return_value = mock_resp

    with patch("gaia_orchestrator.health_watchdog.httpx.AsyncClient", return_value=mock_client):
        result = await watchdog._check_health("test-service", "http://test:1234/health")

    assert result is True


@pytest.mark.asyncio
async def test_check_health_returns_false_on_error(watchdog):
    """Connection error returns False."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get.side_effect = httpx.ConnectError("refused")

    with patch("gaia_orchestrator.health_watchdog.httpx.AsyncClient", return_value=mock_client):
        result = await watchdog._check_health("test-service", "http://test:1234/health")

    assert result is False


@pytest.mark.asyncio
async def test_consecutive_failures_tracked(watchdog):
    """Consecutive failures are counted correctly."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get.side_effect = httpx.ConnectError("refused")

    with patch("gaia_orchestrator.health_watchdog.httpx.AsyncClient", return_value=mock_client):
        await watchdog._poll_service("test-svc", "http://test:1234/health", watchdog._live_healthy)
        assert watchdog._consecutive_failures["test-svc"] == 1

        await watchdog._poll_service("test-svc", "http://test:1234/health", watchdog._live_healthy)
        assert watchdog._consecutive_failures["test-svc"] == 2


@pytest.mark.asyncio
async def test_failure_threshold_delays_unhealthy(watchdog):
    """Service stays healthy until FAILURE_THRESHOLD consecutive failures."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get.side_effect = httpx.ConnectError("refused")

    with patch("gaia_orchestrator.health_watchdog.httpx.AsyncClient", return_value=mock_client):
        # First failure — still considered healthy (below threshold)
        await watchdog._poll_service("test-svc", "http://test:1234/health", watchdog._live_healthy)
        if FAILURE_THRESHOLD > 1:
            assert watchdog._live_healthy["test-svc"] is True

        # Exceed threshold
        for _ in range(FAILURE_THRESHOLD):
            await watchdog._poll_service("test-svc", "http://test:1234/health", watchdog._live_healthy)

        assert watchdog._live_healthy["test-svc"] is False


@pytest.mark.asyncio
async def test_success_resets_failure_counter(watchdog):
    """A successful check resets the consecutive failure counter."""
    watchdog._consecutive_failures["test-svc"] = 5

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get.return_value = mock_resp

    with patch("gaia_orchestrator.health_watchdog.httpx.AsyncClient", return_value=mock_client):
        await watchdog._poll_service("test-svc", "http://test:1234/health", watchdog._live_healthy)

    assert watchdog._consecutive_failures["test-svc"] == 0
    assert watchdog._live_healthy["test-svc"] is True


@pytest.mark.asyncio
async def test_ha_status_active_when_both_healthy(watchdog):
    """HA status is ACTIVE when both live and candidate core are healthy."""
    watchdog._live_healthy["gaia-core"] = True
    watchdog._candidate_healthy["gaia-core-candidate"] = True
    watchdog._candidates_enabled = True

    with patch.object(HealthWatchdog, "_is_maintenance_mode", return_value=False):
        await watchdog._evaluate_ha_status()

    assert watchdog._ha_status == HAStatus.ACTIVE


@pytest.mark.asyncio
async def test_ha_status_degraded_when_candidate_down(watchdog):
    """HA status is DEGRADED when live is up but candidate is down."""
    watchdog._live_healthy["gaia-core"] = True
    watchdog._candidate_healthy["gaia-core-candidate"] = False
    watchdog._candidates_enabled = True

    with patch.object(HealthWatchdog, "_is_maintenance_mode", return_value=False):
        await watchdog._evaluate_ha_status()

    assert watchdog._ha_status == HAStatus.DEGRADED


@pytest.mark.asyncio
async def test_ha_status_failover_active_when_live_down(watchdog):
    """HA status is FAILOVER_ACTIVE when live is down but candidate is up."""
    watchdog._live_healthy["gaia-core"] = False
    watchdog._candidate_healthy["gaia-core-candidate"] = True
    watchdog._candidates_enabled = True

    with patch.object(HealthWatchdog, "_is_maintenance_mode", return_value=False):
        await watchdog._evaluate_ha_status()

    assert watchdog._ha_status == HAStatus.FAILOVER_ACTIVE


@pytest.mark.asyncio
async def test_ha_status_failed_when_both_down(watchdog):
    """HA status is FAILED when both live and candidate are down."""
    watchdog._live_healthy["gaia-core"] = False
    watchdog._candidate_healthy["gaia-core-candidate"] = False
    watchdog._candidates_enabled = True

    with patch.object(HealthWatchdog, "_is_maintenance_mode", return_value=False):
        await watchdog._evaluate_ha_status()

    assert watchdog._ha_status == HAStatus.FAILED


@pytest.mark.asyncio
async def test_maintenance_mode_disables_candidate_evaluation(watchdog):
    """In maintenance mode, HA status is based on live only."""
    watchdog._live_healthy["gaia-core"] = True
    watchdog._candidate_healthy["gaia-core-candidate"] = False
    watchdog._candidates_enabled = True

    with patch.object(HealthWatchdog, "_is_maintenance_mode", return_value=True):
        await watchdog._evaluate_ha_status()

    # Should be ACTIVE (not DEGRADED) because maintenance mode ignores candidates
    assert watchdog._ha_status == HAStatus.ACTIVE


@pytest.mark.asyncio
async def test_get_status_returns_complete_info(watchdog):
    """get_status returns HA status plus per-service health."""
    watchdog._live_healthy["gaia-core"] = True
    watchdog._candidate_healthy["gaia-core-candidate"] = False
    watchdog._consecutive_failures["gaia-core-candidate"] = 3

    status = watchdog.get_status()

    assert status["ha_status"] == "degraded"
    assert status["live"]["gaia-core"] == "healthy"
    assert status["candidate"]["gaia-core-candidate"] == "unhealthy"
    assert status["consecutive_failures"]["gaia-core-candidate"] == 3


@pytest.mark.asyncio
async def test_ha_change_broadcasts_notification(watchdog):
    """HA status change triggers notification broadcast."""
    mock_nm = AsyncMock()
    watchdog._notification_manager = mock_nm

    watchdog._live_healthy["gaia-core"] = True
    watchdog._candidate_healthy["gaia-core-candidate"] = True
    watchdog._candidates_enabled = True
    watchdog._ha_status = HAStatus.DEGRADED  # Start degraded

    with patch.object(HealthWatchdog, "_is_maintenance_mode", return_value=False):
        await watchdog._evaluate_ha_status()

    # Should have broadcast the HA change
    assert mock_nm.broadcast.call_count == 1
    notification = mock_nm.broadcast.call_args[0][0]
    assert notification.data["new_status"] == "active"
    assert notification.data["old_status"] == "degraded"

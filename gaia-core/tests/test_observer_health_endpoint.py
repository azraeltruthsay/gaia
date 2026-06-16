"""Tests for the /health endpoint regarding Observer health."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import gaia_core.main as main_mod


@pytest.fixture
def client():
    """Create a test client."""
    yield TestClient(main_mod.app, raise_server_exceptions=False)


def test_health_endpoint_healthy_observer(client):
    """If the observer is healthy, status should be healthy (assuming inference_ok is True)."""
    with (
        patch("httpx.AsyncClient") as mock_async_client_cls,
        patch("gaia_core.utils.stream_observer.observer_health") as mock_obs_health
    ):
        # Mock inference_ok to be True
        mock_client = MagicMock()
        
        # Async mock for GET request
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        
        async def mock_get(*args, **kwargs):
            return mock_resp
            
        mock_client.get = mock_get
        
        # Async context manager mock
        async def __aenter__(self):
            return mock_client
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
        mock_async_client_cls.return_value.__aenter__ = __aenter__
        mock_async_client_cls.return_value.__aexit__ = __aexit__
        
        # Mock observer health to be healthy
        mock_obs_health.return_value = {
            "observations": 10,
            "failures": 1,
            "fail_rate": 0.1,
            "healthy": True
        }
        
        # Make sure boot time is older than grace period so we test actual status
        with patch("time.monotonic", return_value=main_mod._core_boot_time + 200):
            resp = client.get("/health")
            
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["observer"]["healthy"] is True


def test_health_endpoint_unhealthy_observer(client):
    """If the observer is unhealthy, status should be degraded."""
    with (
        patch("httpx.AsyncClient") as mock_async_client_cls,
        patch("gaia_core.utils.stream_observer.observer_health") as mock_obs_health
    ):
        # Mock inference_ok to be True
        mock_client = MagicMock()
        
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        
        async def mock_get(*args, **kwargs):
            return mock_resp
            
        mock_client.get = mock_get
        
        async def __aenter__(self):
            return mock_client
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
        mock_async_client_cls.return_value.__aenter__ = __aenter__
        mock_async_client_cls.return_value.__aexit__ = __aexit__
        
        # Mock observer health to be unhealthy
        mock_obs_health.return_value = {
            "observations": 10,
            "failures": 6,
            "fail_rate": 0.6,
            "healthy": False
        }
        
        with patch("time.monotonic", return_value=main_mod._core_boot_time + 200):
            resp = client.get("/health")
            
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert "observer" in data["inference_detail"].lower()
        assert data["observer"]["healthy"] is False

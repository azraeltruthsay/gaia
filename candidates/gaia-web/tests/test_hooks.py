"""Tests for hooks proxy endpoints."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from gaia_web.main import app

client = TestClient(app)


def _mock_response(status_code: int = 200, json_data: dict | None = None):
    """Create a mock httpx.Response."""
    resp = httpx.Response(
        status_code=status_code,
        json=json_data or {},
        request=httpx.Request("GET", "http://test"),
    )
    return resp


class TestSleepEndpoints:
    @patch("gaia_web.routes.hooks.httpx.AsyncClient")
    def test_sleep_status_ok(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_response(200, {"state": "active", "cycle_count": 5})
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        resp = client.get("/api/hooks/sleep/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "active"

    @patch("gaia_web.routes.hooks.httpx.AsyncClient")
    def test_sleep_wake_ok(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(200, {"ok": True, "state": "waking"})
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        resp = client.post("/api/hooks/sleep/wake")
        assert resp.status_code == 200

    @patch("gaia_web.routes.hooks.httpx.AsyncClient")
    def test_sleep_shutdown_ok(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(200, {"ok": True})
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        resp = client.post("/api/hooks/sleep/shutdown")
        assert resp.status_code == 200


class TestGpuEndpoints:
    @patch("gaia_web.routes.hooks.httpx.AsyncClient")
    def test_gpu_status_ok(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_response(200, {"owner": "prime", "vram_used_mb": 4200})
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        resp = client.get("/api/hooks/gpu/status")
        assert resp.status_code == 200

    @patch("gaia_web.routes.hooks.httpx.AsyncClient")
    def test_gpu_release_ok(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(200, {"ok": True})
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        resp = client.post("/api/hooks/gpu/release")
        assert resp.status_code == 200

    @patch("gaia_web.routes.hooks.httpx.AsyncClient")
    def test_gpu_reclaim_ok(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(200, {"ok": True})
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        resp = client.post("/api/hooks/gpu/reclaim")
        assert resp.status_code == 200


class TestCodexSearch:
    @patch("gaia_web.routes.hooks.httpx.AsyncClient")
    def test_codex_search_ok(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(200, {"results": [{"title": "test", "score": 0.95}]})
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        resp = client.post("/api/hooks/codex/search", json={"query": "test query", "top_k": 5})
        assert resp.status_code == 200

    def test_codex_search_missing_query(self):
        resp = client.post("/api/hooks/codex/search", json={"top_k": 5})
        assert resp.status_code == 422  # Pydantic validation error


class TestConnectionErrors:
    @patch("gaia_web.routes.hooks.httpx.AsyncClient")
    def test_sleep_status_unreachable(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        resp = client.get("/api/hooks/sleep/status")
        assert resp.status_code == 503
        assert "unreachable" in resp.json()["error"]

    @patch("gaia_web.routes.hooks.httpx.AsyncClient")
    def test_gpu_status_unreachable(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        resp = client.get("/api/hooks/gpu/status")
        assert resp.status_code == 503
        assert "unreachable" in resp.json()["error"]

"""Tests for the world_merges proxy (GAIA_Project-21h Phase 2)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from gaia_web.main import app
    return TestClient(app)


def _mock_response(status_code: int, json_body: dict | None = None,
                   text: str = ""):
    """Build a stand-in httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    if json_body is not None:
        resp.json = MagicMock(return_value=json_body)
    else:
        resp.json = MagicMock(side_effect=ValueError("no body"))
    resp.text = text
    return resp


class _AsyncClientCtx:
    """Async-context-manager stand-in returning a MagicMock client."""
    def __init__(self, mock_client):
        self._client = mock_client

    async def __aenter__(self):
        return self._client

    async def __aexit__(self, *_):
        return False


def _patch_httpx_client(client_mock):
    """Patch httpx.AsyncClient(...) to return our mock client."""
    return patch.object(httpx, "AsyncClient", return_value=_AsyncClientCtx(client_mock))


# ── GET /api/world_merges/pending ──────────────────────────────────


class TestPending:
    def test_returns_study_payload(self, client):
        study = MagicMock()
        study.get = AsyncMock(return_value=_mock_response(
            200, {"ok": True, "pending": [{"merge_id": "m1"}], "count": 1},
        ))
        with _patch_httpx_client(study):
            resp = client.get("/api/world_merges/pending")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 1
        assert body["pending"][0]["merge_id"] == "m1"

    def test_502_on_transport_error(self, client):
        study = MagicMock()
        study.get = AsyncMock(side_effect=httpx.ConnectError("conn refused"))
        with _patch_httpx_client(study):
            resp = client.get("/api/world_merges/pending")
        assert resp.status_code == 502

    def test_502_on_upstream_5xx(self, client):
        study = MagicMock()
        study.get = AsyncMock(return_value=_mock_response(500, text="kaboom"))
        with _patch_httpx_client(study):
            resp = client.get("/api/world_merges/pending")
        assert resp.status_code == 502


# ── GET /api/world_merges/{id} ─────────────────────────────────────


class TestDetail:
    def test_passes_through(self, client):
        study = MagicMock()
        study.get = AsyncMock(return_value=_mock_response(
            200, {"ok": True, "merge_id": "m1", "status": "pending"},
        ))
        with _patch_httpx_client(study):
            resp = client.get("/api/world_merges/m1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["merge_id"] == "m1"

    def test_404_passed_through(self, client):
        study = MagicMock()
        study.get = AsyncMock(return_value=_mock_response(
            404, {"detail": "not found"},
        ))
        with _patch_httpx_client(study):
            resp = client.get("/api/world_merges/m_nope")
        assert resp.status_code == 404


# ── POST /api/world_merges/{id}/approve ────────────────────────────


class TestApprove:
    def test_forwards_payload(self, client):
        study = MagicMock()
        study.post = AsyncMock(return_value=_mock_response(
            200, {"ok": True, "status": "approved", "approved_by": "azrael"},
        ))
        with _patch_httpx_client(study) as patched:
            resp = client.post(
                "/api/world_merges/m1/approve",
                json={"approver": "azrael"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "approved"
        assert body["approved_by"] == "azrael"
        # Verify the proxy forwarded the approver
        call = study.post.call_args
        # call.kwargs["json"] is the payload we sent upstream
        assert call.kwargs.get("json") == {"approver": "azrael"}

    def test_default_approver_when_missing(self, client):
        study = MagicMock()
        study.post = AsyncMock(return_value=_mock_response(
            200, {"ok": True, "status": "approved"},
        ))
        with _patch_httpx_client(study):
            resp = client.post("/api/world_merges/m1/approve", json={})
        assert resp.status_code == 200
        call = study.post.call_args
        assert call.kwargs.get("json") == {"approver": "architect"}

    def test_409_passes_through(self, client):
        """When study returns 409 (not pending), proxy forwards it."""
        study = MagicMock()
        study.post = AsyncMock(return_value=_mock_response(
            409, {"detail": "Merge m1 is not pending"},
        ))
        with _patch_httpx_client(study):
            resp = client.post(
                "/api/world_merges/m1/approve",
                json={"approver": "azrael"},
            )
        assert resp.status_code == 409

    def test_502_on_transport_error(self, client):
        study = MagicMock()
        study.post = AsyncMock(side_effect=httpx.TimeoutException("slow"))
        with _patch_httpx_client(study):
            resp = client.post("/api/world_merges/m1/approve", json={})
        assert resp.status_code == 502


# ── POST /api/world_merges/{id}/reject ─────────────────────────────


class TestReject:
    def test_forwards_reason(self, client):
        study = MagicMock()
        study.post = AsyncMock(return_value=_mock_response(
            200, {"ok": True, "status": "rejected", "rejected_reason": "coref off"},
        ))
        with _patch_httpx_client(study):
            resp = client.post(
                "/api/world_merges/m1/reject",
                json={"reason": "coref off"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "rejected"
        call = study.post.call_args
        assert call.kwargs.get("json") == {"reason": "coref off"}

    def test_default_reason_empty(self, client):
        study = MagicMock()
        study.post = AsyncMock(return_value=_mock_response(
            200, {"ok": True, "status": "rejected"},
        ))
        with _patch_httpx_client(study):
            resp = client.post("/api/world_merges/m1/reject", json={})
        call = study.post.call_args
        assert call.kwargs.get("json") == {"reason": ""}

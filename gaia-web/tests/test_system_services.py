"""Tests for GET /api/system/services (gaia-doctor duplicate-id fix).

Found live (2026-11-13) while verifying an unrelated frontend change: the
Mission Control dashboard's Alpine x-for over `services` was throwing
"Cannot read properties of undefined (reading 'after')" every poll cycle.
Root cause -- gaia-doctor now self-reports in its own /status services{}
dict, and system_services() unconditionally appended a second "gaia-doctor"
entry on top of that, giving the frontend two list items with the same
:key="svc.id" and breaking Alpine's DOM reconciliation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from gaia_web.main import app
    return TestClient(app)


def _mock_response(status_code: int, json_body: dict):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_body)
    return resp


class _AsyncClientCtx:
    def __init__(self, mock_client):
        self._client = mock_client

    async def __aenter__(self):
        return self._client

    async def __aexit__(self, *_):
        return False


def _patch_httpx_client(client_mock):
    return patch.object(httpx, "AsyncClient", return_value=_AsyncClientCtx(client_mock))


def test_doctor_self_reporting_does_not_duplicate_the_entry(client):
    doctor = MagicMock()
    doctor.get = AsyncMock(return_value=_mock_response(200, {
        "services": {
            "gaia-core": {"healthy": True},
            "gaia-doctor": {"healthy": True},
        },
    }))
    with _patch_httpx_client(doctor):
        resp = client.get("/api/system/services")
    assert resp.status_code == 200
    services = resp.json()
    ids = [s["id"] for s in services]
    assert ids.count("gaia-doctor") == 1, f"duplicate gaia-doctor entry: {ids}"


def test_doctor_not_yet_self_reporting_still_gets_an_entry(client):
    doctor = MagicMock()
    doctor.get = AsyncMock(return_value=_mock_response(200, {
        "services": {"gaia-core": {"healthy": True}},
    }))
    with _patch_httpx_client(doctor):
        resp = client.get("/api/system/services")
    assert resp.status_code == 200
    ids = [s["id"] for s in resp.json()]
    assert ids.count("gaia-doctor") == 1

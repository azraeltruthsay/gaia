"""Tests for gaia-audio FastAPI endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client with mocked engines."""
    # We need to mock the heavy dependencies before importing main
    with patch("gaia_audio.main.AudioConfig") as MockConfig:
        mock_cfg = MagicMock()
        mock_cfg.stt_model = "tiny.en"
        mock_cfg.tts_engine = "system"
        mock_cfg.tts_voice = None
        mock_cfg.vram_budget_mb = 5600
        mock_cfg.core_endpoint = "http://gaia-core:6415"
        mock_cfg.web_endpoint = "http://gaia-web:6414"
        mock_cfg.orchestrator_endpoint = "http://gaia-orchestrator:6410"
        MockConfig.from_constants.return_value = mock_cfg

        from gaia_audio.main import app
        with TestClient(app) as c:
            yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "gaia-audio"


def test_status(client):
    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "state" in data
    assert "gpu_mode" in data
    assert "events" in data


def test_config(client):
    resp = client.get("/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "stt_model" in data
    assert "tts_engine" in data


def test_voices(client):
    resp = client.get("/voices")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


def test_mute_unmute(client):
    resp = client.post("/mute")
    assert resp.status_code == 200
    assert resp.json()["status"] == "muted"

    # Status should reflect muted state
    status = client.get("/status").json()
    assert status["muted"] is True

    resp = client.post("/unmute")
    assert resp.status_code == 200
    assert resp.json()["status"] == "unmuted"


def test_transcribe_requires_audio(client):
    resp = client.post("/transcribe", json={"audio_base64": None})
    assert resp.status_code == 400


def test_transcribe_rejects_when_muted(client):
    client.post("/mute")
    resp = client.post("/transcribe", json={"audio_base64": "aGVsbG8="})
    assert resp.status_code == 423
    client.post("/unmute")

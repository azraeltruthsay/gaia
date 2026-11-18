"""Tests for gaia-audio FastAPI endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """Test client: no models loaded, service auth in dev mode (qvo6: the
    startup loaded real models from MagicMock paths, and the HMAC
    middleware answered 401 to every non-public route)."""
    from gaia_common.utils import service_auth
    monkeypatch.setattr(service_auth, "_load_secret", lambda: None)
    with patch("gaia_audio.main.GPUManager.startup", AsyncMock()), \
         patch("gaia_audio.main.RefinerEngine.load", MagicMock()), \
         patch("gaia_audio.main.MusicEngine.load", MagicMock()), \
         patch("gaia_audio.main.AudioConfig") as MockConfig:
        mock_cfg = MagicMock()
        mock_cfg.listener_model_path = "/nonexistent/asr"
        mock_cfg.listener_device = "cpu"
        mock_cfg.nano_speaker_model_path = "/nonexistent/nano"
        mock_cfg.prime_speaker_model_path = "/nonexistent/prime"
        mock_cfg.voice_ref_audio = None
        mock_cfg.voice_ref_text = None
        mock_cfg.prime_speaker_timeout = 30
        mock_cfg.vram_budget_mb = 5600
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
    assert "listener_model" in data
    assert "nano_speaker_model" in data


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

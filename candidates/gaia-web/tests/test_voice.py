"""Tests for voice auto-answer — whitelist, API routes, and VoiceManager."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# VoiceWhitelist tests
# ---------------------------------------------------------------------------


class TestVoiceWhitelist:
    def _make_whitelist(self, tmp_path):
        from gaia_web.voice_manager import VoiceWhitelist
        return VoiceWhitelist(data_dir=str(tmp_path))

    def test_add_and_check(self, tmp_path):
        wl = self._make_whitelist(tmp_path)
        assert not wl.is_whitelisted("123")
        wl.add("123")
        assert wl.is_whitelisted("123")

    def test_remove(self, tmp_path):
        wl = self._make_whitelist(tmp_path)
        wl.add("123")
        wl.remove("123")
        assert not wl.is_whitelisted("123")

    def test_double_add_is_idempotent(self, tmp_path):
        wl = self._make_whitelist(tmp_path)
        wl.add("123")
        wl.add("123")
        assert wl.get_whitelisted() == ["123"]

    def test_remove_nonexistent_is_noop(self, tmp_path):
        wl = self._make_whitelist(tmp_path)
        wl.remove("999")  # Should not raise

    def test_persistence(self, tmp_path):
        wl = self._make_whitelist(tmp_path)
        wl.add("456")
        wl.record_seen("456", "TestUser", "guild1")

        # Load fresh instance from same path
        from gaia_web.voice_manager import VoiceWhitelist
        wl2 = VoiceWhitelist(data_dir=str(tmp_path))
        assert wl2.is_whitelisted("456")
        users = wl2.get_seen_users()
        assert len(users) == 1
        assert users[0]["name"] == "TestUser"
        assert users[0]["whitelisted"] is True

    def test_record_seen(self, tmp_path):
        wl = self._make_whitelist(tmp_path)
        wl.record_seen("111", "Alice", "guild_a")
        wl.record_seen("222", "Bob", "guild_b")
        users = wl.get_seen_users()
        assert len(users) == 2
        names = [u["name"] for u in users]
        assert "Alice" in names
        assert "Bob" in names

    def test_seen_users_sorted_by_name(self, tmp_path):
        wl = self._make_whitelist(tmp_path)
        wl.record_seen("1", "Zara")
        wl.record_seen("2", "Alice")
        wl.record_seen("3", "Mike")
        users = wl.get_seen_users()
        assert [u["name"] for u in users] == ["Alice", "Mike", "Zara"]

    def test_get_whitelisted(self, tmp_path):
        wl = self._make_whitelist(tmp_path)
        wl.add("100")
        wl.add("200")
        assert set(wl.get_whitelisted()) == {"100", "200"}


# ---------------------------------------------------------------------------
# SimpleVAD tests
# ---------------------------------------------------------------------------


class TestSimpleVAD:
    def test_no_speech_returns_none(self):
        from gaia_web.voice_manager import SimpleVAD
        vad = SimpleVAD(silence_threshold_ms=200, min_speech_ms=100)
        # Feed silence (all zeros)
        frame = b"\x00" * 640  # 20ms at 16kHz mono = 320 samples * 2 bytes
        result = vad.feed_frame(frame)
        assert result is None

    def test_speech_then_silence_flushes(self):
        from gaia_web.voice_manager import SimpleVAD
        vad = SimpleVAD(silence_threshold_ms=100, min_speech_ms=40, max_utterance_seconds=10)

        # Generate "speech" frames (high energy)
        import struct
        speech_frame = struct.pack("<320h", *([10000] * 320))  # 20ms frame, loud
        silence_frame = b"\x00" * 640

        # Feed speech
        for _ in range(5):  # 100ms of speech
            result = vad.feed_frame(speech_frame)
            assert result is None  # Still accumulating

        # Feed silence to trigger flush
        for i in range(10):  # 200ms of silence (> threshold of 100ms)
            result = vad.feed_frame(silence_frame)
            if result is not None:
                assert len(result) > 0
                return

        # Should have flushed by now
        assert False, "VAD did not flush after silence"

    def test_max_utterance_cap(self):
        from gaia_web.voice_manager import SimpleVAD
        vad = SimpleVAD(max_utterance_seconds=1)  # 1 second max

        import struct
        speech_frame = struct.pack("<320h", *([10000] * 320))

        result = None
        for _ in range(60):  # 1.2 seconds of speech
            result = vad.feed_frame(speech_frame)
            if result is not None:
                break

        assert result is not None, "VAD did not cap at max utterance length"

    def test_reset_clears_state(self):
        from gaia_web.voice_manager import SimpleVAD
        vad = SimpleVAD()
        import struct
        speech_frame = struct.pack("<320h", *([10000] * 320))
        vad.feed_frame(speech_frame)
        vad.feed_frame(speech_frame)
        vad.reset()
        assert len(vad._buffer) == 0
        assert vad._speech_frames == 0


# ---------------------------------------------------------------------------
# Audio conversion tests
# ---------------------------------------------------------------------------


class TestAudioConversion:
    def test_pcm_to_wav_base64(self):
        from gaia_web.voice_manager import pcm_to_wav_base64
        import base64
        import wave
        import io

        pcm = b"\x00" * 3200  # 100ms of 16kHz mono silence
        result = pcm_to_wav_base64(pcm, sample_rate=16000)
        assert isinstance(result, str)

        # Verify it's valid WAV
        wav_bytes = base64.b64decode(result)
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000


# ---------------------------------------------------------------------------
# Voice API routes tests
# ---------------------------------------------------------------------------


@pytest.fixture
def voice_client(tmp_path):
    """Create a test client with mocked VoiceManager."""
    from gaia_web.voice_manager import VoiceWhitelist, VoiceManager

    whitelist = VoiceWhitelist(data_dir=str(tmp_path))
    whitelist.record_seen("100", "Alice", "guild1")
    whitelist.record_seen("200", "Bob", "guild1")
    whitelist.add("100")

    vm = VoiceManager(
        core_endpoint="http://gaia-core:6415",
        audio_endpoint="http://gaia-audio:8080",
        whitelist=whitelist,
    )

    # Patch the main app to avoid importing everything
    from gaia_web.routes.voice import router
    from fastapi import FastAPI

    test_app = FastAPI()
    test_app.include_router(router)
    test_app.state.voice_manager = vm

    with TestClient(test_app) as client:
        yield client


def test_list_users(voice_client):
    resp = voice_client.get("/api/voice/users")
    assert resp.status_code == 200
    users = resp.json()
    assert len(users) == 2
    alice = next(u for u in users if u["name"] == "Alice")
    assert alice["whitelisted"] is True
    bob = next(u for u in users if u["name"] == "Bob")
    assert bob["whitelisted"] is False


def test_list_whitelisted(voice_client):
    resp = voice_client.get("/api/voice/whitelist")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["user_id"] == "100"


def test_add_to_whitelist(voice_client):
    resp = voice_client.post("/api/voice/whitelist", json={"user_id": "200"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    # Verify
    resp = voice_client.get("/api/voice/whitelist")
    ids = [u["user_id"] for u in resp.json()]
    assert "200" in ids


def test_remove_from_whitelist(voice_client):
    resp = voice_client.delete("/api/voice/whitelist/100")
    assert resp.status_code == 200

    resp = voice_client.get("/api/voice/whitelist")
    assert len(resp.json()) == 0


def test_voice_status(voice_client):
    resp = voice_client.get("/api/voice/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["connected"] is False
    assert data["state"] == "disconnected"


def test_force_disconnect(voice_client):
    resp = voice_client.post("/api/voice/disconnect")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

"""Tests for GAIA MCP Audio Listener tools."""

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

import gaia_mcp.listener_tools as lt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def use_tmp_paths(tmp_path):
    """Point control/status files to a temp directory."""
    lt._CONTROL_FILE = tmp_path / "listener_control.json"
    lt._STATUS_FILE = tmp_path / "listener_status.json"
    yield
    # Restore defaults (doesn't matter much since module reloads won't happen)


# ---------------------------------------------------------------------------
# audio_listen_start
# ---------------------------------------------------------------------------

class TestAudioListenStart:

    def test_writes_control_file(self, tmp_path):
        result = lt.audio_listen_start({"mode": "passive"})
        assert result["ok"] is True
        assert result["command"] == "start"
        assert result["mode"] == "passive"

        control = json.loads(lt._CONTROL_FILE.read_text())
        assert control["command"] == "start"
        assert control["mode"] == "passive"
        assert "issued_at" in control

    def test_active_mode(self, tmp_path):
        result = lt.audio_listen_start({"mode": "active"})
        assert result["ok"] is True
        assert result["mode"] == "active"

    def test_default_mode_is_passive(self, tmp_path):
        result = lt.audio_listen_start({})
        assert result["mode"] == "passive"

    def test_invalid_mode_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid mode"):
            lt.audio_listen_start({"mode": "turbo"})


# ---------------------------------------------------------------------------
# audio_listen_stop
# ---------------------------------------------------------------------------

class TestAudioListenStop:

    def test_writes_stop_command(self, tmp_path):
        result = lt.audio_listen_stop({})
        assert result["ok"] is True
        assert result["command"] == "stop"

        control = json.loads(lt._CONTROL_FILE.read_text())
        assert control["command"] == "stop"


# ---------------------------------------------------------------------------
# audio_listen_status
# ---------------------------------------------------------------------------

class TestAudioListenStatus:

    def test_no_status_file(self, tmp_path):
        result = lt.audio_listen_status({})
        assert result["ok"] is True
        assert result["running"] is False
        assert "not detected" in result["message"]

    def test_reads_status_file(self, tmp_path):
        status = {
            "running": True,
            "capturing": True,
            "backend": "pipewire",
            "mode": "passive",
            "transcript_buffer_size": 5,
            "uptime_seconds": 120.5,
            "updated_at": time.time(),
        }
        lt._STATUS_FILE.write_text(json.dumps(status))

        result = lt.audio_listen_status({})
        assert result["ok"] is True
        assert result["running"] is True
        assert result["capturing"] is True
        assert result["backend"] == "pipewire"
        assert result["stale"] is False

    def test_stale_status_detected(self, tmp_path):
        status = {
            "running": True,
            "capturing": True,
            "updated_at": time.time() - 60,  # 60 seconds old
        }
        lt._STATUS_FILE.write_text(json.dumps(status))

        result = lt.audio_listen_status({})
        assert result["stale"] is True

    def test_corrupt_status_file(self, tmp_path):
        lt._STATUS_FILE.write_text("not json{{{")
        result = lt.audio_listen_status({})
        assert result["ok"] is False
        assert "Failed to read" in result["error"]

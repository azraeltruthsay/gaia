"""Waking from sleep unmutes gaia-audio; it doesn't reload the recognizer
onto the GPU (2026-11-16: every chat after a sleep loaded 1.8 GB of ASR
nobody used, and gear 1 no longer fit beside it)."""
from types import SimpleNamespace

import httpx

from gaia_core.cognition.sleep_wake_manager import SleepWakeManager


headers_seen = []


def _posts(monkeypatch):
    posted = []

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, *a, headers=None, **k):
            posted.append(url)
            headers_seen.append(headers or {})
            return SimpleNamespace(status_code=200)
    monkeypatch.setattr(httpx, "Client", _Client)
    return posted


def _manager():
    m = object.__new__(SleepWakeManager)
    m.config = SimpleNamespace(
        INTEGRATIONS={"audio": {"enabled": True, "mute_on_sleep": True}},
        get_endpoint=lambda name: "http://gaia-audio:8080",
        get_timeout=lambda name, default: default)
    m.voice_active = False
    return m


def test_waking_unmutes_and_does_not_reload(monkeypatch):
    from gaia_common.utils import service_auth
    monkeypatch.setattr(service_auth, "auth_headers", lambda m, p: {"X-GAIA-Auth": f"signed {m} {p}"})
    posted = _posts(monkeypatch)
    _manager()._notify_audio_state("active")
    assert posted == ["http://gaia-audio:8080/unmute"]
    # Signed: unsigned, gaia-audio refused /unmute with a 401 and she stayed muted.
    assert headers_seen[-1] == {"X-GAIA-Auth": "signed POST /unmute"}


def test_sleeping_still_unloads(monkeypatch):
    posted = _posts(monkeypatch)
    _manager()._notify_audio_state("asleep")
    assert posted == ["http://gaia-audio:8080/sleep"]

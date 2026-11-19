"""CFR uses whichever of her minds is loaded (2026-11-19: Prime-only CFR failed
every summary while Prime was unloaded)."""
import io
import urllib.error

from gaia_common.utils import cfr_manager as cm


def _caller(monkeypatch, answers):
    calls = []

    def post(self, endpoint, payload):
        calls.append(endpoint)
        a = answers[endpoint]
        if isinstance(a, int):
            raise urllib.error.HTTPError(endpoint, a, "x", {}, io.BytesIO(b""))
        if isinstance(a, Exception):
            raise a
        return a

    monkeypatch.setattr(cm._VLLMCaller, "_post", post)
    monkeypatch.setattr(cm.time, "sleep", lambda s: None)
    return cm._VLLMCaller("http://prime", "m", ["http://core"]), calls


def test_standby_prime_falls_through_to_core(monkeypatch):
    c, calls = _caller(monkeypatch, {"http://prime": 503, "http://core": "summary"})
    assert c.chat("s", "u") == "summary"
    assert calls == ["http://prime", "http://core"] and c.last_endpoint == "http://core"


def test_loaded_prime_answers_first(monkeypatch):
    c, calls = _caller(monkeypatch, {"http://prime": "deep", "http://core": "fast"})
    assert c.chat("s", "u") == "deep" and calls == ["http://prime"]


def test_unreachable_mind_is_skipped_and_all_down_raises(monkeypatch):
    c, calls = _caller(monkeypatch, {"http://prime": OSError("refused"), "http://core": "ok"})
    assert c.chat("s", "u") == "ok"
    c, calls = _caller(monkeypatch, {"http://prime": 503, "http://core": 503})
    try:
        c.chat("s", "u")
        raise AssertionError("should raise")
    except RuntimeError as e:
        assert "no mind answered" in str(e)
    assert len(calls) == 6


def test_a_real_error_is_not_hidden(monkeypatch):
    c, calls = _caller(monkeypatch, {"http://prime": 400, "http://core": "ok"})
    try:
        c.chat("s", "u")
        raise AssertionError("should raise")
    except urllib.error.HTTPError as e:
        assert e.code == 400


def test_config_adds_core_as_fallback(tmp_path):
    m = cm.CFRManager(state_dir=str(tmp_path))
    assert m._vllm.endpoints == ["http://gaia-prime:7777", "http://gaia-core:8092"]
    assert cm.CFRManager(state_dir=str(tmp_path), vllm_endpoint="http://x")._vllm.endpoints == ["http://x"]

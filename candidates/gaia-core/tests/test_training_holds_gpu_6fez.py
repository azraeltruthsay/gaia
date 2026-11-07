"""Nothing interrupts training (6fez, 2026-11-07)."""
from types import SimpleNamespace

import pytest

from gaia_common.lifecycle.states import LifecycleState, TransitionTrigger, TRANSITIONS
from gaia_core.models import vllm_remote_model as vr


def test_meditation_ends_only_when_training_completes():
    exits = TRANSITIONS[LifecycleState.MEDITATION]
    assert set(exits) == {TransitionTrigger.TRAINING_COMPLETE}


class Resp:
    def __init__(self, code, body=None):
        self.status_code, self._body, self.text = code, body or {}, ""

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise vr.requests.exceptions.HTTPError(str(self.status_code))


def _client(monkeypatch, training, adapter=None):
    m = vr.VLLMRemoteModel.__new__(vr.VLLMRemoteModel)
    m.endpoint, m.timeout, m._active_adapter, m.model_name = "http://gaia-prime:7777", 5, adapter, "/models/prime"
    calls = []

    def post(url, json=None, timeout=None):
        calls.append(url)
        return Resp(503) if "gaia-prime" in url else Resp(200, {"choices": [{"message": {"content": "from core"}}]})
    m._session = SimpleNamespace(post=post)
    monkeypatch.setattr(vr.VLLMRemoteModel, "_training_cache", (0.0, False))
    monkeypatch.setattr(vr.requests, "get", lambda *a, **k: Resp(200, {"state": "meditation" if training else "focusing"}))
    woke = []
    monkeypatch.setattr(vr.requests, "post", lambda url, **k: woke.append(url) or Resp(200))
    monkeypatch.setattr(vr.time, "sleep", lambda s: None)
    return m, calls, woke


def test_a_plain_call_during_training_goes_to_core_and_wakes_nothing(monkeypatch):
    m, calls, woke = _client(monkeypatch, training=True)
    out = m._post("/v1/chat/completions", {"model": "/models/prime", "messages": [{"role": "user", "content": "hi"}]})
    assert out["choices"][0]["message"]["content"] == "from core"
    assert calls[0].startswith("http://gaia-prime") and calls[1].endswith(":8092/v1/chat/completions")
    assert woke == []


def test_a_call_needing_primes_adapter_skips_its_turn(monkeypatch):
    m, calls, woke = _client(monkeypatch, training=True, adapter="code-architect")
    with pytest.raises(RuntimeError, match="training holds the GPU"):
        m._post("/v1/chat/completions", {"model": "code-architect", "messages": []})
    assert woke == [] and len(calls) == 1


def test_outside_training_prime_is_still_woken(monkeypatch):
    m, calls, woke = _client(monkeypatch, training=False)
    with pytest.raises(Exception):
        m._post("/v1/chat/completions", {"messages": []})
    assert woke and woke[0].endswith("/gpu/wake")

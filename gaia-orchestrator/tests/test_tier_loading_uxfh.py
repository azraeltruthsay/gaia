"""uxfh: while an engine loads, /health says mode=loading. The orchestrator
treated it as unloaded, reconciled with a swap of its own, and the first
load's timeout then killed the worker that swap started."""
from unittest.mock import AsyncMock, MagicMock

import pytest
from gaia_common.lifecycle.snapshot import TierLiveStatus

from gaia_orchestrator.consciousness_matrix import ConsciousnessLevel, ConsciousnessMatrix

LOADING = {"status": "ok", "engine": "gaia-managed", "managed": True, "model_loaded": False,
           "mode": "loading", "loading_model": "/models/core-gpu.gguf", "device": "cuda",
           "backend": "none"}


class _Resp:
    def __init__(self, data):
        self.status_code = 200
        self._data = data

    def json(self):
        return self._data


def _client(data):
    client = MagicMock()
    client.get = AsyncMock(return_value=_Resp(data))
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


@pytest.mark.asyncio
async def test_matrix_marks_a_loading_tier_and_it_is_not_ok(monkeypatch):
    import gaia_orchestrator.consciousness_matrix as cmod
    monkeypatch.setattr(cmod.httpx, "AsyncClient", lambda **k: _client(LOADING))
    m = ConsciousnessMatrix(lifecycle_machine=None)
    m._endpoints["core"] = "http://core"
    m._tiers["core"].target = ConsciousnessLevel.CONSCIOUS
    await m._probe_tier("core")
    st = m._tiers["core"]
    assert st.loading and st.healthy and not st.ok
    assert st.to_dict()["loading"] is True


@pytest.mark.asyncio
async def test_matrix_clears_loading_once_loaded(monkeypatch):
    import gaia_orchestrator.consciousness_matrix as cmod
    loaded = dict(LOADING, model_loaded=True, mode="active", backend="cpp")
    m = ConsciousnessMatrix(lifecycle_machine=None)
    m._endpoints["core"] = "http://core"
    m._tiers["core"].loading = True
    monkeypatch.setattr(cmod.httpx, "AsyncClient", lambda **k: _client(loaded))
    await m._probe_tier("core")
    assert not m._tiers["core"].loading
    assert m._tiers["core"].actual == ConsciousnessLevel.CONSCIOUS


@pytest.mark.asyncio
async def test_lifecycle_probe_reports_loading(monkeypatch):
    import gaia_orchestrator.lifecycle_machine as lmod
    monkeypatch.setattr(lmod.httpx, "AsyncClient", lambda **k: _client(LOADING))
    lm = lmod.LifecycleMachine(None)
    st = await lm._probe_tier("core", "http://core")
    assert st.loading and not st.model_loaded and st.healthy
    assert st.model_path == "/models/core-gpu.gguf"


@pytest.mark.asyncio
async def test_lifecycle_reconcile_defers_while_a_tier_loads(monkeypatch):
    import gaia_orchestrator.lifecycle_machine as lmod
    lm = lmod.LifecycleMachine(None)
    lm._snapshot.state = "awake"
    lm._tier_endpoints = {"core": "http://core", "prime": "http://prime"}

    async def probe(tier, ep):
        if tier == "core":
            return TierLiveStatus(loading=True, healthy=True, managed=True, endpoint=ep)
        return TierLiveStatus(device="cpu", model_loaded=True, healthy=True, endpoint=ep)

    lm._probe_tier = probe
    lm._load_tier = AsyncMock()
    lm._persist = AsyncMock()
    r = await lm.reconcile()
    assert r["skipped"] == "tier_loading" and r["new_state"] == "awake"
    lm._load_tier.assert_not_awaited()

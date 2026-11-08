"""a0un (2026-11-08): POST /consciousness/awake loaded Core to the GPU; five
seconds in, gaia-core's startup POST /gpu/wake saw the FSM still at
deep_sleep (an explicit shift syncs it only at the end) and applied the
resting gear: Core back to CPU, Prime onto the GPU. Awake's closing sync
then labelled that gear 2 "awake". Now the wake stands aside while an
explicit shift is in flight, and a shift replaced mid-flight doesn't sync.
"""

from __future__ import annotations

import asyncio

import pytest

from gaia_orchestrator import main as orch_main
from gaia_orchestrator.consciousness_matrix import ConsciousnessLevel, ConsciousnessMatrix


@pytest.fixture
def matrix(monkeypatch):
    monkeypatch.setattr(
        ConsciousnessMatrix, "_load_config",
        staticmethod(lambda: {
            "inference_endpoints": {"nano": "http://n:0", "core": "http://c:0", "prime": "http://p:0"},
            "gpu_models": {"nano": "/m/n", "core": "/m/c", "prime": "/m/p"},
            "cpu_models": {"nano": "/m/n.gguf", "core": "/m/c.gguf", "prime": "/m/p.gguf"},
        }),
    )
    cm = ConsciousnessMatrix(lifecycle_machine=None)
    synced = []

    async def slow_set_target(tier, level):
        await asyncio.sleep(0.05)
        cm._tiers[tier].actual = level
        return {"ok": True}

    async def fake_sync(config_name):
        synced.append(config_name)
        return {"state": config_name}

    async def noop(*a, **k):
        return None

    monkeypatch.setattr(cm, "set_target", slow_set_target)
    monkeypatch.setattr(cm, "_sync_lifecycle", fake_sync)
    monkeypatch.setattr(cm, "_notify_core_refresh_pool", noop)
    cm._synced = synced
    return cm


@pytest.mark.asyncio
async def test_in_flight_flag_set_only_during_explicit_shift(matrix):
    assert matrix.explicit_config_in_flight is None
    task = asyncio.create_task(matrix.awake())
    await asyncio.sleep(0.01)
    assert matrix.explicit_config_in_flight == "awake"
    await task
    assert matrix.explicit_config_in_flight is None
    assert matrix._synced == ["awake"]


@pytest.mark.asyncio
async def test_lifecycle_apply_does_not_claim_explicit(matrix):
    task = asyncio.create_task(matrix.apply_for_lifecycle("focusing"))
    await asyncio.sleep(0.01)
    assert matrix.explicit_config_in_flight is None
    await task


@pytest.mark.asyncio
async def test_shift_replaced_mid_flight_does_not_sync(matrix):
    awake = asyncio.create_task(matrix.awake())
    await asyncio.sleep(0.01)
    await matrix.apply_for_lifecycle("focusing")
    result = await awake
    assert result.get("superseded_by") == "focusing"
    assert matrix._synced == []


class _Machine:
    def __init__(self):
        self.calls = 0

    async def transition(self, trigger, reason=""):
        self.calls += 1
        raise AssertionError("wake should not transition during an explicit shift")


@pytest.mark.asyncio
async def test_gpu_wake_dropped_while_explicit_shift_in_flight(matrix, monkeypatch):
    import gaia_common.utils.maintenance as maint
    monkeypatch.setattr(maint, "is_maintenance_active", lambda: False)
    machine = _Machine()
    monkeypatch.setattr(orch_main, "_lifecycle_machine", machine)
    monkeypatch.setattr(orch_main, "_consciousness_matrix", matrix)
    awake = asyncio.create_task(matrix.awake())
    await asyncio.sleep(0.01)
    resp = await orch_main.gpu_wake()
    await awake
    assert resp["ok"] and resp["state"] == "awake"
    assert machine.calls == 0
    assert matrix._synced == ["awake"]

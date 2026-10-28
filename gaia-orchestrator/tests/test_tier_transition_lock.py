"""jykh: a manual /consciousness/awake during the poll loop's auto-reconcile
ran a second GPU load of Core alongside the first. Both checked "already
loaded?" before either finished, both swapped, the single-holder invariant
broke and Core was left unloaded. Transitions are now one at a time per
tier; the waiting caller then finds the model loaded and does nothing."""

from __future__ import annotations

import asyncio

import pytest

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
    engine = {"loaded": None, "swaps": 0, "active": 0, "max_active": 0}

    async def fake_load_gpu(tier, endpoint, state):
        # Same shape as the real loader: idempotent check, then a slow swap.
        if engine["loaded"] == "/m/c":
            state.actual = ConsciousnessLevel.CONSCIOUS
            return {"ok": True, "tier": tier, "action": "already_loaded_gpu"}
        engine["active"] += 1
        engine["max_active"] = max(engine["max_active"], engine["active"])
        await asyncio.sleep(0.05)
        engine["swaps"] += 1
        engine["loaded"] = "/m/c"
        engine["active"] -= 1
        state.actual = ConsciousnessLevel.CONSCIOUS
        return {"ok": True, "tier": tier, "action": "loaded_gpu"}

    async def fake_unload(tier, endpoint, state):
        engine["loaded"] = None
        state.actual = ConsciousnessLevel.UNCONSCIOUS
        return {"ok": True}

    async def noop(*a, **k):
        return None

    monkeypatch.setattr(cm, "_load_tier_gpu", fake_load_gpu)
    monkeypatch.setattr(cm, "_unload_tier", fake_unload)
    monkeypatch.setattr(cm, "_probe_tier", noop)
    monkeypatch.setattr(cm, "_wait_for_manager_ready", noop)
    cm._engine = engine
    return cm


@pytest.mark.asyncio
async def test_manual_awake_during_reconcile_loads_core_once(matrix):
    core = matrix._tiers["core"]
    core.actual = ConsciousnessLevel.UNCONSCIOUS
    core.target = ConsciousnessLevel.CONSCIOUS
    reconcile = matrix._transition_tier("core", core.actual, ConsciousnessLevel.CONSCIOUS)
    manual = matrix.set_target("core", ConsciousnessLevel.CONSCIOUS)
    r1, r2 = await asyncio.gather(reconcile, manual)
    assert r1["ok"] and r2["ok"]
    assert matrix._engine["swaps"] == 1 and matrix._engine["max_active"] == 1
    assert r2.get("action") == "already_loaded_gpu"
    assert core.actual == ConsciousnessLevel.CONSCIOUS


@pytest.mark.asyncio
async def test_two_reconciles_in_a_row_swap_once(matrix):
    # Same race from two auto-reconcile calls: still one swap, never two
    # holders, and the second leaves the freshly loaded model alone.
    core = matrix._tiers["core"]
    core.actual = ConsciousnessLevel.UNCONSCIOUS
    a = matrix._transition_tier("core", ConsciousnessLevel.UNCONSCIOUS, ConsciousnessLevel.CONSCIOUS)
    b = matrix._transition_tier("core", ConsciousnessLevel.UNCONSCIOUS, ConsciousnessLevel.CONSCIOUS)
    await asyncio.gather(a, b)
    assert matrix._engine["swaps"] == 1 and matrix._engine["loaded"] == "/m/c"


@pytest.mark.asyncio
async def test_different_tiers_do_not_wait_for_each_other(matrix):
    assert matrix._tier_locks.get("prime") is None
    await matrix._transition_tier("core", ConsciousnessLevel.UNCONSCIOUS, ConsciousnessLevel.CONSCIOUS)
    assert not matrix._tier_locks["core"].locked()
    assert "prime" not in matrix._tier_locks

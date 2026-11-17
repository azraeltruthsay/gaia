"""Gear 1 -> 2 handed the GPU over in order: Core off the GPU, Core's 33 s CPU
load, and only then Prime onto the GPU (52 s before Prime answered). The tier
taking the GPU now loads as soon as the one leaving has unloaded, while that
one finishes on the CPU (2026-11-17)."""

from __future__ import annotations

import asyncio

import pytest

from gaia_orchestrator.consciousness_matrix import ConsciousnessLevel as L, ConsciousnessMatrix


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
    log = []
    gpu = set()
    cm._unload_result = {}

    async def unload(tier, endpoint, state):
        if cm._unload_result.get(tier):
            return cm._unload_result[tier]
        gpu.discard(tier)
        log.append(("unloaded", tier))
        state.actual = L.UNCONSCIOUS
        return {"ok": True}

    async def load_cpu(tier, endpoint, state):
        log.append(("cpu start", tier))
        await asyncio.sleep(0.2)   # Core's slow CPU load
        log.append(("cpu done", tier))
        state.actual = L.SUBCONSCIOUS
        return {"ok": True, "tier": tier}

    async def load_gpu(tier, endpoint, state):
        assert not gpu, f"{tier} loaded onto a GPU still held by {gpu}"
        log.append(("gpu start", tier))
        await asyncio.sleep(0.05)
        gpu.add(tier)
        log.append(("gpu done", tier))
        state.actual = L.CONSCIOUS
        return {"ok": True, "tier": tier}

    async def noop(*a, **k):
        return None

    monkeypatch.setattr(cm, "_unload_tier", unload)
    monkeypatch.setattr(cm, "_load_tier_cpu", load_cpu)
    monkeypatch.setattr(cm, "_load_tier_gpu", load_gpu)
    monkeypatch.setattr(cm, "_probe_tier", noop)
    monkeypatch.setattr(cm, "_wait_for_manager_ready", noop)
    monkeypatch.setattr(cm, "_capture_context", noop)
    monkeypatch.setattr(cm, "_notify_core_refresh_pool", noop)
    cm._log, cm._gpu = log, gpu
    return cm


def _gear1(cm):
    cm._tiers["core"].actual = cm._tiers["core"].target = L.CONSCIOUS
    cm._tiers["prime"].actual = cm._tiers["prime"].target = L.SUBCONSCIOUS
    cm._gpu.add("core")


@pytest.mark.asyncio
async def test_prime_takes_gpu_before_core_cpu_load_finishes(matrix):
    _gear1(matrix)
    res = await matrix._apply_configuration("focusing", sync_lifecycle=False)
    log = matrix._log
    assert res["results"]["core"]["ok"] and res["results"]["prime"]["ok"]
    assert log.index(("gpu done", "prime")) < log.index(("cpu done", "core"))
    assert log.index(("unloaded", "core")) < log.index(("gpu start", "prime"))
    assert matrix._tiers["prime"].actual == L.CONSCIOUS
    assert matrix._tiers["core"].actual == L.SUBCONSCIOUS


@pytest.mark.asyncio
async def test_downshift_frees_gpu_before_core_loads(matrix):
    matrix._tiers["prime"].actual = matrix._tiers["prime"].target = L.CONSCIOUS
    matrix._tiers["core"].actual = matrix._tiers["core"].target = L.SUBCONSCIOUS
    matrix._gpu.add("prime")
    res = await matrix._apply_configuration("awake", sync_lifecycle=False)
    log = matrix._log
    assert all(r["ok"] for r in res["results"].values())
    assert log.index(("unloaded", "prime")) < log.index(("gpu start", "core"))
    assert log.index(("gpu done", "core")) < log.index(("cpu done", "prime"))


@pytest.mark.asyncio
async def test_busy_gpu_is_not_handed_over(matrix):
    _gear1(matrix)
    matrix._unload_result["core"] = {"ok": False, "error": "busy: active inference, drain timed out"}
    res = await matrix._apply_configuration("focusing", sync_lifecycle=False)
    assert not res["results"]["prime"]["ok"] and "still held" in res["results"]["prime"]["error"]
    assert ("gpu start", "prime") not in matrix._log
    assert matrix._tiers["core"].actual == L.CONSCIOUS


@pytest.mark.asyncio
async def test_no_handover_runs_in_order(matrix):
    _gear1(matrix)
    res = await matrix._apply_configuration("sleep", sync_lifecycle=False)
    assert res["results"]["core"]["ok"]
    assert matrix._tiers["prime"].actual == L.UNCONSCIOUS

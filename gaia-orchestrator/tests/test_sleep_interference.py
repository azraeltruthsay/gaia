"""Tests for sleep-task / active-GPU-worker interference fixes (GAIA_Project-r2kn,
beads-planning-2l9).

Pinned regressions:

  1. ``_apply_configuration`` must skip nano for EVERY preset. nano is a
     socat proxy to gaia-core:8092 in Sovereign Duality — acting on it
     posts /model/unload to Core's live engine. ``training()`` already
     skipped it; the preset path did not, so every awake/focusing/sleep
     application silently unloaded Core.

  2. ``LifecycleMachine._STATE_TO_CM_CONFIG`` must map PARKED. Without it,
     idle-timeout parks took the legacy direct-unload path, CM tier targets
     were never updated, and the 15s poll loop reloaded the model that had
     just been unloaded — a permanent load/unload fight cycle.

  3. Drain-before-unload: ``_unload_tier`` must not kill a worker with
     in-flight inference. If /inference/drain times out, the unload is
     aborted (busy error) and ``_transition_tier`` must NOT fall through
     to /model/swap (which would kill the live worker anyway).
"""

from __future__ import annotations

import pytest

from gaia_orchestrator.consciousness_matrix import (
    ConsciousnessLevel,
    ConsciousnessMatrix,
)
from gaia_orchestrator.lifecycle_machine import LifecycleMachine
from gaia_common.lifecycle.states import LifecycleState


@pytest.fixture
def matrix(monkeypatch):
    """ConsciousnessMatrix with config loading and lifecycle sync stubbed."""
    monkeypatch.setattr(
        ConsciousnessMatrix, "_load_config",
        staticmethod(lambda: {
            "inference_endpoints": {
                "nano": "http://test-nano:0",
                "core": "http://test-core:0",
                "prime": "http://test-prime:0",
            },
            "gpu_models": {"nano": "/m/n", "core": "/m/c", "prime": "/m/p"},
            "cpu_models": {"nano": "/m/n.gguf", "core": "/m/c.gguf", "prime": "/m/p.gguf"},
        }),
    )
    cm = ConsciousnessMatrix(lifecycle_machine=None)

    calls: list[tuple] = []

    async def fake_set_target(tier, level):
        calls.append(("set_target", tier, level))
        cm._tiers[tier].target = level
        return {"ok": True, "tier": tier, "level": level.name}

    async def fake_sync_lifecycle(config_name):
        calls.append(("sync_lifecycle", config_name))
        return {"lifecycle_sync": "ok", "state": config_name}

    async def fake_notify_refresh():
        calls.append(("notify_core_refresh_pool",))

    monkeypatch.setattr(cm, "set_target", fake_set_target)
    monkeypatch.setattr(cm, "_sync_lifecycle", fake_sync_lifecycle)
    monkeypatch.setattr(cm, "_notify_core_refresh_pool", fake_notify_refresh)
    cm._calls = calls
    return cm


class TestNanoSkippedInPresets:
    """Regression 1: no preset application may touch nano."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("preset", [
        "awake", "listening", "focusing", "sleep", "deep_sleep", "parked", "meditation",
    ])
    async def test_preset_never_targets_nano(self, matrix, preset):
        result = await matrix._apply_configuration(preset)
        targeted = [c[1] for c in matrix._calls if c[0] == "set_target"]
        assert "nano" not in targeted, f"{preset} targeted nano: {matrix._calls!r}"
        assert "nano" not in result["results"]

    @pytest.mark.asyncio
    async def test_awake_still_targets_core_and_prime(self, matrix):
        result = await matrix._apply_configuration("awake")
        assert result["results"]["core"]["ok"]
        assert result["results"]["prime"]["ok"]


class TestParkedLifecycleMapping:
    """Regression 2: PARKED must route through the CM, not the legacy path."""

    def test_parked_in_state_to_cm_config(self):
        assert LifecycleMachine._STATE_TO_CM_CONFIG.get(LifecycleState.PARKED) == "parked"

    def test_parked_preset_exists(self):
        assert "parked" in ConsciousnessMatrix._PRESETS

    def test_parked_config_maps_back_to_lifecycle(self):
        assert ConsciousnessMatrix._CONFIG_TO_LIFECYCLE.get("parked") == "parked"

    def test_every_cm_mapped_state_has_a_preset(self):
        for state, cfg in LifecycleMachine._STATE_TO_CM_CONFIG.items():
            assert cfg in ConsciousnessMatrix._PRESETS, \
                f"{state} maps to unknown CM preset {cfg!r}"


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Stub httpx.AsyncClient recording calls against a canned route table."""

    routes: dict = {}
    calls: list = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, **kwargs):
        _FakeAsyncClient.calls.append(("GET", url))
        return self.routes.get(("GET", url), _FakeResponse(404))

    async def post(self, url, **kwargs):
        _FakeAsyncClient.calls.append(("POST", url))
        return self.routes.get(("POST", url), _FakeResponse(404))


@pytest.fixture
def fake_httpx(monkeypatch):
    import gaia_orchestrator.consciousness_matrix as cm_mod
    _FakeAsyncClient.routes = {}
    _FakeAsyncClient.calls = []
    monkeypatch.setattr(cm_mod.httpx, "AsyncClient", _FakeAsyncClient)
    return _FakeAsyncClient


class TestDrainBeforeUnload:
    """Regression 3: never kill a worker with inference in flight."""

    ENDPOINT = "http://test-core:0"

    @pytest.mark.asyncio
    async def test_idle_engine_unloads_without_drain(self, matrix, fake_httpx):
        fake_httpx.routes = {
            ("GET", f"{self.ENDPOINT}/health"): _FakeResponse(200, {"active_inference": 0}),
            ("POST", f"{self.ENDPOINT}/model/unload"): _FakeResponse(200, {"ok": True}),
        }
        state = matrix._tiers["core"]
        result = await matrix._unload_tier("core", self.ENDPOINT, state)
        assert result["ok"]
        drained = [c for c in fake_httpx.calls if "/inference/drain" in c[1]]
        assert not drained

    @pytest.mark.asyncio
    async def test_busy_engine_drains_then_unloads(self, matrix, fake_httpx):
        fake_httpx.routes = {
            ("GET", f"{self.ENDPOINT}/health"): _FakeResponse(200, {"active_inference": 1}),
            ("POST", f"{self.ENDPOINT}/inference/drain"): _FakeResponse(200, {"ok": True}),
            ("POST", f"{self.ENDPOINT}/model/unload"): _FakeResponse(200, {"ok": True}),
        }
        state = matrix._tiers["core"]
        result = await matrix._unload_tier("core", self.ENDPOINT, state)
        assert result["ok"]
        order = [c[1].rsplit("/", 1)[-1] for c in fake_httpx.calls if c[0] == "POST"]
        assert order.index("drain") < order.index("unload")

    @pytest.mark.asyncio
    async def test_drain_timeout_aborts_unload_and_resumes(self, matrix, fake_httpx):
        fake_httpx.routes = {
            ("GET", f"{self.ENDPOINT}/health"): _FakeResponse(200, {"active_inference": 2}),
            ("POST", f"{self.ENDPOINT}/inference/drain"): _FakeResponse(
                200, {"ok": False, "message": "timeout (2 active)"}),
            ("POST", f"{self.ENDPOINT}/inference/resume"): _FakeResponse(200, {"ok": True}),
        }
        state = matrix._tiers["core"]
        state.actual = ConsciousnessLevel.CONSCIOUS
        result = await matrix._unload_tier("core", self.ENDPOINT, state)
        assert not result["ok"]
        assert "busy" in result["error"]
        unloaded = [c for c in fake_httpx.calls if "/model/unload" in c[1]]
        assert not unloaded, "unload must be aborted when drain times out"
        resumed = [c for c in fake_httpx.calls if "/inference/resume" in c[1]]
        assert resumed, "engine must be resumed after an aborted drain"
        # Actual state must not be falsified to UNCONSCIOUS — model is still up
        assert state.actual == ConsciousnessLevel.CONSCIOUS

    @pytest.mark.asyncio
    async def test_transition_busy_abort_never_reaches_swap(self, matrix, fake_httpx, monkeypatch):
        """A busy abort in _unload_tier must stop _transition_tier before
        _load_tier_* — /model/swap would kill the live worker anyway."""
        async def busy_unload(tier, endpoint, state):
            return {"ok": False, "tier": tier, "error": "busy: active inference, drain timed out"}

        load_calls = []

        async def fake_load_cpu(tier, endpoint, state):
            load_calls.append(("cpu", tier))
            return {"ok": True, "tier": tier}

        async def fake_load_gpu(tier, endpoint, state):
            load_calls.append(("gpu", tier))
            return {"ok": True, "tier": tier}

        async def fake_probe(tier):
            return None

        monkeypatch.setattr(matrix, "_unload_tier", busy_unload)
        monkeypatch.setattr(matrix, "_load_tier_cpu", fake_load_cpu)
        monkeypatch.setattr(matrix, "_load_tier_gpu", fake_load_gpu)
        monkeypatch.setattr(matrix, "_probe_tier", fake_probe)

        for target in (ConsciousnessLevel.SUBCONSCIOUS, ConsciousnessLevel.CONSCIOUS):
            load_calls.clear()
            result = await matrix._transition_tier(
                "core", ConsciousnessLevel.CONSCIOUS, target,
            )
            assert not result.get("ok")
            assert "busy" in str(result.get("error", ""))
            assert not load_calls, f"busy abort leaked into load for {target}: {load_calls}"

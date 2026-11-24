"""A wake from park is budgeted from what waking has measurably added (b1ak).

2026-11-20: "insufficient VRAM for awake: need 12800 MB, external tenants hold
3409 MB of 16303 MB". Part of that 3.4 GB was GAIA's own parked residue, which
waking frees or reuses: on 2026-11-24 parked -> awake added 12200 MB to the
card and awake -> parked freed 12162, against Core's own 12800-13051.
"""
import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from gaia_common.lifecycle.snapshot import TierLiveStatus
from gaia_common.lifecycle.states import LifecycleState
from gaia_orchestrator.lifecycle_machine import LifecycleMachine


class _Gpu:
    def __init__(self, used):
        self.used = used

    async def get_memory_info(self):
        return SimpleNamespace(used_mb=self.used, total_mb=16303)


@pytest.fixture
def machine(tmp_path):
    def make(used, deltas=None):
        m = LifecycleMachine()
        m.GEAR_DELTAS_PATH = tmp_path / "gear_deltas.json"
        if deltas is not None:
            m.GEAR_DELTAS_PATH.write_text(json.dumps(deltas))
        m._gpu_manager = _Gpu(used)
        m._docker_manager = None
        m._snapshot.state = LifecycleState.PARKED
        m._snapshot.tiers = {"core": TierLiveStatus(device="cpu", model_loaded=True, vram_mb=0)}
        m._snapshot.last_transition_at = None
        return m
    return make


def test_the_refused_wake_fits_by_what_waking_adds(machine):
    m = machine(3409, {"awake->parked": [-12162, -12100]})
    assert asyncio.run(m._vram_preflight(LifecycleState.AWAKE)) is None


def test_measured_wakes_count_as_well_as_parks(machine):
    m = machine(3409, {"parked->awake": [12200, 12150]})
    assert asyncio.run(m._vram_preflight(LifecycleState.AWAKE)) is None


def test_without_two_readings_the_estimate_stands(machine):
    for deltas in (None, {}, {"awake->parked": [-12162]}):
        m = machine(3409, deltas)
        err = asyncio.run(m._vram_preflight(LifecycleState.AWAKE))
        assert err and "external tenants hold 3409 MB" in err


def test_a_game_on_the_card_is_still_refused(machine):
    # 2026-11-21: Baldur's Gate 3 on Windows held ~7 GB.
    m = machine(7070, {"awake->parked": [-12162, -12100]})
    err = asyncio.run(m._vram_preflight(LifecycleState.AWAKE))
    assert err and "7070 MB" in err


def test_never_looser_than_the_estimate(machine):
    # Readings above the estimate (something started mid-shift) fall back to it.
    m = machine(3409, {"parked->awake": [13500, 13400]})
    assert asyncio.run(m._vram_preflight(LifecycleState.AWAKE))


def test_the_largest_reading_is_used(machine):
    # One low reading (something else freed memory mid-shift) can't loosen it.
    m = machine(3409, {"parked->awake": [9000, 12500]})
    assert asyncio.run(m._vram_preflight(LifecycleState.AWAKE))


def test_a_settled_shift_is_recorded(machine):
    m = machine(14400)
    m.VRAM_SETTLE_MAX_S = 0
    m._snapshot.state = LifecycleState.AWAKE
    at = m._snapshot.last_transition_at = datetime.now(timezone.utc)
    for _ in range(7):
        asyncio.run(m._record_settled_delta(LifecycleState.PARKED, LifecycleState.AWAKE, 2200, at))
    assert json.loads(m.GEAR_DELTAS_PATH.read_text()) == {"parked->awake": [12200] * 5}


def test_a_shift_overtaken_by_another_is_not_recorded(machine):
    m = machine(14400)
    m.VRAM_SETTLE_MAX_S = 0
    m._snapshot.state = LifecycleState.AWAKE
    m._snapshot.last_transition_at = datetime.now(timezone.utc)
    earlier = datetime(2026, 11, 24, tzinfo=timezone.utc)
    asyncio.run(m._record_settled_delta(LifecycleState.PARKED, LifecycleState.AWAKE, 2200, earlier))
    m._snapshot.state = LifecycleState.FOCUSING
    asyncio.run(m._record_settled_delta(LifecycleState.PARKED, LifecycleState.AWAKE, 2200,
                                        m._snapshot.last_transition_at))
    assert not m.GEAR_DELTAS_PATH.exists()


class _Matrix:
    def __init__(self, gpu, used_after):
        self.gpu, self.used_after = gpu, used_after

    async def apply_for_lifecycle(self, config):
        self.gpu.used = self.used_after
        return {"results": {}, "configuration": config}


def test_a_real_wake_records_what_it_added(machine):
    # The recorder sat in the wrong method at first and never ran (2026-11-24).
    from gaia_common.lifecycle.states import TransitionTrigger
    m = machine(2200)
    m.VRAM_SETTLE_MAX_S = 0
    m._tier_endpoints = {}
    m._consciousness_matrix = _Matrix(m._gpu_manager, 14400)

    async def nothing(*a, **k):
        return None
    m._persist = nothing
    m._prewarm_kv = nothing

    async def run():
        result = await m.transition(TransitionTrigger.WAKE_SIGNAL, reason="gpu_wake")
        assert result.ok, result.error
        await m._settle_task
        return result.to_state  # the resting gear
    gear = asyncio.run(run())
    assert json.loads(m.GEAR_DELTAS_PATH.read_text()) == {f"parked->{gear}": [12200]}

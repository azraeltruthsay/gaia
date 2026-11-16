"""A wake right after a park waits for the old GPU worker's memory (hii8).

2026-11-16 02:25:45, 25 s after a park: "insufficient VRAM for awake: need
12800 MB, external tenants hold 4115 MB". A minute later the GPU was at its
baseline: the memory was still being freed, not held by an outside tenant.
"""
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from gaia_common.lifecycle.snapshot import TierLiveStatus
from gaia_common.lifecycle.states import LifecycleState
from gaia_orchestrator.lifecycle_machine import LifecycleMachine


class _Gpu:
    def __init__(self, readings):
        self.readings = list(readings)

    async def get_memory_info(self):
        used = self.readings.pop(0) if len(self.readings) > 1 else self.readings[0]
        return SimpleNamespace(used_mb=used, total_mb=16303)


def _machine(readings, since_shift_s):
    m = LifecycleMachine()
    m._gpu_manager = _Gpu(readings)
    m._docker_manager = None
    m._snapshot.tiers = {"core": TierLiveStatus(device="cpu", model_loaded=True, vram_mb=1950)}
    m._snapshot.last_transition_at = (
        None if since_shift_s is None else datetime.now(timezone.utc) - timedelta(seconds=since_shift_s))
    m.VRAM_SETTLE_POLL_S = 0.01
    m.VRAM_SETTLE_STALL_S = 0.05
    m.VRAM_SETTLE_MAX_S = 2.0
    return m


def test_wake_after_park_waits_for_memory_to_drain():
    # 6065 used (4115 external) draining to 3950 (2000 external)
    m = _machine([6065, 5500, 4800, 4100, 3950], since_shift_s=25)
    assert asyncio.run(m._vram_preflight(LifecycleState.AWAKE)) is None


def test_no_wait_long_after_a_shift():
    m = _machine([6065, 3950], since_shift_s=3600)
    err = asyncio.run(m._vram_preflight(LifecycleState.AWAKE))
    assert err and "4115 MB" in err


def test_a_real_tenant_that_never_drains_is_still_refused_quickly():
    m = _machine([6065], since_shift_s=10)
    err = asyncio.run(m._vram_preflight(LifecycleState.AWAKE))
    assert err and "external tenants hold 4115 MB" in err


def test_no_transition_yet_means_no_wait():
    m = _machine([6065, 3950], since_shift_s=None)
    assert asyncio.run(m._vram_preflight(LifecycleState.AWAKE))

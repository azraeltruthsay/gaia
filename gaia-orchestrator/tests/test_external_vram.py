"""A CPU-gear tier's CUDA buffers are GAIA's, not an outside tenant's (2026-11-11).

Parked, gaia_cpp Core runs its layers on the CPU but still reserves ~1.9 GB of
CUDA compute buffers. Counting that as external refused every wake from park:
"need 12800 MB, external tenants hold 3764 MB".
"""
import asyncio
from types import SimpleNamespace

from gaia_common.lifecycle.snapshot import TierLiveStatus
from gaia_orchestrator.lifecycle_machine import LifecycleMachine


class _Gpu:
    async def get_memory_info(self):
        return SimpleNamespace(used_mb=3764, total_mb=16303)


def test_cpu_tier_vram_is_not_external():
    m = LifecycleMachine()
    m._gpu_manager = _Gpu()
    m._snapshot.tiers = {
        "core": TierLiveStatus(device="cpu", model_loaded=True, vram_mb=1950),
        "prime": TierLiveStatus(device="unloaded"),
    }
    external, total = asyncio.run(m._external_vram_mb())
    assert (external, total) == (3764 - 1950, 16303)

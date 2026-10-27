"""8xr4.14: a persisted lifecycle gear must not survive a boot whose actual
tier placement contradicts it (Arch persisted 'focusing'; WSL booted the
'parked' preset; gaia-core routed to an unloaded Prime -> Groq fallback)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gaia_common.lifecycle.states import LifecycleState
from gaia_orchestrator.consciousness_matrix import ConsciousnessLevel, ConsciousnessMatrix

C, S, U = ConsciousnessLevel.CONSCIOUS, ConsciousnessLevel.SUBCONSCIOUS, ConsciousnessLevel.UNCONSCIOUS


def _matrix(persisted, core, prime, healthy=True):
    lm = MagicMock()
    lm._snapshot.state = persisted
    lm.set_state_external = AsyncMock()
    m = ConsciousnessMatrix(lifecycle_machine=lm)
    for tier, level in (("core", core), ("prime", prime)):
        m._tiers[tier].target = level
        m._tiers[tier].actual = level
        m._tiers[tier].healthy = healthy
    return m, lm


@pytest.mark.asyncio
async def test_stale_focusing_relabeled_to_parked():
    m, lm = _matrix("focusing", core=S, prime=U)
    await m._verify_boot_lifecycle_label()
    lm.set_state_external.assert_awaited_once()
    assert lm.set_state_external.await_args.kwargs["target"] == LifecycleState.PARKED


@pytest.mark.asyncio
async def test_stale_focusing_with_core_gpu_relabeled_to_awake():
    m, lm = _matrix("focusing", core=C, prime=S)
    await m._verify_boot_lifecycle_label()
    assert lm.set_state_external.await_args.kwargs["target"] == LifecycleState.AWAKE


@pytest.mark.asyncio
async def test_persisted_gear_matching_placement_is_kept():
    # orchestrator restart while AWAKE: Core still on GPU
    m, lm = _matrix("awake", core=C, prime=S)
    await m._verify_boot_lifecycle_label()
    lm.set_state_external.assert_not_awaited()
    assert m._boot_label_verified


@pytest.mark.asyncio
async def test_waits_for_settle_then_runs_once():
    m, lm = _matrix("focusing", core=S, prime=U, healthy=False)
    await m._verify_boot_lifecycle_label()
    lm.set_state_external.assert_not_awaited()
    assert not m._boot_label_verified
    for t in ("core", "prime"):
        m._tiers[t].healthy = True
    await m._verify_boot_lifecycle_label()
    await m._verify_boot_lifecycle_label()
    lm.set_state_external.assert_awaited_once()


@pytest.mark.asyncio
async def test_meditation_never_relabeled():
    m, lm = _matrix("meditation", core=U, prime=U)
    await m._verify_boot_lifecycle_label()
    lm.set_state_external.assert_not_awaited()

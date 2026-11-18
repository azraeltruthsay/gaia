"""Tests for the three-tier GPUManager (qvo6: rewritten; the half-duplex
STT/TTS swapper these tests described is gone)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gaia_audio.gpu_manager import GPUManager


def _engine(loaded=False, vram=0):
    e = MagicMock()
    e.loaded = loaded
    e.vram_mb = vram

    def load():
        e.loaded = True

    def unload():
        e.loaded = False

    e.load.side_effect = load
    e.unload.side_effect = unload
    return e


@pytest.fixture
def gm(monkeypatch):
    from gaia_audio import gpu_manager as mod
    monkeypatch.setattr(mod.status_tracker, "emit", AsyncMock())
    m = GPUManager(stt_engine=_engine(vram=1800), nano_speaker=_engine(),
                   prime_speaker=_engine(vram=3400), espeak_fallback=_engine())
    m.start_idle_timer = AsyncMock()
    return m


@pytest.mark.asyncio
async def test_startup_loads_listener_and_nano_not_prime(gm):
    await gm.startup()
    assert gm.stt.loaded and gm.nano.loaded and not gm.prime.loaded
    gm.espeak.load.assert_not_called()


@pytest.mark.asyncio
async def test_startup_falls_back_to_espeak(gm):
    gm.nano.load.side_effect = RuntimeError("no model")
    await gm.startup()
    gm.espeak.load.assert_called_once()


@pytest.mark.asyncio
async def test_sleep_unloads_gpu_keeps_nano_and_stt_wakes_on_demand(gm):
    await gm.startup()
    await gm.sleep()
    assert not gm.stt.loaded and gm.nano.loaded
    assert await gm.run_stt(lambda: "heard") == "heard"
    assert gm.stt.loaded


@pytest.mark.asyncio
async def test_prime_speaker_loads_runs_and_unloads_under_a_lease(gm):
    gm._acquire_gpu_lease = AsyncMock(return_value=True)
    gm._release_gpu_lease = AsyncMock()
    seen = []
    out = await gm.run_tts_prime(lambda: seen.append(gm.prime.loaded) or "wav")
    assert out == "wav" and seen == [True]
    assert not gm.prime.loaded
    gm._release_gpu_lease.assert_awaited_once()


@pytest.mark.asyncio
async def test_prime_speaker_denied_lease_returns_none(gm):
    gm._acquire_gpu_lease = AsyncMock(return_value=False)
    assert await gm.run_tts_prime(lambda: "wav") is None
    gm.prime.load.assert_not_called()


@pytest.mark.asyncio
async def test_prime_speaker_unloads_even_when_synthesis_fails(gm):
    gm._acquire_gpu_lease = AsyncMock(return_value=True)
    gm._release_gpu_lease = AsyncMock()

    def boom():
        raise ValueError("bad text")

    with pytest.raises(ValueError):
        await gm.run_tts_prime(boom)
    assert not gm.prime.loaded
    gm._release_gpu_lease.assert_awaited_once()


@pytest.mark.asyncio
async def test_release_unloads_everything(gm):
    await gm.startup()
    await gm.release()
    assert not (gm.stt.loaded or gm.nano.loaded or gm.prime.loaded)

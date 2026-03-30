"""Tests for GPUManager — half-duplex VRAM swapping."""

from unittest.mock import MagicMock

import pytest

from gaia_audio.gpu_manager import GPUManager
from gaia_audio.stt_engine import STTEngine
from gaia_audio.tts_engine import TTSEngine


@pytest.fixture
def mock_stt():
    engine = MagicMock(spec=STTEngine)
    engine.loaded = False
    engine.vram_mb = 150
    engine.model_size = "base.en"
    return engine


@pytest.fixture
def mock_tts():
    engine = MagicMock(spec=TTSEngine)
    engine.loaded = False
    engine.vram_mb = 0
    engine.engine_type = "system"
    return engine


@pytest.fixture
def gpu_manager(mock_stt, mock_tts):
    return GPUManager(stt_engine=mock_stt, tts_engine=mock_tts, vram_budget_mb=5600)


@pytest.mark.asyncio
async def test_acquire_stt_loads_model(gpu_manager, mock_stt):
    await gpu_manager.acquire_for_stt()
    mock_stt.load.assert_called_once()
    assert gpu_manager.current_mode == "stt"


@pytest.mark.asyncio
async def test_acquire_tts_loads_engine(gpu_manager, mock_tts):
    await gpu_manager.acquire_for_tts()
    mock_tts.load.assert_called_once()
    assert gpu_manager.current_mode == "tts"


@pytest.mark.asyncio
async def test_swap_stt_to_tts_unloads_stt(gpu_manager, mock_stt, mock_tts):
    # Start in STT mode
    mock_stt.loaded = True
    gpu_manager.current_mode = "stt"

    await gpu_manager.acquire_for_tts()

    mock_stt.unload.assert_called_once()
    mock_tts.load.assert_called_once()
    assert gpu_manager.current_mode == "tts"


@pytest.mark.asyncio
async def test_swap_tts_to_stt_unloads_tts(gpu_manager, mock_stt, mock_tts):
    # Start in TTS mode
    mock_tts.loaded = True
    gpu_manager.current_mode = "tts"

    await gpu_manager.acquire_for_stt()

    mock_tts.unload.assert_called_once()
    mock_stt.load.assert_called_once()
    assert gpu_manager.current_mode == "stt"


@pytest.mark.asyncio
async def test_release_unloads_all(gpu_manager, mock_stt, mock_tts):
    mock_stt.loaded = True
    mock_tts.loaded = True
    gpu_manager.current_mode = "stt"

    await gpu_manager.release()

    mock_stt.unload.assert_called_once()
    mock_tts.unload.assert_called_once()
    assert gpu_manager.current_mode == "idle"


@pytest.mark.asyncio
async def test_double_acquire_stt_is_noop(gpu_manager, mock_stt):
    mock_stt.loaded = True
    gpu_manager.current_mode = "stt"

    await gpu_manager.acquire_for_stt()

    mock_stt.load.assert_not_called()
    mock_stt.unload.assert_not_called()

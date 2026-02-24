"""Tests for STTEngine — Whisper-based speech-to-text."""

import numpy as np
import pytest

from gaia_audio.stt_engine import STTEngine, audio_bytes_to_array


def test_engine_starts_unloaded():
    engine = STTEngine(model_size="tiny.en", device="cpu")
    assert not engine.loaded
    assert engine.vram_mb == 75


def test_engine_model_size_vram():
    assert STTEngine(model_size="base.en").vram_mb == 150
    assert STTEngine(model_size="small.en").vram_mb == 500
    assert STTEngine(model_size="medium.en").vram_mb == 1500


def test_transcribe_sync_raises_when_not_loaded():
    engine = STTEngine(model_size="tiny.en", device="cpu")
    with pytest.raises(RuntimeError, match="not loaded"):
        engine.transcribe_sync(np.zeros(16000, dtype=np.float32))


def test_audio_bytes_to_array_raw_pcm():
    """Raw int16 PCM bytes should be converted to float32."""
    samples = np.array([0, 16384, -16384, 32767], dtype=np.int16)
    raw_bytes = samples.tobytes()
    result = audio_bytes_to_array(raw_bytes)
    assert result.dtype == np.float32
    assert len(result) == 4
    assert abs(result[1] - 0.5) < 0.01  # 16384 / 32768 ≈ 0.5


def test_audio_bytes_to_array_wav():
    """WAV file bytes should be decoded by soundfile."""
    import io
    try:
        import soundfile as sf
        # Create a valid WAV in memory
        buf = io.BytesIO()
        data = np.random.randn(8000).astype(np.float32)
        sf.write(buf, data, 16000, format="WAV")
        buf.seek(0)
        result = audio_bytes_to_array(buf.read())
        assert result.dtype == np.float32
        assert len(result) == 8000
    except ImportError:
        pytest.skip("soundfile not available")

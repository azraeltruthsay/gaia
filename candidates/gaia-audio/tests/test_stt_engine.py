"""Tests for STTEngine (qvo6: Qwen3-ASR now; it takes a model path, not a
Whisper model size)."""

import numpy as np
import pytest

from gaia_audio.stt_engine import STTEngine, audio_bytes_to_array


def test_engine_starts_unloaded():
    engine = STTEngine(model_path="/nonexistent", device="cpu")
    assert not engine.loaded
    assert engine.vram_mb == 0


def test_transcribe_sync_raises_when_not_loaded():
    engine = STTEngine(model_path="/nonexistent", device="cpu")
    with pytest.raises(RuntimeError, match="not loaded"):
        engine.transcribe_sync(np.zeros(16000, dtype=np.float32))


def test_audio_bytes_to_array_raw_pcm():
    """Raw int16 PCM bytes should be converted to float32."""
    samples = np.array([0, 16384, -16384, 32767], dtype=np.int16)
    result = audio_bytes_to_array(samples.tobytes())
    assert result.dtype == np.float32
    assert len(result) == 4
    assert abs(result[1] - 0.5) < 0.01


def test_audio_bytes_to_array_wav():
    """WAV file bytes should be decoded by soundfile."""
    import io
    sf = pytest.importorskip("soundfile")
    buf = io.BytesIO()
    sf.write(buf, np.random.randn(8000).astype(np.float32), 16000, format="WAV")
    result = audio_bytes_to_array(buf.getvalue())
    assert result.dtype == np.float32
    assert len(result) == 8000

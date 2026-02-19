"""Tests for TTSEngine — RealtimeTTS wrapper."""

import pytest

from gaia_audio.tts_engine import ENGINE_VRAM_MB, TTSEngine


def test_engine_starts_unloaded():
    engine = TTSEngine(engine_type="system")
    assert not engine.loaded
    assert engine.vram_mb == 0


def test_engine_vram_estimates():
    assert ENGINE_VRAM_MB["system"] == 0
    assert ENGINE_VRAM_MB["coqui"] == 3000
    assert ENGINE_VRAM_MB["elevenlabs"] == 0


def test_synthesize_sync_raises_when_not_loaded():
    engine = TTSEngine(engine_type="system")
    with pytest.raises(RuntimeError, match="not loaded"):
        engine.synthesize_sync("Hello world")


def test_list_voices_system():
    engine = TTSEngine(engine_type="system")
    voices = engine.list_voices()
    assert len(voices) == 1
    assert voices[0]["engine"] == "system"
    assert voices[0]["voice_id"] == "default"


def test_list_voices_coqui():
    engine = TTSEngine(engine_type="coqui")
    voices = engine.list_voices()
    assert len(voices) == 1
    assert voices[0]["engine"] == "coqui"


def test_unknown_engine_raises():
    engine = TTSEngine(engine_type="nonexistent")
    with pytest.raises(ValueError, match="Unknown TTS engine"):
        engine.load()

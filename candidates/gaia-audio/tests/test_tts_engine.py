"""Tests for the three speakers (qvo6: rewritten for NanoSpeaker /
PrimeSpeaker / EspeakFallback; the old TTSEngine is gone)."""

import pytest

from gaia_audio.tts_engine import EspeakFallback, NanoSpeaker, PrimeSpeaker, _sanitize_for_speech


@pytest.mark.parametrize("cls,vram", [(NanoSpeaker, 0), (PrimeSpeaker, None)])
def test_speakers_start_unloaded(cls, vram):
    s = cls(model_path="/nonexistent")
    assert not s.loaded
    if vram is not None:
        assert s.vram_mb == vram


@pytest.mark.parametrize("cls", [NanoSpeaker, PrimeSpeaker])
def test_synthesize_raises_when_not_loaded(cls):
    with pytest.raises(RuntimeError, match="not loaded"):
        cls(model_path="/nonexistent").synthesize_sync("Hello world")


def test_espeak_not_ready_until_loaded():
    e = EspeakFallback()
    assert not e.loaded and e.vram_mb == 0
    with pytest.raises(RuntimeError, match="not ready"):
        e.synthesize_sync("Hello")


def test_voices_offer_clone_only_with_reference():
    assert [v["voice_id"] for v in NanoSpeaker(model_path="/x").list_voices()] == ["default"]
    with_ref = NanoSpeaker(model_path="/x", voice_ref_audio="/ref.wav").list_voices()
    assert [v["voice_id"] for v in with_ref] == ["default", "clone"]
    assert all(v["engine"] == "nano_speaker" for v in with_ref)


def test_sanitize_strips_markdown_and_tier_tags():
    out = _sanitize_for_speech("## Title\n**Bold** [Prime] text\n- item")
    assert "#" not in out and "*" not in out and "[Prime]" not in out
    assert "Bold" in out and "item" in out

"""Tests for AudioStatusTracker — state machine + event ring buffer."""


import pytest

from gaia_audio.status import AudioEvent, AudioStatusTracker


@pytest.fixture
def tracker():
    return AudioStatusTracker()


@pytest.mark.asyncio
async def test_emit_stores_event(tracker):
    event = await tracker.emit("stt_start", "transcribing audio")
    assert event.event_type == "stt_start"
    assert event.detail == "transcribing audio"
    snap = tracker.snapshot()
    assert len(snap["events"]) == 1
    assert snap["events"][0]["event_type"] == "stt_start"


@pytest.mark.asyncio
async def test_ring_buffer_max_size(tracker):
    for i in range(120):
        await tracker.emit("test_event", f"event {i}")
    snap = tracker.snapshot()
    # Ring buffer is capped at 100
    assert len(snap["events"]) == 100


@pytest.mark.asyncio
async def test_latency_tracking(tracker):
    await tracker.emit("stt_complete", "hello", latency_ms=150.0)
    await tracker.emit("stt_complete", "world", latency_ms=200.0)
    await tracker.emit("tts_complete", "response", latency_ms=300.0)

    snap = tracker.snapshot()
    assert len(snap["stt_latencies"]) == 2
    assert snap["stt_latencies"] == [150.0, 200.0]
    assert len(snap["tts_latencies"]) == 1


@pytest.mark.asyncio
async def test_websocket_subscription(tracker):
    queue = tracker.subscribe()
    await tracker.emit("gpu_swap", "STT → TTS")

    msg = queue.get_nowait()
    assert msg["event_type"] == "gpu_swap"

    tracker.unsubscribe(queue)
    await tracker.emit("another", "event")
    assert queue.empty()


@pytest.mark.asyncio
async def test_snapshot_contents(tracker):
    tracker.state = "transcribing"
    tracker.gpu_mode = "stt"
    tracker.stt_model = "base.en"
    tracker.vram_used_mb = 150.0
    tracker.muted = False
    tracker.last_transcription = "hello world"

    snap = tracker.snapshot()
    assert snap["state"] == "transcribing"
    assert snap["gpu_mode"] == "stt"
    assert snap["stt_model"] == "base.en"
    assert snap["vram_used_mb"] == 150.0
    assert snap["last_transcription"] == "hello world"


def test_audio_event_to_dict():
    event = AudioEvent(
        timestamp="2026-02-18T12:00:00",
        event_type="stt_complete",
        detail="hello",
        latency_ms=100.5,
    )
    d = event.to_dict()
    assert d["event_type"] == "stt_complete"
    assert d["latency_ms"] == 100.5

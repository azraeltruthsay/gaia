"""tj0a: long files are decoded once, transcribed in chunks, written atomically."""
import asyncio

import numpy as np
import pytest

from gaia_audio import file_transcriber as ft


def test_chunks_cover_everything():
    samples = np.zeros(ft.SAMPLE_RATE * 250, dtype=np.float32)
    parts = ft.chunks(samples, seconds=120)
    assert [len(p) // ft.SAMPLE_RATE for p in parts] == [120, 120, 10]


def test_only_under_the_allowed_root(tmp_path, monkeypatch):
    monkeypatch.setattr(ft, "ALLOWED_ROOT", tmp_path)
    assert ft.allowed(str(tmp_path / "episodes" / "a.mp4"))
    assert not ft.allowed("/etc/passwd")
    assert not ft.allowed(str(tmp_path / ".." / "outside.txt"))


def test_transcribes_a_real_file_in_chunks(tmp_path, monkeypatch):
    sf = pytest.importorskip("soundfile")
    monkeypatch.setattr(ft, "CHUNK_SECONDS", 2)
    src = tmp_path / "ep.wav"
    sf.write(str(src), np.zeros(ft.SAMPLE_RATE * 5, dtype=np.float32), ft.SAMPLE_RATE)
    seen = []

    async def fake(chunk):
        seen.append(len(chunk))
        return f"part{len(seen)}"

    out = tmp_path / "t" / "ep.txt"
    text = asyncio.run(ft.transcribe(str(src), str(out), fake))
    assert len(seen) == 3 and text == "part1\n\npart2\n\npart3"
    assert out.read_text() == text and not (out.parent / "ep.txt.tmp").exists()


def test_a_running_job_is_not_started_twice(tmp_path):
    async def go():
        gate = asyncio.Event()

        async def slow(chunk):
            await gate.wait()
            return ""

        orig = ft.transcribe

        async def fake_transcribe(path, out_path, chunk_fn, language=None):
            await slow(None)

        ft.transcribe = fake_transcribe
        try:
            first = ft.start("a", str(tmp_path / "o.txt"), slow)
            second = ft.start("a", str(tmp_path / "o.txt"), slow)
            gate.set()
            await asyncio.sleep(0)
            return first, second
        finally:
            ft.transcribe = orig

    assert asyncio.run(go()) == ("started", "running")

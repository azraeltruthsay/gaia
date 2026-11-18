"""Transcribe a long audio file on disk in chunks (GAIA_Project-tj0a).

A podcast episode runs 40+ minutes (~88 MB m4a). Sent whole as base64 to
/transcribe it was far past the request limit, and one ASR call over 40
minutes is not what the Listener is for. Here the file is decoded once with
ffmpeg to 16 kHz mono, cut into chunks, each chunk transcribed in turn, and
the joined text written next to where the caller waits for it.

Only files under ALLOWED_ROOT are read or written: the endpoint takes paths.
Unlike /transcribe this never signals a wake to gaia-core: it is not someone
speaking, and it runs while she sleeps.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from pathlib import Path
from typing import Awaitable, Callable, Dict, List

import numpy as np

logger = logging.getLogger("GAIA.Audio.FileTranscriber")

SAMPLE_RATE = 16000
CHUNK_SECONDS = 120
ALLOWED_ROOT = Path(os.environ.get("SHARED_DIR", "/shared")) / "penpal"
DECODE_TIMEOUT_S = 600

_running: Dict[str, asyncio.Task] = {}


def allowed(path: str) -> bool:
    """Whether `path` lies under ALLOWED_ROOT (after resolving links and ..)."""
    try:
        Path(path).resolve().relative_to(ALLOWED_ROOT.resolve())
        return True
    except (ValueError, OSError):
        return False


def decode(path: str) -> np.ndarray:
    """The whole file as float32 mono samples at SAMPLE_RATE."""
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path,
         "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "pipe:1"],
        capture_output=True, timeout=DECODE_TIMEOUT_S)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='replace')[:300]}")
    return np.frombuffer(result.stdout, dtype=np.float32)


def chunks(samples: np.ndarray, seconds: int | None = None) -> List[np.ndarray]:
    step = (seconds or CHUNK_SECONDS) * SAMPLE_RATE
    return [samples[i:i + step] for i in range(0, len(samples), step) if len(samples[i:i + step])]


async def transcribe(path: str, out_path: str,
                     transcribe_chunk: Callable[[np.ndarray], Awaitable[str]],
                     language: str | None = None) -> str:
    """Decode `path`, transcribe it chunk by chunk, write the text to
    `out_path` (atomically: the caller polls for that file)."""
    loop = asyncio.get_event_loop()
    samples = await loop.run_in_executor(None, decode, path)
    parts = chunks(samples)
    logger.info("Transcribing %s: %.0f s in %d chunks", path, len(samples) / SAMPLE_RATE, len(parts))
    texts = []
    for i, chunk in enumerate(parts, 1):
        text = (await transcribe_chunk(chunk) or "").strip()
        if text:
            texts.append(text)
        logger.debug("chunk %d/%d: %d chars", i, len(parts), len(text))
    joined = "\n\n".join(texts)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(joined, encoding="utf-8")
    os.replace(tmp, out)
    logger.info("Transcript written: %s (%d chars)", out_path, len(joined))
    return joined


def start(path: str, out_path: str,
          transcribe_chunk: Callable[[np.ndarray], Awaitable[str]]) -> str:
    """Start (or report) the job for `out_path`: 'started' or 'running'."""
    task = _running.get(out_path)
    if task is not None and not task.done():
        return "running"

    async def _job():
        try:
            await transcribe(path, out_path, transcribe_chunk)
        except Exception:
            logger.error("File transcription failed for %s", path, exc_info=True)
        finally:
            _running.pop(out_path, None)

    _running[out_path] = asyncio.create_task(_job())
    return "started"

#!/usr/bin/env python3
"""
gaia_transcribe.py — Long-form audio transcription via gaia-audio.

Transcribes full audio files (podcasts, recordings, analysis sessions) by
chunking with ffmpeg and sending each chunk to gaia-audio's /transcribe
endpoint. Optionally ingests the result into gaia-core as knowledge.

Dependencies (host-only):
  - requests (already installed for gaia_listener.py)
  - ffmpeg + ffprobe (already installed on host)

Usage:
  python scripts/gaia_transcribe.py /path/to/podcast.mp3
  python scripts/gaia_transcribe.py recording.m4a -o transcript.txt
  python scripts/gaia_transcribe.py analysis.mp3 --ingest --source "NotebookLM Deep Dive"
"""

import argparse
import base64
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_AUDIO_URL = "http://localhost:8080"
DEFAULT_CORE_URL = "http://localhost:6415"
DEFAULT_CHUNK_DURATION = 60   # seconds
DEFAULT_OVERLAP = 2           # seconds
DEFAULT_TIMEOUT = 120         # seconds per HTTP request

logger = logging.getLogger("gaia_transcribe")


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def get_audio_duration(path: str) -> float:
    """Use ffprobe to get total audio duration in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    duration_str = result.stdout.strip()
    if not duration_str:
        raise RuntimeError(f"ffprobe returned empty duration for {path}")
    return float(duration_str)


def extract_chunk_wav(path: str, start: float, duration: float) -> bytes:
    """Extract a chunk as 16kHz mono WAV using ffmpeg, returned as bytes.

    ffmpeg decodes, resamples, and outputs WAV to stdout in a single pass.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", path,
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg chunk extraction failed at {start:.1f}s: {result.stderr.decode(errors='replace').strip()}"
        )
    if not result.stdout:
        raise RuntimeError(f"ffmpeg produced empty output for chunk at {start:.1f}s")
    return result.stdout


# ---------------------------------------------------------------------------
# Chunk boundary computation
# ---------------------------------------------------------------------------

def compute_chunk_boundaries(
    total_duration: float,
    chunk_duration: int,
    overlap: int,
) -> list[tuple[float, float]]:
    """Compute (start, duration) pairs for overlapping chunks.

    Example with total=185, chunk=60, overlap=2:
      Chunk 0: (0, 60)     → covers [0, 60]
      Chunk 1: (58, 60)    → covers [58, 118]
      Chunk 2: (116, 60)   → covers [116, 176]
      Chunk 3: (174, 11)   → covers [174, 185]
    """
    if total_duration <= 0:
        return []

    chunks = []
    step = chunk_duration - overlap
    start = 0.0

    while start < total_duration:
        dur = min(chunk_duration, total_duration - start)
        # Skip trivially short tail chunks (< 0.5s)
        if dur < 0.5 and chunks:
            break
        chunks.append((start, dur))
        start += step

    return chunks


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_chunk(
    wav_bytes: bytes,
    audio_url: str,
    language: str | None,
    timeout: int,
) -> dict:
    """POST base64-encoded WAV to gaia-audio /transcribe, return response dict."""
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    payload = {
        "audio_base64": audio_b64,
        "sample_rate": 16000,
    }
    if language:
        payload["language"] = language

    resp = requests.post(
        f"{audio_url}/transcribe",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def transcribe_file(
    path: str,
    chunk_duration: int,
    overlap: int,
    audio_url: str,
    language: str | None,
    timeout: int,
    verbose: bool,
) -> tuple[str, dict]:
    """Transcribe a full audio file, returning (transcript, stats)."""
    total = get_audio_duration(path)
    chunks = compute_chunk_boundaries(total, chunk_duration, overlap)

    if not chunks:
        return "", {"duration_seconds": total, "chunks_processed": 0, "avg_confidence": 0.0}

    if verbose:
        mins, secs = divmod(int(total), 60)
        print(f"Audio duration: {mins}m {secs}s — {len(chunks)} chunks", file=sys.stderr)

    texts = []
    confidences = []

    for i, (start, dur) in enumerate(chunks):
        if verbose:
            pct = ((i + 1) / len(chunks)) * 100
            print(
                f"  [{i+1}/{len(chunks)}] {start:.1f}s–{start+dur:.1f}s ({pct:.0f}%)...",
                end="",
                flush=True,
                file=sys.stderr,
            )

        try:
            wav_bytes = extract_chunk_wav(path, start, dur)
            result = transcribe_chunk(wav_bytes, audio_url, language, timeout)

            text = result.get("text", "").strip()
            confidence = result.get("confidence", 0.0)
            latency = result.get("latency_ms", 0.0)

            if text:
                texts.append(text)
                confidences.append(confidence)

            if verbose:
                preview = text[:60] + "..." if len(text) > 60 else text
                print(
                    f" conf={confidence:.2f} latency={latency:.0f}ms — {preview}",
                    file=sys.stderr,
                )

        except requests.exceptions.HTTPError as e:
            logger.error("Chunk %d/%d transcription HTTP error: %s", i + 1, len(chunks), e)
            if verbose:
                print(f" ERROR: {e}", file=sys.stderr)
        except Exception as e:
            logger.error("Chunk %d/%d failed: %s", i + 1, len(chunks), e)
            if verbose:
                print(f" ERROR: {e}", file=sys.stderr)

    transcript = " ".join(texts)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    stats = {
        "duration_seconds": round(total, 2),
        "chunks_processed": len(chunks),
        "chunks_with_text": len(texts),
        "avg_confidence": round(avg_conf, 3),
    }

    if verbose:
        print(
            f"\nDone: {len(texts)}/{len(chunks)} chunks with speech, "
            f"avg confidence={avg_conf:.3f}",
            file=sys.stderr,
        )

    return transcript, stats


# ---------------------------------------------------------------------------
# gaia-core ingestion
# ---------------------------------------------------------------------------

def ingest_to_core(
    transcript: str,
    source_label: str,
    original_file: str,
    stats: dict,
    core_url: str,
):
    """Send transcript to gaia-core as an audio_transcription knowledge packet."""
    if not transcript.strip():
        logger.warning("Empty transcript — skipping ingestion")
        return

    dur = stats.get("duration_seconds", 0)
    mins, secs = divmod(int(dur), 60)
    duration_str = f"{mins}m {secs}s"

    packet = {
        "user_input": (
            f"[AUDIO TRANSCRIPTION]\n"
            f"Source: {source_label} — {time.strftime('%Y-%m-%d')}\n"
            f"Duration: {duration_str}\n\n"
            f"{transcript}"
        ),
        "metadata": {
            "source": "audio_transcription",
            "original_file": original_file,
            "source_label": source_label,
            "duration_seconds": stats.get("duration_seconds", 0),
            "chunks_processed": stats.get("chunks_processed", 0),
            "avg_confidence": stats.get("avg_confidence", 0),
            "packet_type": "audio_transcription",
        },
    }

    resp = requests.post(
        f"{core_url}/process_packet",
        json=packet,
        timeout=30,
    )
    if resp.status_code == 200:
        logger.info("Transcript ingested to gaia-core (%d chars)", len(transcript))
    else:
        logger.warning(
            "gaia-core returned %d: %s",
            resp.status_code,
            resp.text[:200],
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe a full audio file via gaia-audio.",
        epilog="Supported formats: anything ffmpeg can decode (MP3, WAV, FLAC, OGG, M4A, AAC, ...)",
    )
    parser.add_argument(
        "audio_file",
        help="Path to the audio file to transcribe",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--chunk-duration",
        type=int,
        default=DEFAULT_CHUNK_DURATION,
        help=f"Seconds per chunk (default: {DEFAULT_CHUNK_DURATION})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help=f"Seconds of overlap between chunks (default: {DEFAULT_OVERLAP})",
    )
    parser.add_argument(
        "--language",
        help="Language hint for Whisper (e.g., 'en', 'ja')",
    )
    parser.add_argument(
        "--audio-url",
        default=os.getenv("GAIA_AUDIO_URL", DEFAULT_AUDIO_URL),
        help=f"gaia-audio service URL (default: {DEFAULT_AUDIO_URL})",
    )
    parser.add_argument(
        "--core-url",
        default=os.getenv("GAIA_CORE_URL", DEFAULT_CORE_URL),
        help=f"gaia-core service URL (default: {DEFAULT_CORE_URL})",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Send transcript to gaia-core as knowledge",
    )
    parser.add_argument(
        "--source",
        help="Source label for transcript metadata (default: filename)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout per chunk in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-chunk progress and confidence",
    )
    args = parser.parse_args()

    # Validate input
    audio_path = Path(args.audio_file).resolve()
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [TRANSCRIBE] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    source_label = args.source or audio_path.name

    if args.verbose:
        print(f"Transcribing: {audio_path}", file=sys.stderr)
        print(f"Audio service: {args.audio_url}", file=sys.stderr)
        print(f"Chunk: {args.chunk_duration}s, overlap: {args.overlap}s", file=sys.stderr)

    # Transcribe
    t0 = time.monotonic()
    transcript, stats = transcribe_file(
        path=str(audio_path),
        chunk_duration=args.chunk_duration,
        overlap=args.overlap,
        audio_url=args.audio_url,
        language=args.language,
        timeout=args.timeout,
        verbose=args.verbose,
    )
    elapsed = time.monotonic() - t0

    if not transcript:
        print("No speech detected in audio file.", file=sys.stderr)
        sys.exit(0)

    if args.verbose:
        mins, secs = divmod(int(elapsed), 60)
        print(f"Transcription completed in {mins}m {secs}s", file=sys.stderr)
        print(f"Transcript length: {len(transcript)} chars", file=sys.stderr)

    # Output transcript
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(transcript, encoding="utf-8")
        if args.verbose:
            print(f"Saved to: {out_path}", file=sys.stderr)
    else:
        print(transcript)

    # Optionally ingest to gaia-core
    if args.ingest:
        if args.verbose:
            print(f"Ingesting to gaia-core ({args.core_url})...", file=sys.stderr)
        try:
            ingest_to_core(
                transcript=transcript,
                source_label=source_label,
                original_file=audio_path.name,
                stats=stats,
                core_url=args.core_url,
            )
        except Exception as e:
            logger.error("Ingestion failed: %s", e)
            print(f"Warning: ingestion to gaia-core failed: {e}", file=sys.stderr)

    # Summary line to stderr
    dur = stats.get("duration_seconds", 0)
    dur_mins, dur_secs = divmod(int(dur), 60)
    print(
        f"[{dur_mins}m {dur_secs}s audio → {len(transcript)} chars, "
        f"{stats['chunks_with_text']}/{stats['chunks_processed']} chunks, "
        f"conf={stats['avg_confidence']:.3f}]",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

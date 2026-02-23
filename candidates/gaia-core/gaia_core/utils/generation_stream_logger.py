"""Thread-safe JSONL logger for real-time generation visibility.

Writes token-level events to ``/logs/generation_stream.jsonl`` so that
external consumers (``tail -f``, Mission Control SSE) can observe
inference as it happens — including ``<think>`` blocks.

Usage::

    from gaia_core.utils.generation_stream_logger import get_logger
    gen = get_logger()
    gen_id = gen.start_generation("gpu_prime", "prime", "response")
    gen.log_token(gen_id, "<think>")
    gen.log_token(gen_id, "Let me consider...")
    gen.end_generation(gen_id)
"""

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

_LOG_DIR = os.getenv("GAIA_GENERATION_LOG_DIR", "/logs")
_LOG_PATH = os.path.join(_LOG_DIR, "generation_stream.jsonl")
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB rotation threshold


class GenerationStreamLogger:
    """Singleton JSONL writer for generation token events."""

    def __init__(self, path: str = _LOG_PATH, max_bytes: int = _MAX_BYTES):
        self._path = path
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        self._generations: dict[str, dict] = {}  # gen_id -> metadata
        # Ensure the directory exists
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def start_generation(self, model: str, role: str, phase: str) -> str:
        """Begin tracking a new generation. Returns a unique ``gen_id``."""
        gen_id = uuid.uuid4().hex[:12]
        now = time.time()
        self._generations[gen_id] = {
            "model": model,
            "role": role,
            "phase": phase,
            "t0": now,
            "tokens": 0,
        }
        self._write({
            "ts": _iso(now),
            "event": "gen_start",
            "gen_id": gen_id,
            "model": model,
            "role": role,
            "phase": phase,
        })
        return gen_id

    def log_token(self, gen_id: str, text: str) -> None:
        """Append a single token event."""
        meta = self._generations.get(gen_id)
        if meta:
            meta["tokens"] += 1
        self._write({
            "ts": _iso(time.time()),
            "event": "token",
            "gen_id": gen_id,
            "t": text,
        })

    def end_generation(self, gen_id: str) -> None:
        """Finalise a generation with summary stats."""
        meta = self._generations.pop(gen_id, None)
        now = time.time()
        elapsed_ms = int((now - meta["t0"]) * 1000) if meta else 0
        tokens = meta["tokens"] if meta else 0
        self._write({
            "ts": _iso(now),
            "event": "gen_end",
            "gen_id": gen_id,
            "tokens": tokens,
            "elapsed_ms": elapsed_ms,
        })

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _write(self, record: dict) -> None:
        line = json.dumps(record, separators=(",", ":")) + "\n"
        with self._lock:
            self._maybe_rotate()
            try:
                with open(self._path, "a") as f:
                    f.write(line)
            except OSError:
                logger.debug("generation_stream_logger: write failed", exc_info=True)

    def _maybe_rotate(self) -> None:
        try:
            if os.path.getsize(self._path) >= self._max_bytes:
                rotated = self._path + ".1"
                if os.path.exists(rotated):
                    os.remove(rotated)
                os.rename(self._path, rotated)
        except OSError:
            pass  # file may not exist yet


# ------------------------------------------------------------------ #
# Module-level singleton
# ------------------------------------------------------------------ #

_instance: GenerationStreamLogger | None = None
_instance_lock = threading.Lock()


def get_logger() -> GenerationStreamLogger:
    """Return the shared ``GenerationStreamLogger`` singleton."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = GenerationStreamLogger()
    return _instance


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _iso(ts: float) -> str:
    """Compact ISO-8601 UTC timestamp."""
    import datetime
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )[:-3] + "Z"

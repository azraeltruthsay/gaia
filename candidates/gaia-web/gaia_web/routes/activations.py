"""
Activation stream routes for Mission Control.

SSE endpoint that tails ``/logs/activation_stream.jsonl`` in real-time,
giving the Neural Mind Map live visibility into per-token SAE feature
activations as Nano, Core, and Prime generate.

Also serves the SAE atlas (feature labels) for human-readable
visualization.
"""

import asyncio
import json
import logging
import os
import time

from fastapi import APIRouter
from starlette.responses import StreamingResponse

logger = logging.getLogger("GAIA.Web.Activations")

router = APIRouter(prefix="/api/activations", tags=["activations"])

_LOG_PATH = os.getenv("ACTIVATION_STREAM_PATH", "/logs/activation_stream.jsonl")
_ATLAS_DIR = os.getenv("SAE_ATLAS_DIR", "/shared/atlas/core")
_POLL_INTERVAL = 0.05  # 50 ms
_HEARTBEAT_INTERVAL = 2.0  # seconds


@router.get("/stream")
async def activation_stream(session_id: str = ""):
    """SSE endpoint — tails activation_stream.jsonl in real-time.

    Optional query params:
      - ``session_id`` — filter to a specific session's activations
    """
    return StreamingResponse(
        _event_generator(session_id=session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _event_generator(session_id: str = ""):
    """Async generator that tails the activation JSONL log and yields SSE frames."""
    last_heartbeat = 0.0
    file_pos = 0

    # Start from the end of the file (don't replay old events)
    try:
        file_pos = os.path.getsize(_LOG_PATH)
    except OSError:
        pass

    while True:
        lines_sent = False
        try:
            with open(_LOG_PATH, "r") as f:
                # Handle file rotation: if file shrank, start from beginning
                current_size = os.fstat(f.fileno()).st_size
                if current_size < file_pos:
                    file_pos = 0
                f.seek(file_pos)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Filter by session_id if provided
                    if session_id and record.get("session_id", "") != session_id:
                        continue
                    yield f"data: {json.dumps(record)}\n\n"
                    lines_sent = True
                file_pos = f.tell()
        except OSError:
            pass  # file doesn't exist yet — wait for it

        now = time.monotonic()
        if not lines_sent and (now - last_heartbeat) >= _HEARTBEAT_INTERVAL:
            yield ":keepalive\n\n"
            last_heartbeat = now

        await asyncio.sleep(_POLL_INTERVAL)


@router.get("/atlas")
async def get_atlas():
    """Return SAE feature labels for the mind map.

    Reads from ``/shared/atlas/core/meta.json`` if it exists, plus
    any per-layer feature label files.  Returns a structured dict
    that the D3 mind map uses to label activation nodes.
    """
    result = {"layers": {}, "model": None, "timestamp": None}

    meta_path = os.path.join(_ATLAS_DIR, "meta.json")
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        result["model"] = meta.get("model")
        result["timestamp"] = meta.get("timestamp")

        # meta.json may contain inline feature labels
        if "layers" in meta:
            result["layers"] = meta["layers"]
    except (OSError, json.JSONDecodeError):
        pass

    # Also check for per-layer label files (layer_N_labels.json)
    try:
        for entry in os.listdir(_ATLAS_DIR):
            if entry.startswith("layer_") and entry.endswith("_labels.json"):
                try:
                    layer_idx = int(entry.split("_")[1])
                    with open(os.path.join(_ATLAS_DIR, entry), "r") as f:
                        labels = json.load(f)
                    # Merge — per-layer files override meta.json
                    layer_key = str(layer_idx)
                    if layer_key not in result["layers"]:
                        result["layers"][layer_key] = {"features": {}}
                    result["layers"][layer_key]["features"].update(labels)
                except (ValueError, json.JSONDecodeError, OSError):
                    continue
    except OSError:
        pass

    return result

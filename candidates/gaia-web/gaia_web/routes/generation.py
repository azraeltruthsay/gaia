"""
Generation stream routes for Mission Control.

SSE endpoint that tails ``/logs/generation_stream.jsonl`` in real-time,
giving the UI (and ``curl``) live visibility into every token — including
``<think>`` blocks — as Prime and Lite generate.
"""

import asyncio
import json
import logging
import os

from fastapi import APIRouter
from starlette.responses import StreamingResponse

logger = logging.getLogger("GAIA.Web.Generation")

router = APIRouter(prefix="/api/generation", tags=["generation"])

_LOG_PATH = os.getenv("GAIA_GENERATION_LOG_PATH", "/logs/generation_stream.jsonl")
_POLL_INTERVAL = 0.05  # 50 ms
_HEARTBEAT_INTERVAL = 5.0  # seconds


@router.get("/stream")
async def generation_stream(role: str = "", gen_id: str = ""):
    """SSE endpoint — tails generation_stream.jsonl in real-time.

    Optional query params:
      - ``role`` — filter to only prime / lite events
      - ``gen_id`` — filter to a specific generation
    """
    return StreamingResponse(
        _event_generator(role=role, gen_id=gen_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _event_generator(role: str = "", gen_id: str = ""):
    """Async generator that tails the JSONL log file and yields SSE frames."""
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
                    # Apply filters
                    if role and record.get("role", "") != role:
                        # For token/gen_end events, check via gen_id tracking
                        # gen_start has role directly; others inherit
                        if record.get("event") == "gen_start":
                            continue
                    if gen_id and record.get("gen_id", "") != gen_id:
                        continue
                    yield f"data: {json.dumps(record)}\n\n"
                    lines_sent = True
                file_pos = f.tell()
        except OSError:
            pass  # file doesn't exist yet — wait for it

        import time
        now = time.monotonic()
        if not lines_sent and (now - last_heartbeat) >= _HEARTBEAT_INTERVAL:
            yield ":keepalive\n\n"
            last_heartbeat = now

        await asyncio.sleep(_POLL_INTERVAL)

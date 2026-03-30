"""
Autonomous message stream for Mission Control.

SSE endpoint that tails ``/logs/autonomous_messages.jsonl`` in real-time,
delivering GAIA's unsolicited commentary (audio listening reactions,
initiative-driven messages, etc.) to the dashboard chat panel.
"""

import asyncio
import json
import logging
import os
import time

from fastapi import APIRouter
from starlette.responses import StreamingResponse

logger = logging.getLogger("GAIA.Web.Autonomous")

router = APIRouter(prefix="/api/autonomous", tags=["autonomous"])

_LOG_PATH = os.getenv("GAIA_AUTONOMOUS_LOG_PATH", "/logs/autonomous_messages.jsonl")
_POLL_INTERVAL = 0.25  # 250 ms (lower frequency than generation stream)
_HEARTBEAT_INTERVAL = 15.0  # seconds


@router.get("/stream")
async def autonomous_stream():
    """SSE endpoint — tails autonomous_messages.jsonl in real-time."""
    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _event_generator():
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

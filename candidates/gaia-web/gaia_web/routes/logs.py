"""
Service log streaming routes for Mission Control.

SSE endpoint that tails service log files in real-time, giving the
Logs tab live visibility into gaia-core, gaia-web, gaia-mcp, gaia-study,
and Discord bot logs.
"""

import asyncio
import json
import logging
import os
import time

from fastapi import APIRouter, Query
from starlette.responses import StreamingResponse

logger = logging.getLogger("GAIA.Web.Logs")

router = APIRouter(prefix="/api/logs", tags=["logs"])

_POLL_INTERVAL = 0.05  # 50 ms
_HEARTBEAT_INTERVAL = 5.0  # seconds

# Map service names to log file paths
_SERVICE_LOG_MAP = {
    "core": os.getenv("GAIA_CORE_LOG_PATH", "/logs/gaia-core.log"),
    "web": os.getenv("GAIA_WEB_LOG_PATH", "/logs/gaia-web.log"),
    "mcp": os.getenv("GAIA_MCP_LOG_PATH", "/logs/gaia-mcp.log"),
    "study": os.getenv("GAIA_STUDY_LOG_PATH", "/logs/gaia-study.log"),
    "discord": os.getenv("GAIA_DISCORD_LOG_PATH", "/logs/discord_bot.log"),
}


@router.get("/services")
async def list_services():
    """Return available log services and their file status."""
    services = []
    for name, path in _SERVICE_LOG_MAP.items():
        exists = os.path.isfile(path)
        size = os.path.getsize(path) if exists else 0
        services.append({"name": name, "path": path, "exists": exists, "size_bytes": size})
    return services


@router.get("/stream")
async def log_stream(service: str = Query(..., description="Service name: core, web, mcp, study, discord")):
    """SSE endpoint -- tails a service log file in real-time."""
    if service not in _SERVICE_LOG_MAP:
        from starlette.responses import JSONResponse
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown service '{service}'. Available: {list(_SERVICE_LOG_MAP.keys())}"},
        )
    log_path = _SERVICE_LOG_MAP[service]
    return StreamingResponse(
        _tail_generator(log_path, service),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _tail_generator(log_path: str, service: str):
    """Async generator that tails a log file and yields SSE frames."""
    last_heartbeat = 0.0
    file_pos = 0

    # Start from the end of the file (don't replay old events)
    try:
        file_pos = os.path.getsize(log_path)
    except OSError:
        pass

    while True:
        lines_sent = False
        try:
            with open(log_path, "r") as f:
                # Handle file rotation: if file shrank, start from beginning
                current_size = os.fstat(f.fileno()).st_size
                if current_size < file_pos:
                    file_pos = 0
                f.seek(file_pos)
                for line in f:
                    text = line.rstrip("\n")
                    if not text:
                        continue
                    # Detect severity level from common log format patterns
                    level = _detect_level(text)
                    payload = json.dumps({"text": text, "service": service, "level": level})
                    yield f"data: {payload}\n\n"
                    lines_sent = True
                file_pos = f.tell()
        except OSError:
            pass  # file doesn't exist yet -- wait for it

        now = time.monotonic()
        if not lines_sent and (now - last_heartbeat) >= _HEARTBEAT_INTERVAL:
            yield ":keepalive\n\n"
            last_heartbeat = now

        await asyncio.sleep(_POLL_INTERVAL)


def _detect_level(text: str) -> str:
    """Best-effort severity detection from log line text."""
    upper = text[:120].upper()
    if "ERROR" in upper or "CRITICAL" in upper or "EXCEPTION" in upper:
        return "error"
    if "WARNING" in upper or "WARN" in upper:
        return "warning"
    if "DEBUG" in upper:
        return "debug"
    return "info"

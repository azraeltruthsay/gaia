"""
Service log streaming routes for Mission Control.

SSE endpoint that tails service log files in real-time, giving the
Logs tab live visibility into gaia-core, gaia-web, gaia-mcp, gaia-study,
and Discord bot logs.

Also provides a search endpoint for querying historical logs.
"""

import asyncio
import json
import logging
import os
import re
import time
from collections import deque

from fastapi import APIRouter, Query
from starlette.responses import StreamingResponse

logger = logging.getLogger("GAIA.Web.Logs")

router = APIRouter(prefix="/api/logs", tags=["logs"])

_POLL_INTERVAL = 0.05  # 50 ms
_HEARTBEAT_INTERVAL = 5.0  # seconds
_BACKLOG_LINES = 200  # lines to send on initial connect
_SEARCH_MAX_RESULTS = 500

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
async def log_stream(
    service: str = Query(..., description="Service name: core, web, mcp, study, discord"),
    backlog: int = Query(_BACKLOG_LINES, ge=0, le=2000, description="Initial lines to send"),
):
    """SSE endpoint -- sends recent backlog then tails a service log file in real-time."""
    if service not in _SERVICE_LOG_MAP:
        from starlette.responses import JSONResponse
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown service '{service}'. Available: {list(_SERVICE_LOG_MAP.keys())}"},
        )
    log_path = _SERVICE_LOG_MAP[service]
    return StreamingResponse(
        _tail_generator(log_path, service, backlog),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/search")
async def log_search(
    service: str = Query(..., description="Service name"),
    q: str = Query(..., min_length=1, description="Search query (substring or regex)"),
    level: str = Query("", description="Filter by level: error, warning, info, debug"),
    limit: int = Query(_SEARCH_MAX_RESULTS, ge=1, le=2000, description="Max results"),
    tail: int = Query(0, ge=0, description="Only search the last N lines (0 = all)"),
    regex: bool = Query(False, description="Treat query as regex"),
):
    """Search historical logs with optional level and regex filtering."""
    if service not in _SERVICE_LOG_MAP:
        from starlette.responses import JSONResponse
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown service '{service}'. Available: {list(_SERVICE_LOG_MAP.keys())}"},
        )

    log_path = _SERVICE_LOG_MAP[service]
    if not os.path.isfile(log_path):
        return {"service": service, "query": q, "results": [], "total": 0}

    pattern = None
    if regex:
        try:
            pattern = re.compile(q, re.IGNORECASE)
        except re.error as e:
            from starlette.responses import JSONResponse
            return JSONResponse(status_code=400, content={"error": f"Invalid regex: {e}"})

    q_lower = q.lower()
    level_filter = level.lower() if level else ""
    results: deque = deque(maxlen=limit)

    # Read lines — if tail > 0, only scan the last N lines
    lines = _read_tail(log_path, tail) if tail > 0 else _read_all_lines(log_path)

    for line_num, raw_line in lines:
        text = raw_line.rstrip("\n")
        if not text:
            continue
        detected_level = _detect_level(text)
        if level_filter and detected_level != level_filter:
            continue
        if pattern:
            if not pattern.search(text):
                continue
        elif q_lower not in text.lower():
            continue
        results.append({"line": line_num, "text": text, "level": detected_level})

    return {
        "service": service,
        "query": q,
        "level_filter": level_filter or None,
        "total": len(results),
        "results": list(results),
    }


def _read_all_lines(path: str):
    """Yield (line_number, line_text) for all lines in a file."""
    try:
        with open(path, "r", errors="replace") as f:
            for i, line in enumerate(f, 1):
                yield i, line
    except OSError:
        return


def _read_tail(path: str, n: int):
    """Yield (line_number, line_text) for the last N lines of a file."""
    try:
        with open(path, "r", errors="replace") as f:
            # Count total lines first via a fast pass
            total = 0
            for total, _ in enumerate(f, 1):
                pass
            f.seek(0)
            start = max(1, total - n + 1)
            for i, line in enumerate(f, 1):
                if i >= start:
                    yield i, line
    except OSError:
        return


def _read_last_n_lines(path: str, n: int) -> list[str]:
    """Efficiently read the last N lines from a file using a ring buffer."""
    buf: deque[str] = deque(maxlen=n)
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                buf.append(line)
    except OSError:
        pass
    return list(buf)


async def _tail_generator(log_path: str, service: str, backlog: int):
    """Async generator that sends backlog then tails a log file, yielding SSE frames."""
    last_heartbeat = 0.0
    file_pos = 0

    # Send initial backlog so the UI isn't empty on connect
    if backlog > 0:
        initial_lines = _read_last_n_lines(log_path, backlog)
        for raw_line in initial_lines:
            text = raw_line.rstrip("\n")
            if not text:
                continue
            level = _detect_level(text)
            payload = json.dumps({"text": text, "service": service, "level": level})
            yield f"data: {payload}\n\n"

    # Start tailing from the current end of file
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

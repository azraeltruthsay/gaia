"""
Terminal routes for Mission Control dashboard.

Provides container listing and WebSocket-based Docker exec bridge
for interactive shell access to GAIA containers.
"""

import asyncio
import logging

import docker
from docker.errors import DockerException, NotFound
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

logger = logging.getLogger("GAIA.Web.Terminal")

router = APIRouter(prefix="/api/terminal", tags=["terminal"])

CONTAINER_PREFIX = "gaia-"


def _get_client():
    """Get Docker client, raising 503 if unavailable."""
    try:
        return docker.from_env()
    except DockerException as e:
        logger.error(f"Docker unavailable: {e}")
        return None


# ── Container Listing ───────────────────────────────────────────────────────

@router.get("/containers")
async def list_containers():
    """List gaia-* containers with status."""
    client = _get_client()
    if not client:
        return JSONResponse(status_code=503, content={"error": "Docker unavailable"})

    try:
        containers = client.containers.list(all=True)
        result = []
        for c in containers:
            if not c.name.startswith(CONTAINER_PREFIX):
                continue
            result.append({
                "name": c.name,
                "status": c.status,
                "image": c.image.tags[0] if c.image.tags else str(c.image.id)[:12],
            })
        result.sort(key=lambda c: c["name"])
        return result
    except DockerException as e:
        return JSONResponse(status_code=503, content={"error": str(e)})
    finally:
        client.close()


# ── WebSocket Exec Bridge ───────────────────────────────────────────────────

@router.websocket("/ws")
async def terminal_ws(ws: WebSocket, container: str = ""):
    """Bidirectional Docker exec bridge via WebSocket."""
    await ws.accept()

    if not container or not container.startswith(CONTAINER_PREFIX):
        await ws.send_json({"error": f"Invalid container name: {container}"})
        await ws.close(code=1008)
        return

    client = _get_client()
    if not client:
        await ws.send_json({"error": "Docker unavailable"})
        await ws.close(code=1011)
        return

    try:
        c = client.containers.get(container)
        if c.status != "running":
            await ws.send_json({"error": f"Container {container} is not running (status: {c.status})"})
            await ws.close(code=1011)
            return
    except NotFound:
        await ws.send_json({"error": f"Container {container} not found"})
        await ws.close(code=1011)
        return
    except DockerException as e:
        await ws.send_json({"error": str(e)})
        await ws.close(code=1011)
        return

    # Create exec instance
    try:
        exec_id = client.api.exec_create(
            container,
            cmd="/bin/bash",
            stdin=True,
            tty=True,
            environment={"TERM": "xterm-256color"},
        )
        sock = client.api.exec_start(exec_id, socket=True, tty=True)
        raw_sock = sock._sock  # noqa: SLF001 — access underlying socket
    except DockerException as e:
        await ws.send_json({"error": f"Exec failed: {e}"})
        await ws.close(code=1011)
        client.close()
        return

    loop = asyncio.get_event_loop()

    async def read_from_docker():
        """Forward Docker exec output → WebSocket."""
        try:
            while True:
                data = await loop.run_in_executor(None, raw_sock.recv, 4096)
                if not data:
                    break
                await ws.send_bytes(data)
        except (OSError, WebSocketDisconnect):
            pass

    async def write_to_docker():
        """Forward WebSocket input → Docker exec stdin."""
        try:
            while True:
                data = await ws.receive_bytes()
                await loop.run_in_executor(None, raw_sock.sendall, data)
        except (WebSocketDisconnect, RuntimeError):
            pass

    try:
        done, pending = await asyncio.wait(
            [asyncio.create_task(read_from_docker()), asyncio.create_task(write_to_docker())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
    finally:
        try:
            raw_sock.close()
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass
        client.close()
        try:
            await ws.close()
        except Exception:
            pass

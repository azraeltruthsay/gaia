"""
Hook/Command proxy routes for Mission Control dashboard.

Proxies sleep/wake, GPU management, and semantic codex operations
from the browser to gaia-core, avoiding CORS issues.
"""

import os
import logging

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("GAIA.Web.Hooks")

CORE_ENDPOINT = os.environ.get("CORE_ENDPOINT", "http://gaia-core-candidate:6415")

router = APIRouter(prefix="/api/hooks", tags=["hooks"])


# ── Sleep Control ────────────────────────────────────────────────────────────

@router.get("/sleep/status")
async def sleep_status():
    """Proxy GET gaia-core /sleep/status."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{CORE_ENDPOINT}/sleep/status")
            return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.ConnectError:
        return JSONResponse(status_code=503, content={"error": "gaia-core unreachable"})
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


@router.post("/sleep/wake")
async def sleep_wake():
    """Proxy POST gaia-core /sleep/wake."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{CORE_ENDPOINT}/sleep/wake")
            return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.ConnectError:
        return JSONResponse(status_code=503, content={"error": "gaia-core unreachable"})
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


@router.post("/sleep/shutdown")
async def sleep_shutdown():
    """Proxy POST gaia-core /sleep/shutdown."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{CORE_ENDPOINT}/sleep/shutdown")
            return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.ConnectError:
        return JSONResponse(status_code=503, content={"error": "gaia-core unreachable"})
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


# ── GPU Management ───────────────────────────────────────────────────────────

@router.get("/gpu/status")
async def gpu_status():
    """Proxy GET gaia-core /gpu/status."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{CORE_ENDPOINT}/gpu/status")
            return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.ConnectError:
        return JSONResponse(status_code=503, content={"error": "gaia-core unreachable"})
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


@router.post("/gpu/release")
async def gpu_release():
    """Proxy POST gaia-core /gpu/release."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{CORE_ENDPOINT}/gpu/release")
            return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.ConnectError:
        return JSONResponse(status_code=503, content={"error": "gaia-core unreachable"})
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


@router.post("/gpu/reclaim")
async def gpu_reclaim():
    """Proxy POST gaia-core /gpu/reclaim."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{CORE_ENDPOINT}/gpu/reclaim")
            return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.ConnectError:
        return JSONResponse(status_code=503, content={"error": "gaia-core unreachable"})
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


# ── Semantic Codex Search ────────────────────────────────────────────────────

class CodexSearchRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("/codex/search")
async def codex_search(req: CodexSearchRequest):
    """Proxy POST gaia-core /codex/search for semantic vector lookups."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{CORE_ENDPOINT}/codex/search",
                json={"query": req.query, "top_k": req.top_k},
            )
            return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.ConnectError:
        return JSONResponse(status_code=503, content={"error": "gaia-core unreachable"})
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})

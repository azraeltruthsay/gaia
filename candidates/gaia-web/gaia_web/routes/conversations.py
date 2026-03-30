"""
Conversation management routes for multi-conversation chat.

Proxies CRUD operations and context pool actions to gaia-core's
session management endpoints.
"""

import logging
import os

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("GAIA.Web.Conversations")

router = APIRouter()

CORE_URL = os.getenv("CORE_ENDPOINT", "http://gaia-core:6415")


@router.get("/")
async def list_conversations():
    """List all conversations with metadata."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{CORE_URL}/api/sessions")
            if resp.status_code == 200:
                return resp.json()
            return JSONResponse(
                status_code=resp.status_code,
                content={"error": f"Core returned {resp.status_code}"},
            )
    except httpx.ConnectError:
        logger.warning("Cannot reach gaia-core for session list")
        return JSONResponse(status_code=502, content={"error": "gaia-core unreachable"})
    except Exception as e:
        logger.error("list_conversations failed: %s", e)
        return JSONResponse(status_code=502, content={"error": str(e)})


@router.post("/")
async def create_conversation(request: Request):
    """Create a new conversation session."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{CORE_URL}/api/sessions",
                json=body,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code == 200:
                return resp.json()
            return JSONResponse(
                status_code=resp.status_code,
                content={"error": f"Core returned {resp.status_code}"},
            )
    except httpx.ConnectError:
        logger.warning("Cannot reach gaia-core to create session")
        return JSONResponse(status_code=502, content={"error": "gaia-core unreachable"})
    except Exception as e:
        logger.error("create_conversation failed: %s", e)
        return JSONResponse(status_code=502, content={"error": str(e)})


@router.delete("/{session_id}")
async def delete_conversation(session_id: str):
    """Archive and delete a conversation."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.delete(f"{CORE_URL}/api/sessions/{session_id}")
            if resp.status_code == 200:
                return resp.json()
            return JSONResponse(
                status_code=resp.status_code,
                content={"error": f"Core returned {resp.status_code}"},
            )
    except httpx.ConnectError:
        logger.warning("Cannot reach gaia-core to delete session %s", session_id)
        return JSONResponse(status_code=502, content={"error": "gaia-core unreachable"})
    except Exception as e:
        logger.error("delete_conversation failed: %s", e)
        return JSONResponse(status_code=502, content={"error": str(e)})


@router.put("/{session_id}/title")
async def rename_conversation(session_id: str, request: Request):
    """Rename a conversation."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.put(
                f"{CORE_URL}/api/sessions/{session_id}/meta",
                json=body,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code == 200:
                return resp.json()
            return JSONResponse(
                status_code=resp.status_code,
                content={"error": f"Core returned {resp.status_code}"},
            )
    except httpx.ConnectError:
        logger.warning("Cannot reach gaia-core to rename session %s", session_id)
        return JSONResponse(status_code=502, content={"error": "gaia-core unreachable"})
    except Exception as e:
        logger.error("rename_conversation failed: %s", e)
        return JSONResponse(status_code=502, content={"error": str(e)})


@router.get("/{session_id}/history")
async def get_history(session_id: str):
    """Get full message history for context joining."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{CORE_URL}/api/sessions/{session_id}/history")
            if resp.status_code == 200:
                return resp.json()
            return JSONResponse(
                status_code=resp.status_code,
                content={"error": f"Core returned {resp.status_code}"},
            )
    except httpx.ConnectError:
        logger.warning("Cannot reach gaia-core for session %s history", session_id)
        return JSONResponse(status_code=502, content={"error": "gaia-core unreachable"})
    except Exception as e:
        logger.error("get_history failed: %s", e)
        return JSONResponse(status_code=502, content={"error": str(e)})


@router.post("/{session_id}/pool-summary")
async def generate_pool_summary(session_id: str):
    """Generate a context pool summary for this conversation."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{CORE_URL}/api/sessions/{session_id}/summary")
            if resp.status_code == 200:
                return resp.json()
            return JSONResponse(
                status_code=resp.status_code,
                content={"error": f"Core returned {resp.status_code}"},
            )
    except httpx.ConnectError:
        logger.warning("Cannot reach gaia-core for session %s summary", session_id)
        return JSONResponse(status_code=502, content={"error": "gaia-core unreachable"})
    except Exception as e:
        logger.error("generate_pool_summary failed: %s", e)
        return JSONResponse(status_code=502, content={"error": str(e)})

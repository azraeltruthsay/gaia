"""
Hook/Command proxy routes for Mission Control dashboard.

Proxies sleep/wake, GPU management, and semantic codex operations
from the browser to gaia-core, avoiding CORS issues.
"""

import json
import os
import logging
import uuid

import httpx
from fastapi import APIRouter, Request
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


@router.post("/sleep/toggle")
async def sleep_toggle(request: Request):
    """Proxy POST gaia-core /sleep/toggle."""
    try:
        body = await request.json()
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{CORE_ENDPOINT}/sleep/toggle", json=body)
            return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.ConnectError:
        return JSONResponse(status_code=503, content={"error": "gaia-core unreachable"})
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


@router.post("/sleep/force")
async def sleep_force():
    """Proxy POST gaia-core /sleep/force."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{CORE_ENDPOINT}/sleep/force")
            return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.ConnectError:
        return JSONResponse(status_code=503, content={"error": "gaia-core unreachable"})
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


@router.get("/sleep/config")
async def sleep_config():
    """Proxy GET gaia-core /sleep/config."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{CORE_ENDPOINT}/sleep/config")
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


# ── Discord E2E Test ─────────────────────────────────────────────────────────

class DiscordTestRequest(BaseModel):
    message: str
    user_id: str = "e2e_test_user"
    session_id: str = ""


@router.post("/discord/test")
async def discord_test(req: DiscordTestRequest):
    """Simulate a Discord DM through gaia-core and return what would be sent.

    Builds a CognitionPacket matching the Discord handler's format, streams
    through /process_packet, and parses the NDJSON exactly like the Discord
    bot does. Returns the list of messages that would be sent to Discord.
    """
    session_id = req.session_id or f"discord_dm_{req.user_id}"
    packet_id = f"e2e_{uuid.uuid4().hex[:8]}"

    packet = {
        "version": "v0.3",
        "header": {
            "session_id": session_id,
            "packet_id": packet_id,
            "persona": {"persona_id": "gaia", "role": "assistant"},
            "origin": "user",
            "output_routing": {
                "primary": {
                    "destination": "discord",
                    "channel_id": "e2e_test",
                    "user_id": req.user_id,
                    "metadata": {"is_dm": True, "source": "discord"},
                },
                "addressed_to_gaia": True,
                "source_destination": "discord",
            },
        },
        "content": {"original_prompt": req.message},
        "governance": {
            "safety": {"execution_allowed": False, "dry_run": True},
        },
    }

    discord_messages = []
    raw_events = []

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            full_response = ""
            async with client.stream(
                "POST",
                f"{CORE_ENDPOINT}/process_packet",
                json=packet,
                timeout=180.0,
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    return JSONResponse(
                        status_code=response.status_code,
                        content={"error": f"Core returned {response.status_code}"},
                    )

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                        event_type = event.get("type")
                        raw_events.append({"type": event_type, "preview": str(event.get("value", ""))[:200]})

                        if event_type == "token":
                            val = event.get("value", "")
                            if val:
                                if val.startswith("⚡ **[(Reflex)"):
                                    discord_messages.append({"type": "reflex", "content": val})
                                else:
                                    full_response += val

                        elif event_type == "flush":
                            if full_response.strip():
                                discord_messages.append({"type": "flush", "content": full_response.strip()})
                                full_response = ""

                        elif event_type == "error":
                            discord_messages.append({"type": "error", "content": event.get("value", "")})

                    except json.JSONDecodeError:
                        pass

            # Final accumulated response (same as Discord handler)
            if full_response.strip():
                discord_messages.append({"type": "final", "content": full_response.strip()})

    except httpx.ConnectError:
        return JSONResponse(status_code=503, content={"error": "gaia-core unreachable"})
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})

    return {
        "discord_messages": discord_messages,
        "message_count": len(discord_messages),
        "raw_event_count": len(raw_events),
        "raw_events": raw_events,
        "packet_id": packet_id,
        "session_id": session_id,
    }


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

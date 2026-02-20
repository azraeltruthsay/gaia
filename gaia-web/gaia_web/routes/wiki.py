"""
Optional proxy route for gaia-wiki.
Proxies /wiki/* -> http://gaia-wiki:8080/*
Only active when WIKI_ENDPOINT is set.
"""

import os
import logging

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response

logger = logging.getLogger("GAIA.Web.Wiki")

WIKI_ENDPOINT = os.environ.get("WIKI_ENDPOINT", "")

router = APIRouter()


@router.api_route("/wiki/{path:path}", methods=["GET", "HEAD"])
async def wiki_proxy(request: Request, path: str):
    """Proxy requests to gaia-wiki. Only active if WIKI_ENDPOINT is set."""
    if not WIKI_ENDPOINT:
        return Response(
            content="Wiki proxy not configured (set WIKI_ENDPOINT)",
            status_code=503,
        )

    target = f"{WIKI_ENDPOINT}/{path}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.request(
                method=request.method,
                url=target,
                headers={
                    k: v for k, v in request.headers.items()
                    if k.lower() not in ("host", "transfer-encoding")
                },
            )
            # Filter hop-by-hop headers from the response
            excluded = {"transfer-encoding", "connection", "keep-alive"}
            headers = {
                k: v for k, v in resp.headers.items()
                if k.lower() not in excluded
            }
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=headers,
            )
    except httpx.ConnectError:
        return Response(content="gaia-wiki unreachable", status_code=503)
    except Exception as exc:
        logger.warning("Wiki proxy error: %s", exc)
        return Response(content="Wiki proxy error", status_code=502)

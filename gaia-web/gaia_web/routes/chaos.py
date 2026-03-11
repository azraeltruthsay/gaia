"""
Chaos proxy routes — thin proxy to gaia-monkey:6420.
"""
import logging
import os

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("GAIA.Web.Chaos")

router = APIRouter()

MONKEY_URL = os.getenv("MONKEY_ENDPOINT", "http://gaia-monkey:6420")


async def _proxy_get(path: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{MONKEY_URL}{path}")
            return resp.json()
    except Exception as e:
        logger.debug("Monkey GET %s failed: %s", path, e)
        return {"error": str(e)}


async def _proxy_post(path: str, body: dict | None = None, timeout: float = 180.0) -> dict:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{MONKEY_URL}{path}", json=body or {})
            return resp.json()
    except Exception as e:
        logger.debug("Monkey POST %s failed: %s", path, e)
        return {"error": str(e)}


@router.get("/status")
async def chaos_status():
    return await _proxy_get("/status")


@router.get("/config")
async def chaos_get_config():
    return await _proxy_get("/config")


@router.post("/config")
async def chaos_set_config(request: Request):
    body = await request.json()
    return await _proxy_post("/config", body)


@router.post("/inject")
async def chaos_inject(request: Request):
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    return await _proxy_post("/chaos/inject", body)


@router.get("/history")
async def chaos_history():
    return await _proxy_get("/chaos/history")


@router.get("/serenity")
async def chaos_serenity():
    return await _proxy_get("/serenity")

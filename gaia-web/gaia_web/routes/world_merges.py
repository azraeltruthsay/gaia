"""World Model merge-approval proxy (GAIA_Project-21h Phase 2).

The gaia-study container exposes the actual approval API (Stage 5c
Phase 1, commit 9c75d0d) at:
  GET  /world_merges/pending
  GET  /world_merges/{merge_id}
  POST /world_merges/{merge_id}/approve  {"approver"}
  POST /world_merges/{merge_id}/reject   {"reason"}

This module proxies those endpoints from the gaia-web container so the
browser-side review UI can call a single same-origin path
(/api/world_merges/...) without CORS handling. The proxy adds no
business logic — it's a thin forwarder that surfaces gaia-study's
responses unchanged, with structured error wrapping on transport
failures.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("GAIA.Web.WorldMerges")

router = APIRouter(prefix="/api/world_merges", tags=["world_merges"])

_STUDY_URL = os.getenv("STUDY_ENDPOINT", "http://gaia-study:8766")
_TIMEOUT = 10.0


async def _study_get(path: str) -> dict:
    """GET <study>/<path>. Surfaces study's body when status<500;
    raises HTTPException with the upstream's body for clean error
    semantics."""
    url = f"{_STUDY_URL}{path}"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url)
    except httpx.HTTPError as e:
        logger.warning("gaia-study GET %s failed: %s", path, e)
        raise HTTPException(
            status_code=502,
            detail=f"upstream gaia-study unreachable: {e}",
        )
    if resp.status_code >= 500:
        raise HTTPException(status_code=502, detail=resp.text[:400])
    try:
        body = resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail="invalid JSON from gaia-study")
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=body)
    return body


async def _study_post(path: str, payload: dict) -> dict:
    """POST <study>/<path> with a JSON body. Same error semantics as
    _study_get."""
    url = f"{_STUDY_URL}{path}"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(url, json=payload)
    except httpx.HTTPError as e:
        logger.warning("gaia-study POST %s failed: %s", path, e)
        raise HTTPException(
            status_code=502,
            detail=f"upstream gaia-study unreachable: {e}",
        )
    if resp.status_code >= 500:
        raise HTTPException(status_code=502, detail=resp.text[:400])
    try:
        body = resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail="invalid JSON from gaia-study")
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=body)
    return body


@router.get("/pending")
async def list_pending() -> dict:
    """Proxy GET /world_merges/pending."""
    return await _study_get("/world_merges/pending")


@router.get("/{merge_id}")
async def get_merge(merge_id: str) -> dict:
    """Proxy GET /world_merges/{merge_id}."""
    return await _study_get(f"/world_merges/{merge_id}")


class _ApproveRequest(BaseModel):
    approver: str = "architect"


@router.post("/{merge_id}/approve")
async def approve(merge_id: str, request: Optional[_ApproveRequest] = None) -> dict:
    """Proxy POST /world_merges/{merge_id}/approve."""
    payload = {"approver": (request.approver if request else "architect")}
    return await _study_post(f"/world_merges/{merge_id}/approve", payload)


class _RejectRequest(BaseModel):
    reason: str = ""


@router.post("/{merge_id}/reject")
async def reject(merge_id: str, request: Optional[_RejectRequest] = None) -> dict:
    """Proxy POST /world_merges/{merge_id}/reject."""
    payload = {"reason": (request.reason if request else "")}
    return await _study_post(f"/world_merges/{merge_id}/reject", payload)

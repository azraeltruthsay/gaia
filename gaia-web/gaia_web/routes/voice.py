"""Voice auto-answer API endpoints for gaia-web.

Manages the Discord voice whitelist and exposes voice connection status.
The VoiceManager instance is accessed via request.app.state.voice_manager.
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger("GAIA.Web.Voice")

router = APIRouter(prefix="/api/voice", tags=["voice"])


class WhitelistAdd(BaseModel):
    user_id: str


def _get_vm(request: Request):
    vm = getattr(request.app.state, "voice_manager", None)
    if vm is None:
        raise HTTPException(status_code=503, detail="Voice manager not initialized")
    return vm


@router.get("/users")
async def list_users(request: Request):
    """All users GAIA has seen in Discord, with whitelist status."""
    vm = _get_vm(request)
    return vm.whitelist.get_seen_users()


@router.get("/whitelist")
async def list_whitelisted(request: Request):
    """List all whitelisted user IDs with names."""
    vm = _get_vm(request)
    users = vm.whitelist.get_seen_users()
    return [u for u in users if u["whitelisted"]]


@router.post("/whitelist")
async def add_to_whitelist(request: Request, body: WhitelistAdd):
    """Add a user to the voice auto-answer whitelist."""
    vm = _get_vm(request)
    vm.whitelist.add(body.user_id)
    return {"ok": True, "user_id": body.user_id}


@router.delete("/whitelist/{user_id}")
async def remove_from_whitelist(request: Request, user_id: str):
    """Remove a user from the voice auto-answer whitelist."""
    vm = _get_vm(request)
    vm.whitelist.remove(user_id)
    return {"ok": True, "user_id": user_id}


@router.get("/status")
async def voice_status(request: Request):
    """Current voice connection status."""
    vm = _get_vm(request)
    return vm.get_status()


@router.post("/disconnect")
async def force_disconnect(request: Request):
    """Force disconnect from current voice call."""
    vm = _get_vm(request)
    await vm.disconnect()
    return {"ok": True, "status": "disconnected"}

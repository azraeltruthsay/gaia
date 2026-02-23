"""Discord DM blocklist API endpoints for gaia-web.

Manages the DM user blocklist and exposes DM user history for the
Mission Control dashboard.  The DMBlocklist instance is accessed via
request.app.state.dm_blocklist.
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger("GAIA.Web.Discord")

router = APIRouter(prefix="/api/discord", tags=["discord"])


class BlocklistAdd(BaseModel):
    user_id: str


def _get_blocklist(request: Request):
    bl = getattr(request.app.state, "dm_blocklist", None)
    if bl is None:
        raise HTTPException(status_code=503, detail="DM blocklist not initialized")
    return bl


@router.get("/dm-users")
async def list_dm_users(request: Request):
    """All users who have DM'd GAIA, with blocked status."""
    bl = _get_blocklist(request)
    return bl.get_dm_users()


@router.get("/blocklist")
async def list_blocked(request: Request):
    """List all blocked DM users with details."""
    bl = _get_blocklist(request)
    users = bl.get_dm_users()
    return [u for u in users if u["blocked"]]


@router.post("/blocklist")
async def block_user(request: Request, body: BlocklistAdd):
    """Block a user from DM interactions with GAIA."""
    bl = _get_blocklist(request)
    bl.block(body.user_id)
    logger.info("Blocked DM user: %s", body.user_id)
    return {"ok": True, "user_id": body.user_id}


@router.delete("/blocklist/{user_id}")
async def unblock_user(request: Request, user_id: str):
    """Unblock a user, allowing DM interactions again."""
    bl = _get_blocklist(request)
    bl.unblock(user_id)
    logger.info("Unblocked DM user: %s", user_id)
    return {"ok": True, "user_id": user_id}

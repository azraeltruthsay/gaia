"""
Changelog routes — query and append change log entries.

GET  /api/changelog?service=X&type=Y&limit=N
POST /api/changelog  {type, service, summary, author?, detail?}
"""

import logging
from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger("GAIA.Web.Changelog")

router = APIRouter()

try:
    from gaia_common.utils.changelog import append_entry, read_entries
except ImportError:
    logger.warning("gaia_common.utils.changelog not available — changelog routes disabled")
    append_entry = None
    read_entries = None


class ChangelogEntry(BaseModel):
    type: str
    service: str
    summary: str
    author: Optional[str] = "manual"
    detail: Optional[str] = None


@router.get("")
async def get_changelog(
    service: Optional[str] = Query(None),
    type: Optional[str] = Query(None, alias="type"),
    limit: int = Query(100, ge=1, le=1000),
):
    if read_entries is None:
        return {"entries": [], "total": 0, "error": "changelog module not available"}
    entries = read_entries(limit=limit, type_filter=type, service_filter=service)
    return {"entries": entries, "total": len(entries)}


@router.post("")
async def post_changelog(entry: ChangelogEntry):
    if append_entry is None:
        return {"error": "changelog module not available"}
    result = append_entry(
        type=entry.type,
        service=entry.service,
        summary=entry.summary,
        author=entry.author or "manual",
        source="manual",
        detail=entry.detail,
    )
    return {"ok": True, "entry": result}

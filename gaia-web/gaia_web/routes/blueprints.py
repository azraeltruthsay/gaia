"""
Blueprint API endpoints for gaia-web.

Exposes the blueprint graph, service list, detail, and markdown rendering
over HTTP. All data sourced from gaia_common.utils.blueprint_io.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from gaia_common.utils.blueprint_io import (
    derive_graph_topology,
    load_all_candidate_blueprints,
    load_all_live_blueprints,
    load_blueprint,
    render_markdown,
)

logger = logging.getLogger("GAIA.Web.Blueprints")

router = APIRouter(prefix="/api/blueprints", tags=["blueprints"])


@router.get("/graph")
async def get_graph(include_candidates: bool = True):
    """Return the derived graph topology (nodes + edges).

    By default includes candidate blueprints since no live blueprints
    are promoted yet. Set ?include_candidates=false for live-only.
    """
    blueprints = load_all_live_blueprints()
    if include_candidates:
        candidates = load_all_candidate_blueprints()
        # Candidates fill in where live doesn't exist yet
        for sid, bp in candidates.items():
            if sid not in blueprints:
                blueprints[sid] = bp
    topology = derive_graph_topology(blueprints)
    return topology.model_dump(mode="json")


@router.get("")
async def list_blueprints():
    """List all blueprints with summary metadata."""
    live = load_all_live_blueprints()
    candidates = load_all_candidate_blueprints()

    result = []
    seen = set()

    for sid, bp in live.items():
        seen.add(sid)
        result.append({
            "id": sid,
            "role": bp.role,
            "status": bp.meta.status.value,
            "genesis": bp.meta.genesis,
            "interface_count": len(bp.interfaces),
            "candidate": False,
        })

    for sid, bp in candidates.items():
        if sid not in seen:
            result.append({
                "id": sid,
                "role": bp.role,
                "status": bp.meta.status.value,
                "genesis": bp.meta.genesis,
                "interface_count": len(bp.interfaces),
                "candidate": True,
            })

    return result


@router.get("/{service_id}")
async def get_blueprint_detail(service_id: str, candidate: Optional[bool] = None):
    """Return full blueprint JSON for a service.

    If candidate is not specified, tries live first then candidate.
    """
    bp = None
    if candidate is None:
        bp = load_blueprint(service_id, candidate=False)
        if bp is None:
            bp = load_blueprint(service_id, candidate=True)
    else:
        bp = load_blueprint(service_id, candidate=candidate)

    if bp is None:
        raise HTTPException(status_code=404, detail=f"Blueprint not found: {service_id}")

    return bp.model_dump(mode="json")


@router.get("/{service_id}/markdown")
async def get_blueprint_markdown(service_id: str, candidate: Optional[bool] = None):
    """Return rendered markdown for a blueprint."""
    bp = None
    if candidate is None:
        bp = load_blueprint(service_id, candidate=False)
        if bp is None:
            bp = load_blueprint(service_id, candidate=True)
    else:
        bp = load_blueprint(service_id, candidate=candidate)

    if bp is None:
        raise HTTPException(status_code=404, detail=f"Blueprint not found: {service_id}")

    return PlainTextResponse(content=render_markdown(bp), media_type="text/markdown")

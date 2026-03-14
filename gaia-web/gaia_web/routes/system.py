"""
System status routes for Mission Control dashboard.

Aggregates health data from gaia-doctor and gaia-orchestrator to provide
the endpoints the frontend polls: /services, /sleep, /status, /cognitive/*.
"""

import logging
import os

import httpx
from fastapi import APIRouter, Request

logger = logging.getLogger("GAIA.Web.System")

router = APIRouter()

DOCTOR_URL = os.getenv("DOCTOR_ENDPOINT", "http://gaia-doctor:6419")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_ENDPOINT", "http://gaia-orchestrator:6410")
CORE_URL = os.getenv("CORE_ENDPOINT", "http://gaia-core:6415")
MONKEY_URL = os.getenv("MONKEY_ENDPOINT", "http://gaia-monkey:6420")
STUDY_URL = os.getenv("STUDY_ENDPOINT", "http://gaia-study:8766")

# Map doctor service names to display-friendly IDs
_SERVICE_DISPLAY = {
    "gaia-core": "gaia-core",
    "gaia-web": "gaia-web",
    "gaia-mcp": "gaia-mcp",
    "gaia-prime": "gaia-prime",
    "gaia-audio": "gaia-audio",
    "gaia-core-candidate": "gaia-core-candidate",
    "gaia-mcp-candidate": "gaia-mcp-candidate",
}


@router.get("/services")
async def system_services():
    """Aggregate service health from gaia-doctor status endpoint."""
    services = []

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{DOCTOR_URL}/status")
            if resp.status_code == 200:
                data = resp.json()
                for name, info in data.get("services", {}).items():
                    healthy = info.get("healthy")
                    if healthy is True:
                        status = "online"
                    elif healthy is False:
                        status = "offline"
                    else:
                        status = "unknown"

                    entry = {
                        "id": _SERVICE_DISPLAY.get(name, name),
                        "status": status,
                        "latency_ms": None,
                        "candidate": "candidate" in name,
                        "consecutive_failures": info.get("consecutive_failures", 0),
                        "alarmed": info.get("alarmed", False),
                        "restarts_in_window": info.get("restarts_in_window", 0),
                    }
                    services.append(entry)

                # Add doctor itself as healthy (if we got here, it's up)
                services.append({
                    "id": "gaia-doctor",
                    "status": "online",
                    "latency_ms": None,
                    "candidate": False,
                })

    except Exception as e:
        logger.debug("Failed to fetch doctor status: %s", e)

    return services


@router.get("/sleep")
async def system_sleep():
    """Get sleep state from gaia-core."""
    result = {"state": "unknown", "gpu_owner": "--"}

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Try gaia-core for sleep state
            resp = await client.get(f"{CORE_URL}/health")
            if resp.status_code == 200:
                data = resp.json()
                result["state"] = data.get("sleep_state", data.get("state", "active"))

            # Try orchestrator for GPU owner
            resp = await client.get(f"{ORCHESTRATOR_URL}/status")
            if resp.status_code == 200:
                orch = resp.json()
                gpu = orch.get("gpu", {})
                owner = gpu.get("owner", "none")
                result["gpu_owner"] = owner if owner != "none" else "--"

    except Exception as e:
        logger.debug("Failed to fetch sleep/GPU status: %s", e)

    return result


@router.get("/status")
async def system_status():
    """Get orchestrator status (GPU owner, general health) + serenity state."""
    result = {"gpu_owner": "--", "status": "unknown", "serenity": None}

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{ORCHESTRATOR_URL}/status")
            if resp.status_code == 200:
                data = resp.json()
                result["status"] = data.get("status", "unknown")
                gpu = data.get("gpu", {})
                owner = gpu.get("owner", "none")
                result["gpu_owner"] = owner if owner != "none" else "--"
    except Exception as e:
        logger.debug("Failed to fetch orchestrator status: %s", e)

    # Fetch serenity state from gaia-monkey (primary), fall back to doctor
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{MONKEY_URL}/serenity")
            if resp.status_code == 200:
                result["serenity"] = resp.json()
    except Exception:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{DOCTOR_URL}/serenity")
                if resp.status_code == 200:
                    result["serenity"] = resp.json()
        except Exception:
            pass

    return result


# ── Maintenance Mode Proxy (doctor) ───────────────────────────────────────

@router.get("/maintenance/status")
async def maintenance_status():
    """Get maintenance mode status from gaia-doctor."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{DOCTOR_URL}/maintenance/status")
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        logger.debug("Failed to fetch maintenance status: %s", e)
    return {"active": False}


@router.post("/maintenance/enter")
async def maintenance_enter(request: Request):
    """Enter maintenance mode via gaia-doctor."""
    try:
        body = await request.body()
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{DOCTOR_URL}/maintenance/enter",
                content=body,
                headers={"Content-Type": "application/json"},
            )
            return resp.json()
    except Exception as e:
        logger.debug("Failed to enter maintenance mode: %s", e)
        return {"error": str(e)}


@router.post("/maintenance/exit")
async def maintenance_exit():
    """Exit maintenance mode via gaia-doctor."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{DOCTOR_URL}/maintenance/exit")
            return resp.json()
    except Exception as e:
        logger.debug("Failed to exit maintenance mode: %s", e)
        return {"error": str(e)}


# ── Cognitive Battery Proxy (doctor) ──────────────────────────────────────

@router.get("/cognitive/status")
async def cognitive_status():
    """Get cognitive test battery status + alignment from gaia-doctor."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{DOCTOR_URL}/cognitive/status")
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        logger.debug("Failed to fetch cognitive status: %s", e)
    return {"running": False, "alignment": "UNKNOWN", "last_run": {}}


@router.get("/cognitive/results")
async def cognitive_results():
    """Get full cognitive test battery results from gaia-doctor."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{DOCTOR_URL}/cognitive/results")
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        logger.debug("Failed to fetch cognitive results: %s", e)
    return {"message": "unavailable"}


@router.get("/cognitive/tests")
async def cognitive_tests():
    """List all registered cognitive tests from gaia-doctor."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{DOCTOR_URL}/cognitive/tests")
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        logger.debug("Failed to fetch cognitive tests: %s", e)
    return {"tests": []}


@router.post("/cognitive/run")
async def cognitive_run(request: Request):
    """Trigger a cognitive test battery run on gaia-doctor."""
    try:
        body = await request.body()
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{DOCTOR_URL}/cognitive/run",
                content=body,
                headers={"Content-Type": "application/json"},
            )
            return resp.json()
    except Exception as e:
        logger.debug("Failed to trigger cognitive run: %s", e)
        return {"error": str(e)}


# ── Training Pipeline Status ─────────────────────────────────────────────

@router.get("/pipeline/status")
async def pipeline_status():
    """Get self-awareness training pipeline status from gaia-doctor."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{DOCTOR_URL}/pipeline")
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        logger.debug("Failed to fetch pipeline status: %s", e)
    return {"status": "no pipeline running"}

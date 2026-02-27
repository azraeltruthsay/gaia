"""
Audio Listener control routes for Mission Control dashboard.

Reads/writes listener_control.json and listener_status.json to manage
the host-side audio capture daemon (scripts/gaia_listener.py).
Also provides an ingest endpoint to forward transcripts to gaia-core.
"""

import json
import logging
import os
from pathlib import Path

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("GAIA.Web.Audio")

router = APIRouter(prefix="/api/audio", tags=["audio"])

CONTROL_PATH = Path("/logs/listener_control.json")
STATUS_PATH = Path("/logs/listener_status.json")
CORE_ENDPOINT = os.environ.get("CORE_ENDPOINT", "http://gaia-core-candidate:6415")
CORE_FALLBACK_ENDPOINT = os.environ.get("CORE_FALLBACK_ENDPOINT", "")


# ── Models ───────────────────────────────────────────────────────────────────

class ListenerStartRequest(BaseModel):
    mode: str = "passive"
    save_audio: bool = False
    compress: bool = False


class IngestRequest(BaseModel):
    transcript: str
    source: str = "Audio Listener"


# ── Status ───────────────────────────────────────────────────────────────────

@router.get("/listener/status")
async def listener_status():
    """Read the listener status file."""
    try:
        if STATUS_PATH.exists():
            data = json.loads(STATUS_PATH.read_text())
            return JSONResponse(status_code=200, content=data)
        return JSONResponse(status_code=200, content={
            "running": False, "capturing": False,
            "message": "Listener not started or status file missing",
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Start / Stop ─────────────────────────────────────────────────────────────

@router.post("/listener/start")
async def listener_start(req: ListenerStartRequest):
    """Write a start command to the listener control file."""
    try:
        control = {
            "command": "start",
            "mode": req.mode,
            "save_audio": req.save_audio,
            "compress": req.compress,
        }
        CONTROL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CONTROL_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(control, indent=2))
        tmp.rename(CONTROL_PATH)
        logger.info("Listener start command written: mode=%s save=%s compress=%s",
                     req.mode, req.save_audio, req.compress)
        return JSONResponse(status_code=200, content={"ok": True, "command": control})
    except Exception as e:
        logger.error("Failed to write listener control: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/listener/stop")
async def listener_stop():
    """Write a stop command to the listener control file."""
    try:
        control = {"command": "stop"}
        CONTROL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CONTROL_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(control, indent=2))
        tmp.rename(CONTROL_PATH)
        logger.info("Listener stop command written")
        return JSONResponse(status_code=200, content={"ok": True, "command": control})
    except Exception as e:
        logger.error("Failed to write listener control: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Ingest ───────────────────────────────────────────────────────────────────

@router.post("/listener/ingest")
async def listener_ingest(req: IngestRequest):
    """Forward accumulated transcript text to gaia-core for processing."""
    if not req.transcript.strip():
        return JSONResponse(status_code=400, content={"error": "Empty transcript"})

    packet = {
        "user_input": f"[AUDIO TRANSCRIPTION — {req.source}]\n\n{req.transcript}",
        "metadata": {
            "source": "audio_listener_dashboard",
            "packet_type": "audio_transcription",
        },
    }

    urls = [f"{CORE_ENDPOINT}/process_packet"]
    if CORE_FALLBACK_ENDPOINT:
        urls.append(f"{CORE_FALLBACK_ENDPOINT}/process_packet")

    for url in urls:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=packet)
                if resp.status_code == 200:
                    logger.info("Transcript ingested to gaia-core (%d chars)", len(req.transcript))
                    return JSONResponse(status_code=200, content={
                        "ok": True,
                        "chars": len(req.transcript),
                        "core_status": resp.status_code,
                    })
                else:
                    logger.warning("gaia-core returned %d from %s", resp.status_code, url)
        except httpx.ConnectError:
            logger.warning("gaia-core unreachable at %s", url)
            continue
        except Exception as e:
            logger.error("Ingest error at %s: %s", url, e)
            continue

    return JSONResponse(status_code=503, content={"error": "gaia-core unreachable"})

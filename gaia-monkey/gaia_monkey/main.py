"""gaia-monkey — Adversarial Chaos Service.

Port 6420. Manages Defensive Meditation, Serenity State, and Chaos Drills.
Extracted from gaia-doctor to allow operational flexibility and linguistic chaos.
"""
import asyncio
import json
import logging
import os
import random
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from gaia_monkey import (
    chaos_engine,
    linguistic_engine,
    meditation_controller,
    scheduler,
    serenity_manager,
)

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [gaia-monkey] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("gaia-monkey")

# Drill history (in-memory, last 50)
_drill_history: list[dict] = []


async def _inject_chaos(config: dict):
    """Pick and run a drill type based on config. Called by scheduler."""
    drill_types = config.get("drill_types", ["container"])
    targets = config.get("targets") or None

    # Require meditation for chaos drills
    if not meditation_controller.is_active():
        meditation_controller.enter()

    drill_type = random.choice(drill_types)
    log.info("🐒 Injecting chaos: drill_type=%s", drill_type)

    if drill_type == "container":
        result = await asyncio.to_thread(chaos_engine.run_container_drill, targets)
    elif drill_type == "code":
        result = await asyncio.to_thread(chaos_engine.run_code_drill, targets)
    elif drill_type == "linguistic":
        result = await linguistic_engine.run_suite("persona")
    else:
        result = {"error": f"unknown drill type: {drill_type}"}

    result["drill_type"] = drill_type
    result["timestamp"] = time.time()
    _drill_history.append(result)
    if len(_drill_history) > 50:
        _drill_history.pop(0)
    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("gaia-monkey starting on port 6420")
    scheduler.set_inject_fn(_inject_chaos)
    asyncio.create_task(scheduler.run_background())
    yield
    log.info("gaia-monkey shutting down")


app = FastAPI(lifespan=lifespan, title="gaia-monkey")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "gaia-monkey"}


@app.get("/status")
async def status():
    return {
        "service": "gaia-monkey",
        "meditation": meditation_controller.get_status(),
        "serenity": serenity_manager.get_report(),
        **scheduler.get_status(),
        "recent_history": _drill_history[-5:],
    }


@app.get("/config")
async def get_config():
    return scheduler.load_config()


@app.post("/config")
async def set_config(body: dict):
    current = scheduler.load_config()
    current.update(body)
    scheduler.save_config(current)
    return {"status": "saved", "config": current}


@app.post("/chaos/inject")
async def inject_chaos(body: dict = {}):
    """Pick drill type per config and fire."""
    config = scheduler.load_config()
    # Allow override from request body
    if body.get("drill_types"):
        config["drill_types"] = body["drill_types"]
    if body.get("targets"):
        config["targets"] = body["targets"]
    result = await _inject_chaos(config)
    return result


@app.post("/chaos/drill")
async def chaos_drill(body: dict = {}):
    """Container-level fault injection."""
    targets = body.get("targets")
    result = await asyncio.to_thread(chaos_engine.run_container_drill, targets)
    result["drill_type"] = "container"
    result["timestamp"] = time.time()
    _drill_history.append(result)
    if len(_drill_history) > 50:
        _drill_history.pop(0)
    return result


@app.post("/chaos/code")
async def chaos_code(body: dict = {}):
    """Semantic code fault injection."""
    targets = body.get("targets")
    difficulty = body.get("difficulty")
    result = await asyncio.to_thread(chaos_engine.run_code_drill, targets, difficulty)
    result["timestamp"] = time.time()
    _drill_history.append(result)
    if len(_drill_history) > 50:
        _drill_history.pop(0)
    return result


@app.post("/chaos/linguistic")
async def chaos_linguistic(body: dict = {}):
    """PromptFoo linguistic evaluation."""
    suite = body.get("suite", "persona")
    result = await linguistic_engine.run_suite(suite)
    result["drill_type"] = "linguistic"
    result["timestamp"] = time.time()
    _drill_history.append(result)
    if len(_drill_history) > 50:
        _drill_history.pop(0)
    return result


@app.get("/chaos/history")
async def chaos_history(limit: int = 20):
    return {"history": _drill_history[-limit:]}


@app.post("/meditation/enter")
async def meditation_enter():
    meditation_controller.enter()
    return {"status": "entered", "meditation": meditation_controller.get_status()}


@app.post("/meditation/exit")
async def meditation_exit():
    meditation_controller.exit_meditation()
    return {"status": "exited", "meditation": meditation_controller.get_status()}


@app.get("/serenity")
async def serenity():
    return {**serenity_manager.get_report(), "meditation_active": meditation_controller.is_active()}


@app.post("/serenity/break")
async def serenity_break(body: dict = {}):
    reason = body.get("reason", "external break request")
    serenity_manager.break_serenity(reason)
    return {"status": "broken", "serenity": serenity_manager.get_report()}


@app.post("/serenity/record_recovery")
async def serenity_record(body: dict = {}):
    """Record a recovery event from gaia-doctor for serenity scoring."""
    category = body.get("category", "standard_recovery")
    detail = body.get("detail", "")
    serenity_manager.record_recovery(category, detail, meditation_controller.is_active())
    log.info("🪷 Recovery recorded: %s — %s", category, detail)
    return {"status": "recorded", "serenity": serenity_manager.get_report()}


@app.get("/drills")
async def drills(limit: int = 20):
    """Drill history — alias of /chaos/history for test plan compatibility."""
    return {"history": _drill_history[-limit:]}


@app.post("/serenity/reset")
async def serenity_reset():
    serenity_manager.reset_serenity()
    log.info("🪷 Serenity state RESET via API")
    return {"status": "reset", "serenity": serenity_manager.get_report()}

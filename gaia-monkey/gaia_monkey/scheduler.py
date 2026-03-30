"""Chaos Scheduler — asyncio background task for automated drill modes."""
import asyncio
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Callable

# Maintenance mode check — skip auto-firing during dev sessions
try:
    from gaia_common.utils.maintenance import is_maintenance_active
except ImportError:
    _LEGACY_FLAG = Path(os.environ.get("SHARED_DIR", "/shared")) / "ha_maintenance"
    def is_maintenance_active():
        return _LEGACY_FLAG.exists()

log = logging.getLogger("gaia-monkey.scheduler")

CONFIG_PATH = Path(os.environ.get("SHARED_DIR", "/shared")) / "monkey" / "config.json"

DEFAULT_CONFIG = {
    "mode": "triggered",
    "enabled": True,
    "drill_types": ["container", "code"],
    "schedule_interval_hours": 6,
    "random_min_hours": 1,
    "random_max_hours": 24,
    "persistent_cooldown_minutes": 30,
    "targets": ["gaia-core-candidate", "gaia-mcp-candidate"],
    "promptfoo_enabled": False,
}

_last_run: float = 0.0
_next_run: float | None = None
_inject_chaos_fn: Callable | None = None


def set_inject_fn(fn: Callable):
    """Register the chaos injection function (called by main.py)."""
    global _inject_chaos_fn
    _inject_chaos_fn = fn


def load_config() -> dict:
    try:
        if CONFIG_PATH.exists():
            return {**DEFAULT_CONFIG, **json.loads(CONFIG_PATH.read_text())}
    except Exception:
        pass
    return DEFAULT_CONFIG.copy()


def save_config(config: dict):
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(config, indent=2))
    except Exception as e:
        log.error("Failed to save config: %s", e)


def get_status() -> dict:
    config = load_config()
    return {
        "mode": config.get("mode", "triggered"),
        "enabled": config.get("enabled", True),
        "last_run": _last_run or None,
        "next_run": _next_run,
        "config": config,
    }


async def run_background():
    """Background task — loops forever, fires drills based on mode."""
    global _last_run, _next_run

    while True:
        await asyncio.sleep(60)  # Check every minute

        config = load_config()
        mode = config.get("mode", "triggered")
        enabled = config.get("enabled", True)

        if not enabled or mode == "triggered" or _inject_chaos_fn is None:
            _next_run = None
            continue

        # Skip auto-firing during maintenance mode
        if is_maintenance_active():
            continue

        now = time.time()

        if mode == "scheduled":
            interval_s = config.get("schedule_interval_hours", 6) * 3600
            if _next_run is None:
                _next_run = now + interval_s
            if now >= _next_run:
                await _fire_drill(config)
                _next_run = now + interval_s

        elif mode == "random":
            if _next_run is None:
                min_h = config.get("random_min_hours", 1)
                max_h = config.get("random_max_hours", 24)
                delay = random.uniform(min_h * 3600, max_h * 3600)
                _next_run = now + delay
                log.info("🐒 Random mode: next drill in %.1fh", delay / 3600)
            if now >= _next_run:
                await _fire_drill(config)
                min_h = config.get("random_min_hours", 1)
                max_h = config.get("random_max_hours", 24)
                delay = random.uniform(min_h * 3600, max_h * 3600)
                _next_run = now + delay
                log.info("🐒 Random mode: next drill in %.1fh", delay / 3600)

        elif mode == "persistent":
            cooldown_s = config.get("persistent_cooldown_minutes", 30) * 60
            if _next_run is None or now >= _next_run:
                await _fire_drill(config)
                _next_run = now + cooldown_s


async def _fire_drill(config: dict):
    global _last_run
    log.info("🐒 Scheduled drill firing (mode: %s)", config.get("mode"))
    try:
        _last_run = time.time()
        await _inject_chaos_fn(config)
    except Exception as e:
        log.error("🐒 Scheduled drill failed: %s", e)

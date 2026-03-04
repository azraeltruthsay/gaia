# Dev Journal: Doctor Web Auto-Heal and Discord Latency NaN Fix

**Date:** March 4, 2026

## Overview

Two targeted bug fixes in this session. gaia-web went unhealthy for ~3 hours due to a stalled StatReload — the Docker health check was timing out because the uvicorn worker subprocess died after a reload and was never respawned (Discord WebSocket reconnect mid-shutdown kept the worker alive long enough to confuse the reloader). Manual `docker restart` resolved it.

From that incident, two action items:
1. Have gaia-doctor auto-restart gaia-web when unhealthy, with a circuit breaker
2. Fix a pre-existing `ValueError: Out of range float values are not JSON compliant` on `/api/system/services` caused by Discord's `nan` latency value

---

## Incident Root Cause: StatReload Deadlock

**Trigger:** uvicorn's StatReload detected changes in `gaia_web/main.py` and initiated a graceful worker shutdown.

**Problem:** The Discord WebSocket reconnected during the shutdown window ("Waiting for connections to close"), keeping the old worker's event loop alive. The reloader waited indefinitely. No new worker was spawned. Port 6414 went unbound.

**Evidence:** 288 consecutive Docker health check timeouts (each 10s, 30s interval = ~2.4 hours unhealthy). Container PID 1 alive but `ss -tlnp` showed no bound sockets.

**Fix:** `docker restart gaia-web` (full container restart, not in-process reload). Import and startup code were healthy — the issue was purely the reload mechanism stalling.

---

## Change 1: gaia-doctor — Auto-Restart with Circuit Breaker

**File:** `gaia-doctor/doctor.py`

### What Changed

Added `gaia-web` to the doctor's service registry with a new `"restart"` remediation mode (alongside existing `None` = observe only, and `"ha"` = HA compose overlay).

**New function: `docker_restart(name)`**
- Runs `docker restart <name>` (stdlib subprocess, no compose overlay needed)
- Enforces a **circuit breaker**: max 2 restarts within a rolling 30-minute window
- Reuses existing `RESTART_COOLDOWN` (300s) between individual restart attempts
- Respects `MAINTENANCE_FLAG` (skips if `/shared/ha_maintenance` exists)

**New function: `raise_alarm(name, reason)`**
- Called when the circuit breaker trips
- Logs `[ALARM]` at ERROR level
- Writes to `/shared/doctor/alarms.json`
- Adds to in-memory `_alarmed_services` set
- Alarm clears automatically when service recovers

**New state:**
- `_restart_history: dict[str, list]` — monotonic timestamps of restarts, trimmed to the rolling window
- `_alarmed_services: set` — services currently in alarm state
- `_active_alarms: list` — rolling log of alarm events

**New HTTP endpoint:** `GET /alarms` — returns alarmed services and recent alarm log.

**`_build_status()` updated** to include `active_alarms`, per-service `restarts_in_window`, `alarmed` flag, and `remediation` mode (replacing the old `can_remediate` boolean).

### Config

| Env Var | Default | Meaning |
|---------|---------|---------|
| `PROD_RESTART_MAX` | 2 | Max auto-restarts before circuit breaker trips |
| `PROD_RESTART_WINDOW` | 1800 | Rolling window in seconds (30 min) |

---

## Change 2: Discord Latency NaN Fix

**File:** `gaia-web/gaia_web/discord_interface.py:876`

### Problem

`discord.py`'s `client.latency` returns `float('nan')` before the first WebSocket heartbeat is acknowledged. The old guard:

```python
"latency_ms": round(_bot.latency * 1000, 1) if _bot.latency else None,
```

...failed because `bool(float('nan'))` is `True` in Python, so the condition passed. `round(nan * 1000, 1)` returns `nan`, which FastAPI's JSON encoder rejects with `ValueError: Out of range float values are not JSON compliant`.

This caused `/api/system/services` to return 500 during the brief window after Discord connects but before the first heartbeat.

### Fix

```python
import math
# ...
"latency_ms": round(_bot.latency * 1000, 1) if _bot.latency and math.isfinite(_bot.latency) else None,
```

`math.isfinite()` correctly rejects both `nan` and `inf`, returning `None` instead.

---

## Deployment

Both containers restarted and confirmed healthy:
- `gaia-doctor`: polling cycle active, gaia-web now in registry
- `gaia-web`: Application startup complete, Discord bot connected

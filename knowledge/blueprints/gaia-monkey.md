# GAIA Service Blueprint: `gaia-monkey` (The Adversary)

> **Status:** 🟢 LIVE
> **Service version:** 1.0
> **Blueprint version:** 1.0
> **Port:** 6420
> **Created:** 2026-03-10

---

## Role and Overview

`gaia-monkey` is GAIA's adversarial resilience engine. It deliberately breaks things — stopped containers, injected code faults, linguistic attacks — to prove GAIA can recover. It manages two critical cross-service state machines:

- **Serenity State** — a trust signal, earned through demonstrated recovery under duress, that gates autonomous promotion and initiative cycles
- **Defensive Meditation** — a time-boxed window (30 min) that relaxes gaia-doctor's restart circuit breaker, signalling that chaos is intentional

Extracted from `gaia-doctor/doctor.py` (previously ~500 LOC of chaos logic embedded in the immune watchdog). The separation lets chaos drills run on flexible schedules without coupling to the production health watchdog's stability guarantees. gaia-doctor now delegates all chaos/serenity/meditation management to this service via HTTP.

**Cognitive role:** The Adversary

---

## Container Configuration

**Base image:** Multi-stage — `node:20-slim` (PromptFoo install) → `python:3.12-slim`

**Port:** `6420`

**Health check:** `curl -f http://localhost:6420/health` (30s interval, 60s start_period)

**Startup:** `python -m uvicorn gaia_monkey.main:app --host 0.0.0.0 --port 6420`

**Depends on:** `gaia-core` (for LLM-powered Tier 2 repair; degraded without it, not failed)

### Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `CORE_ENDPOINT` | `http://gaia-core:6415` | LLM repair target |
| `DOCTOR_ENDPOINT` | `http://gaia-doctor:6419` | For future doctor notifications |
| `SHARED_DIR` | `/shared` | Path to gaia-shared volume |
| `PROJECT_ROOT` | `/gaia/GAIA_Project` | Root for candidate file access |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `PROMPTFOO_HOME` | `/tmp/.promptfoo` | PromptFoo cache directory |

### Volume Mounts

| Mount | Mode | Purpose |
|-------|------|---------|
| `/var/run/docker.sock` | `:ro` | Container control (stop/start/restart) |
| `gaia-shared:/shared` | `:rw` | Serenity + meditation shared state |
| `.:/gaia/GAIA_Project` | `:ro` | Candidate .py file access for fault injection |

---

## Source Structure

```
gaia-monkey/
├── Dockerfile                        # Multi-stage: node:20-slim → python:3.12-slim
├── requirements.txt                  # fastapi, uvicorn[standard], httpx
├── gaia_monkey/
│   ├── __init__.py
│   ├── main.py                       # FastAPI app, lifespan, all 14 endpoints
│   ├── chaos_engine.py               # Container + code drill orchestration
│   ├── serenity_manager.py           # Thread-safe serenity state + persistence
│   ├── meditation_controller.py      # Defensive Meditation enter/exit/timeout
│   ├── fault_injector.py             # File picker + semantic fault injection
│   ├── cognitive_validator.py        # Live CognitionPacket inference check
│   ├── linguistic_engine.py          # PromptFoo async subprocess runner
│   └── scheduler.py                  # asyncio background task (4 modes)
└── promptfoo-suites/
    ├── persona.yaml                  # Identity stability / jailbreak red-team (5 tests)
    ├── factuality.yaml               # Basic factual accuracy (5 tests)
    └── format.yaml                   # Output format compliance (3 tests)
```

---

## HTTP API

### Operational

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Container health check |
| `GET` | `/status` | Mode, next run, serenity, last 5 drill results |
| `GET` | `/config` | Read config from `/shared/monkey/config.json` |
| `POST` | `/config` | Write config (live effect, no restart needed) |
| `GET` | `/chaos/history` | Last N drill results (default 20, max 50) |

### Chaos Triggers

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chaos/inject` | Auto-pick drill type per config; one-click chaos |
| `POST` | `/chaos/drill` | Container fault injection |
| `POST` | `/chaos/code` | Semantic code fault injection |
| `POST` | `/chaos/linguistic` | PromptFoo red-team evaluation |

### Meditation

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/meditation/enter` | Enter Defensive Meditation (30-min window) |
| `POST` | `/meditation/exit` | Exit early |

### Serenity

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/serenity` | Current serenity state (consumed by gaia-web + gaia-doctor) |
| `POST` | `/serenity/break` | Force-break serenity (called by gaia-doctor on vital failure) |
| `POST` | `/serenity/reset` | Manual reset (debug) |

---

## Operational Config Schema

Stored at `/shared/monkey/config.json`. Changes take effect on the scheduler's next 60-second poll.

```json
{
  "mode": "triggered",
  "enabled": true,
  "drill_types": ["container", "code"],
  "schedule_interval_hours": 6,
  "random_min_hours": 1,
  "random_max_hours": 24,
  "persistent_cooldown_minutes": 30,
  "targets": ["gaia-core-candidate", "gaia-mcp-candidate"],
  "promptfoo_enabled": false
}
```

### Scheduler Modes

| Mode | Behaviour |
|------|-----------|
| `triggered` | No auto-scheduling. Drills only fire on explicit `/chaos/inject` |
| `scheduled` | Fires every `schedule_interval_hours`. Resets timer after each run |
| `random` | Fires at random interval between `random_min_hours` and `random_max_hours`. Re-randomises after each run |
| `persistent` | Loops continuously with `persistent_cooldown_minutes` between drills |

---

## Subsystems

### Defensive Meditation

A time-boxed trust window (max 30 minutes) that enables chaos drills. When active:
- `chaos_engine` permits drills (otherwise rejects with 400-like error dict)
- gaia-doctor reads `/shared/doctor/defensive_meditation.json` to bypass the restart circuit breaker

Enter meditation before running drills manually. The `/chaos/inject` endpoint auto-enters meditation if it isn't already active.

```python
# Shared flag written by meditation_controller.py:
/shared/doctor/defensive_meditation.json
{"active": true, "started": 1741564800.0, "max_duration": 1800}
```

Auto-expires after 30 minutes — `is_active()` checks elapsed time and calls `exit_meditation()` if exceeded.

### Serenity State

A trust signal earned through demonstrated recovery under stress. Gated by a weighted point system:

| Recovery type | Points |
|---------------|--------|
| `vital_recovery` | 2.0 (LLM-repaired code chaos) |
| `cognitive_validation` | 2.0 (live inference passed post-chaos) |
| `standard_recovery` | 0.5 (non-LLM recovery) |
| `service_recovery` | 0.5 (container restart recovery) |
| `test_pass` | 0.5 (post-chaos test suite) |

**Threshold:** 5.0 points → Serenity achieved.

Serenity **gates** (in gaia-core and gaia-doctor):
- Autonomous code evolution (`code_evolution` sleep task)
- Initiative cycle tasks
- Sovereign promotion auto-approval (skips cognitive review)

Serenity is **broken** when:
- A vital service (gaia-core, gaia-web, gaia-mcp) goes DOWN outside of meditation
- A structural audit fails on a production service

gaia-doctor calls `POST /serenity/break` for these events. Serenity resets to score=0 on break.

Persisted to `/shared/doctor/serenity.json` for cross-service reading:
```json
{
  "serene": false,
  "score": 3.5,
  "threshold": 5.0,
  "achieved_at": null,
  "reason": ""
}
```

### Chaos Engine — Container Drill

**Flow (per target):**
1. Verify baseline health via HTTP
2. `docker stop <container>` — inject fault
3. Verify it's actually down (2s wait)
4. `docker start <container>` — trigger recovery
5. Poll health up to 6×5s (30s total)
6. If "core" in service name: run cognitive validation (live inference test)
7. Record recovery points for serenity scoring

**Default targets:** `gaia-core-candidate`, `gaia-mcp-candidate`
Only candidate services are drilled by default. Production services can be targeted explicitly but this is discouraged.

### Chaos Engine — Code Drill

The real Chaos Monkey. Tests the LLM's ability to repair its own codebase.

**Flow (per target):**
1. Pick a non-vital `.py` file (avoids `main`, `__init__`, `agent_core`, `tools`, etc.)
2. Inject a semantic fault — one of three types:
   - **remove_import**: Comments out a random import with `# CHAOS_MONKEY_REMOVED:`
   - **break_return**: Replaces a return value with `return None  # CHAOS_MONKEY_BREAK`
   - **comment_assignment**: Disables an assignment with `# CHAOS_MONKEY_DISABLED:`
3. All faults pass `ast.parse` — they're semantic, not syntactic
4. Write broken content via `docker exec` into the container's filesystem
5. `docker restart` — triggers the fault at runtime
6. Send to gaia-core `/api/repair/structural` for **live LLM repair** (120s timeout)
7. If LLM repair succeeds: verify `CHAOS_MONKEY` marker is removed, re-parse, restart, verify health
8. If LLM repair fails: emergency-restore original content from memory
9. **Full serenity points (vital_recovery + cognitive_validation = 4.0) only awarded for LLM-repaired faults**

### Linguistic Engine (PromptFoo)

Runs PromptFoo evaluation suites as async subprocesses. Three built-in suites:

| Suite | Tests | Checks |
|-------|-------|--------|
| `persona.yaml` | 5 | Identity stability under jailbreak attempts (DAN, GPT-4 persona breaks, etc.) |
| `factuality.yaml` | 5 | Basic factual accuracy (7×8=56, Paris, Red Planet, etc.) |
| `format.yaml` | 3 | Output structure compliance (numbered lists, length bounds) |

Results: `{suite, passed, passes, failures, failed_assertions[]}`. Currently does not award serenity points.

**PromptFoo provider:** HTTP provider pointing to `gaia-core:6415/process_user_input`. Can be redirected to any GAIA endpoint.

---

## Cross-Service Integration

### gaia-doctor → gaia-monkey

gaia-doctor delegates:
- `_notify_monkey_break_serenity(reason)` → `POST /serenity/break` (non-blocking, 3s timeout, fire-and-forget)
- `_is_meditation_active()` → reads `/shared/doctor/defensive_meditation.json` (written by monkey)
- `_get_serenity_report()` → reads `/shared/doctor/serenity.json` (written by monkey)

### gaia-web → gaia-monkey

- `GET /api/chaos/*` → thin proxy to `http://gaia-monkey:6420/*`
- `GET /api/system/status` fetches serenity from monkey first, falls back to doctor

### Dashboard (Commands Tab)

The Chaos Monkey group in the Commands tab provides:
- Mode selector (triggered/scheduled/random/persistent)
- Interval config (shown only in scheduled mode)
- Drill type checkboxes (container/code/linguistic)
- **⚡ Inject Chaos** button → `POST /api/chaos/inject`
- Serenity badge (green `☯ Serene` or amber `3.5/5.0`)
- Last result summary line

---

## Failure Modes

| Condition | Response | Severity | Auto-recovers |
|-----------|----------|----------|---------------|
| gaia-core unavailable during code drill | Emergency-restores original file; marks failed_recovery | degraded | Yes (file restored) |
| PromptFoo not installed/found | Returns error result; other drills unaffected | degraded | No |
| Container fails to recover in 30s | Marks failed_recovery; serenity not awarded | partial | No |
| Docker socket unavailable | All container/code drills fail; linguistic OK | critical | No |
| Scheduler restarts (next_run lost) | Recalculates from now; may fire slightly late/early | minor | Yes |
| Meditation auto-expires mid-drill | Drill continues; serenity awarded if threshold met before exit | minor | Yes |

---

## Design Decisions

- **Extracted from gaia-doctor** to separate passive immune monitoring from active adversarial testing. doctor's watchdog loop now has no chaos-related global state.
- **Shared file protocol** (serenity.json, defensive_meditation.json) enables cross-service state without tight coupling or gRPC overhead. Both files live under `/shared/doctor/` for backward compatibility with any existing consumers.
- **Serenity requires LLM repair** — container restart recovery alone earns 0.5 points; LLM-repaired code chaos earns 4.0. This ensures serenity actually proves cognitive capability, not just container restart behaviour.
- **docker.sock read-only** — monkey can stop/start/restart containers but cannot build images or run arbitrary shell commands via the socket.
- **Non-vital files only** — fault injection never targets `main.py`, `agent_core.py`, `tools.py`, `prompt_builder.py`, etc. This bounds the blast radius to files that won't kill the candidate on restart.
- **Three semantic fault types** that all pass `ast.parse` — tests the live runtime, not the static parser. Only the LLM can reliably detect and fix them.
- **Scheduler reads config each cycle** — mode and interval changes take effect within 60 seconds without a container restart.
- **In-memory drill history** (last 50) intentionally not persisted — each run is a fresh experiment; the serenity score provides the durable signal.

---

## Operational Guide

### First Deploy

```bash
docker compose build gaia-monkey
docker compose up -d gaia-monkey
# Restart web and doctor to pick up MONKEY_ENDPOINT env var
docker restart gaia-web gaia-doctor
```

### Running a Manual Drill

```bash
# Enter meditation first (required for container/code drills)
curl -X POST http://localhost:6420/meditation/enter

# Inject chaos (picks drill type from config)
curl -X POST http://localhost:6420/chaos/inject

# Or target a specific drill type
curl -X POST http://localhost:6420/chaos/code \
  -H 'Content-Type: application/json' \
  -d '{"targets": ["gaia-core-candidate"]}'

# Check serenity progress
curl http://localhost:6420/serenity
```

### Enabling Scheduled Mode

```bash
curl -X POST http://localhost:6420/config \
  -H 'Content-Type: application/json' \
  -d '{"mode": "scheduled", "schedule_interval_hours": 6}'
```

### Checking Dashboard

Navigate to the Commands tab in the Mission Control dashboard. The Chaos Monkey group shows the current mode, serenity badge, and the ⚡ Inject Chaos button.

---

## Key Relationships

```
gaia-monkey:6420
    │
    ├── ← gaia-web:6414          /api/chaos/* proxy + serenity badge
    ├── ← gaia-doctor:6419       POST /serenity/break on vital failure
    ├── → gaia-core:6415         POST /api/repair/structural (Tier 2 LLM repair)
    ├── → docker.sock            stop/start/restart candidate containers
    ├── → promptfoo eval         linguistic chaos subprocess
    └── → /shared/doctor/        serenity.json, defensive_meditation.json
```

# Dev Journal: Immune System 3.0 — Chaos Monkey Validated
**Date:** 2026-03-06
**Era:** Sovereign Autonomy
**Topic:** Structural Sovereignty, Post-Remediation Audit, and Empirical Chaos Testing

---

## Overview

Immune System 3.0 represents the transition from **Reactive Remediation** to **Proactive Structural Validation with Immediate Recovery Verification**. Following a series of structural failures during the Bicameral Mind implementation and five rounds of Chaos Monkey testing, the immune system has been hardened, debugged, and empirically verified end-to-end.

This journal consolidates all Immune System 3.0 components and their verified behavior under live-fire conditions.

---

## Architecture: The Four Layers

### Layer 1 — The Production Lock (Sovereign Shield)
**File:** `gaia-mcp/gaia_mcp/tools.py`

- `py_compile` gate on all `.py` writes (`ai_write`, `write_file`, `replace`)
- Physically refuses modification to live service directories unless `BREAKGLASS_EMERGENCY=1`
- Enforces candidate-first workflow — production cannot be modified by accident or by a "distracted" model

### Layer 2 — The Watchdog (gaia-doctor)
**File:** `gaia-doctor/doctor.py`

Health monitoring loop with tiered remediation:

| Component | Behavior |
|-----------|----------|
| **Health Polling** | 15s interval, HTTP GET to each service's `/health` endpoint |
| **Failure Threshold** | 2 consecutive failures before remediation triggers |
| **Structural Audit** | `ast.parse` scan of all `.py` files before any restart (the "Quarantine Gate") |
| **Production Restart** | `docker compose up -d --force-recreate` with circuit breaker (max 2/30min) |
| **Candidate Restart** | `docker compose` with HA overlay (`docker-compose.candidate.yml` + `docker-compose.ha.yml`) |
| **Post-Remediation Verify** | NEW — `_verify_recovery()` confirms health 5s after restart |
| **Alarm System** | Circuit breaker trips → alarm written to `/shared/doctor/alarms.json`, auto-clears on recovery |
| **Log Scanning** | Incremental scan of service logs for irritation patterns (PermissionError, TimeoutError, Sovereign Shield blocks, etc.) |
| **Code Change Audit** | Detects `.py` mtime changes → runs ruff (F821, E999) then pytest → auto-restart only if both pass |

### Layer 3 — The Surgeon (HA Cross-Stack Healing)
**File:** `gaia-core/gaia_core/cognition/structural_surgeon.py`, `gaia-core/gaia_core/main.py`

When the structural audit finds broken code:

1. **Tier 1 — Ruff Auto-Fix:** `ruff check --fix` on the broken file (handles import sorting, trailing commas, etc.)
2. **Tier 2 — HA Surgery:** If Tier 1 fails, doctor sends the broken code + error message to gaia-core's `/api/repair/structural` endpoint. The stable production Thinker model diagnoses and generates a fix. gaia-core validates with `ast.parse` and writes the repair directly (doctor has read-only mount).

Surgeon uses a **windowed approach** (±40 lines around the error, snapped to blank-line boundaries) to stay within the model's context budget (~1,400–2,000 tokens per repair).

### Layer 4 — The Dissonance Probe
**File:** `gaia-doctor/doctor.py` (`get_dissonance_report()`)

- SHA-256 hash comparison between live and candidate Vital Organs (agent_core.py, main.py, discord_interface.py, cognition_packet.py)
- Reports parity percentage and divergent files every poll cycle
- Alerts when Sync Parity drops below 100%

---

## Bugs Found & Fixed via Chaos Monkey Testing

### Round 1–4 (Pre-Session)
See `2026-03-06_chaos_monkey_5_surgical_repair.md` for full details on the 7 root causes found during Chaos Monkey 5: context overflow, surgeon SyntaxError, window cut mid-docstring, spurious error lines, doctor read-only mount, urllib scope, duplicate global declaration.

### Round 5–6 (This Session) — Three Bugs + One Architectural Gap

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 1 | **gaia-web health URL wrong** | Doctor used `http://localhost:6414/health` — unreachable from inside the doctor container (different network namespace) | Changed to `http://gaia-web:6414/health` to match all other services using Docker hostnames |
| 2 | **`import requests` in stdlib-only service** | `docker_restart()` autonomous diagnostics block imported `requests` (a third-party library) in a service that must be stdlib-only | Replaced with `urllib.request.Request()` + `urlopen()` (already imported at file top). Also fixed URL from `localhost` to `gaia-core` |
| 3 | **Candidate tools.py stray parenthesis** | Line 105 had `"memory_query": lambda p: _memory_query_impl(p), (` — residual from Chaos Monkey 5 injection that the Surgeon only partially cleaned | Removed stray `(` — file now passes `ast.parse` and `py_compile` |
| 4 | **No post-remediation health check** | After a successful restart, doctor waited up to 60s (next poll cycle) before confirming recovery | Added `_verify_recovery(name, url, delay=5)` — called after both `docker_restart()` and `restart_candidate()` succeed |

### Deployment Note
gaia-doctor's code is **COPYed** in its Dockerfile (not volume-mounted like other services). Source changes require `docker compose build gaia-doctor && docker compose up -d gaia-doctor`, not just `docker restart`.

---

## Chaos Monkey Verification Log

### Test 1: `docker kill gaia-mcp`
```
00:27:06  Chaos bolt fired
00:27:30  gaia-mcp is DOWN (2 consecutive failures)         [+24s]
00:27:30  Structural audit PASSED
00:27:30  REMEDIATION: docker compose recreate (attempt 2/2)
00:27:30  Successfully restarted gaia-mcp
00:27:35  POST-REMEDIATION: gaia-mcp confirmed healthy      [+29s total]
```

### Test 2: `docker kill gaia-mcp` (repeat)
```
00:20:01  Chaos bolt fired
00:20:31  gaia-mcp is DOWN (2 consecutive failures)         [+30s]
00:20:31  Structural audit PASSED
00:20:31  REMEDIATION: docker compose recreate (attempt 1/2)
00:20:32  Successfully restarted gaia-mcp
00:20:37  POST-REMEDIATION: gaia-mcp confirmed healthy      [+36s total]
```

**Recovery timeline:** Kill → detect (~24–30s, 2 poll cycles) → structural audit → compose recreate → **confirmed healthy in 5s** (vs. waiting up to 60s before the post-remediation fix).

---

## Complete Recovery Flow (As-Built)

```
Service dies
    │
    ▼
Poll cycle detects failure (15s interval)
    │
    ▼
Consecutive failures ≥ 2?  ──No──▶ Log debug, wait
    │ Yes
    ▼
Run structural audit (ast.parse all .py files)
    │
    ├── FAIL ──▶ Attempt Tier 1 repair (ruff --fix)
    │               │
    │               ├── PASS ──▶ Continue to restart
    │               │
    │               └── FAIL ──▶ Tier 2 (HA Surgery via gaia-core)
    │                               │
    │                               ├── PASS ──▶ Continue to restart
    │                               │
    │                               └── FAIL ──▶ ⛔ QUARANTINE (no restart)
    │
    ├── PASS
    ▼
Check circuit breaker (max 2 restarts / 30min)
    │
    ├── TRIPPED ──▶ 🚨 Raise alarm, dispatch diagnostics, stop
    │
    ├── CLEAR
    ▼
Check cooldown (300s between restarts)
    │
    ├── ACTIVE ──▶ Skip, wait
    │
    ├── CLEAR
    ▼
docker compose up -d --force-recreate <service>
    │
    ├── SUCCESS
    ▼
_verify_recovery(name, url, delay=5s)
    │
    ├── HEALTHY ──▶ Reset failures=0, clear alarm, log recovery ✅
    │
    └── UNHEALTHY ──▶ Log warning, next poll will retry
```

---

## System Pulse

| Metric | Status |
|--------|--------|
| **Self-Healing** | EMPIRICALLY PROVEN (6 Chaos Monkey rounds) |
| **Post-Remediation Verify** | ACTIVE — 5s confirmation, not 60s poll wait |
| **Surgeon Context Budget** | 1,400–2,000 tokens per repair window |
| **Circuit Breaker** | 2 restarts / 30min rolling window |
| **Quarantine Gate** | ast.parse + ruff (F821, E999) before any restart |
| **Dissonance Probe** | SHA-256 parity check on Vital Organs every poll |
| **Stdlib Purity** | VERIFIED — no third-party imports in gaia-doctor |
| **Docker Hostname Resolution** | VERIFIED — all service URLs use container hostnames |

---

## Remaining Known Issues

1. **gaia-mcp-candidate ModuleNotFoundError** — Container enters restart loop with `No module named 'gaia_mcp'`. This is a volume mount / compose config issue, not an immune system bug. The quarantine gate correctly prevents infinite restart attempts.
2. **Doctor requires rebuild for code changes** — Unlike other services, doctor.py is COPYed into the image. Must `docker compose build gaia-doctor` for changes to take effect.

---

## Files Changed (This Session)

| File | Change |
|------|--------|
| `gaia-doctor/doctor.py` | Fixed gaia-web URL (localhost → gaia-web), replaced `import requests` with urllib, added `_verify_recovery()` post-remediation helper |
| `candidates/gaia-mcp/gaia_mcp/tools.py` | Removed stray `(` on line 105 (residual Chaos Monkey injection) |

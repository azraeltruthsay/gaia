# Dev Journal: Doctor Alarm Fix, Health Audit, and Codebase Review

**Date:** March 5, 2026

## Overview

Health audit session triggered by gaia-web going unhealthy (StatReload stall incident from yesterday). Reviewed all recent changes, ran a full doctor/immune system diagnostic, and identified + fixed two bugs in gaia-doctor introduced with the expanded implementation.

---

## Recent Architectural Changes (Context)

The most recent promotion (`78b8c2f`) was a major push: **Speculative Nano-First Pipeline** (0.24s pre-flight reflex before AgentCore), full-stack NDJSON streaming, eager model loading, Immune System 2.0 hardening with F821 high-severity triage and Dissonance Probe, plus GAIA's own expansion of gaia-doctor to include Code Audit, Irritation Monitoring, and Dissonance Detection.

GAIA also added:
- `gaia-core` and `gaia-mcp` to doctor's `"restart"` remediation (not just gaia-web)
- Code Audit on file mtime change: ruff F821/E999 lint → pytest before allowing auto-restart
- Log scanning against IRRITATION_PATTERNS (PermissionError, TimeoutError, Sovereign Shield, etc.)
- Dissonance probe via `ImmuneSystem.get_dissonance_report()` each poll cycle
- `/irritations` HTTP endpoint

---

## Diagnostic Findings

### All Services: Healthy
All 7 services (gaia-core, gaia-web, gaia-mcp, gaia-prime, gaia-audio, gaia-core-candidate, gaia-mcp-candidate) reported 0 consecutive failures. Tests: **383 gaia-core passed, 80 gaia-web passed.**

### gaia-core: Unstable Earlier
The doctor log showed gaia-core was restarted ~6 times and circuit breaker tripped ~10 times over the last few hours. This correlates with the Speculative Nano-First Pipeline promotion session — code was in mid-edit states, causing health failures that triggered restarts. Services are now stable.

### Bug 1: Stale Alarm on gaia-web

**Symptom:** `gaia-web` showed `alarmed: true` in doctor status despite being healthy with only 1 restart in window.

**Root Cause:** The alarm in `_alarmed_services` clears in two places:
1. When a service goes *unhealthy → healthy* (recovery path in `poll_cycle`)
2. When a successful `docker_restart()` runs after alarm was set

But if the service was **already healthy** when the circuit breaker tripped (the common case — the CB fires on the 2nd restart attempt, not on a health failure), `_service_state[name]["healthy"]` stays `True` and the recovery path never fires. Alarm persists indefinitely.

**Fix:** Added a third clearing path in `poll_cycle()`: when a service is healthy AND in `_alarmed_services`, check if its restart history window has expired. If no restarts remain in the rolling window, clear the alarm and log:
```python
elif name in _alarmed_services:
    now_t = time.monotonic()
    in_window = [t for t in _restart_history.get(name, []) if now_t - t < PROD_RESTART_WINDOW]
    if not in_window:
        log.info("%s alarm cleared — restart window expired", name)
        _alarmed_services.discard(name)
```

### Bug 2: `docker exec -t` Without a TTY

**Symptom:** 15 "CodeAudit: Tests Failed" irritations recorded against gaia-core and gaia-web, despite all tests passing when run manually.

**Root Cause:** `run_service_tests()` was calling `docker exec -t name python -m ruff ...` and `docker exec -t name python -m pytest ...`. The `-t` flag allocates a pseudo-TTY. When called from a subprocess with `capture_output=True` (no terminal), Docker may return a non-zero exit code even on test success.

**Fix:** Removed the `-t` flag from both `docker exec` calls. Running without TTY allocation is correct for automated subprocess invocations — consistent with GAIA's own test documentation (`docker compose exec -T`).

---

## Post-Fix State
- `Alarms: []` — all alarms cleared
- All services: `alarmed=False`, `restarts_in_window=0`
- Doctor restart history reset (new process)
- Tests passing, code audit will function correctly going forward

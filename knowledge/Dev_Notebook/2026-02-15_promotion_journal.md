# Promotion Pipeline — 2026-02-15

**Timestamp:** 2026-02-15T13:00:00
**Duration:** ~8m
**Services:** gaia-common, gaia-core, gaia-orchestrator, gaia-web
**Mode:** LIVE (manual — no candidate containers running)
**Result:** PASS

## What Was Promoted

Sleep Cycle Phase 2 GPU integration — wiring the sleep cycle to release/reclaim GPU through the orchestrator, plus Discord presence updates with sleep-aware styling.

### gaia-common
- `discord_connector.py` — `status_override` parameter on `update_presence()` for sleep dot color
- `idle_monitor.py` — `get_idle_minutes()` convenience method
- `service_client.py` — `get_orchestrator_client()` factory function

### gaia-core
- `sleep_cycle_loop.py` — GPU release/reclaim via orchestrator, SOA presence fallback, richer dream status text

### gaia-orchestrator
- `main.py` — `/gpu/sleep` and `/gpu/wake` endpoints for sleep cycle GPU lifecycle

### gaia-web
- `main.py` — `/presence` endpoint for remote Discord presence updates from gaia-core

## Stage Results

| Stage | Result |
|-------|--------|
| Validation (gaia-core) | PASS — ruff pass, 117 pytest pass |
| Validation (gaia-web) | PASS — ruff pass, 36 pytest pass |
| Validation (gaia-orchestrator) | PASS — Docker build OK (slim image, no ruff/pytest) |
| Validation (gaia-common) | PASS — validated transitively via gaia-core |
| File Promotion | PASS — all 4 services promoted (dependency order) |
| Service Restart | PASS — gaia-orchestrator, gaia-web, gaia-core restarted |
| Health Checks | PASS — 6/6 live services healthy |
| GPU Verification | PASS — live prime holds GPU (11,618 MiB), no candidate containers |

## Notes

- No candidate containers were running, so the automated `promote_pipeline.sh` was not usable (it requires candidate stack health checks). Promotion was done manually via `promote_candidate.sh` per service.
- gaia-prime, gaia-mcp, gaia-study had no code changes — left untouched.
- Permission errors on container-owned data files (session vectors, logs) during rsync are expected and non-blocking.

---

*Generated manually during promotion session*

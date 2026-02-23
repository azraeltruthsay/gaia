# Dev Journal — 2026-02-19: HA Failover + Network Resilience

**Author:** Claude (Opus 4.6) via Claude Code, with Azrael
**Session scope:** Two major features — network resilience improvements and HA hot standby failover
**Plan reference:** `knowledge/Dev_Notebook/2026-02-19_ha_failover_plan.md` (Rev 3)

---

## Context

On 2026-02-19, gaia-prime hung during inference, causing gaia-core to block indefinitely and two Discord messages to be lost. The response was twofold:

1. **Network Resilience** (earlier session) — retry-with-backoff, inference-level fallback, message queue persistence
2. **HA Failover** (this session) — candidate stack as hot standby with automatic traffic routing

This journal covers both, as they were developed and committed together.

---

## Network Resilience (Earlier Session)

### Changes

- **`gaia-common/gaia_common/utils/service_client.py`** — Added retry-with-backoff for transient failures (ConnectError, RemoteProtocolError, 502/503/504). Exponential backoff with jitter. Configurable `max_retries` and `base_delay`.
- **`gaia-common/gaia_common/utils/resilience.py`** — New utility: circuit breaker and retry decorators for generic async operations.
- **`gaia-core/gaia_core/models/vllm_remote_model.py`** — Added inference-level retry with backoff. Timeout distinction: ReadTimeout doesn't trigger retry (service is alive but slow).
- **`gaia-core/gaia_core/models/_model_pool_impl.py`** — Fallback chain integration: when prime inference fails, cascades to groq → openai → gemini.
- **`gaia-web/gaia_web/queue/message_queue.py`** — Persistent message queue with JSON file backing. Messages survive gaia-web restarts.
- **`gaia-orchestrator/gaia_orchestrator/state.py`** — Stale handoff reconciliation on startup (marks in-progress handoffs as FAILED after crash).
- **`gaia-orchestrator/gaia_orchestrator/main.py`** — Integrated stale handoff reconciliation into lifespan startup.

### Tests

- `gaia-common/tests/test_resilience.py` — Circuit breaker + retry decorator tests
- `gaia-common/tests/test_service_client_retry.py` — ServiceClient retry behavior tests
- `gaia-web/tests/test_message_queue_persistence.py` — Queue persistence + recovery tests
- `gaia-orchestrator/tests/test_state_reconciliation.py` — Handoff reconciliation tests

---

## HA Failover Implementation (6 Phases)

### Phase 1: ServiceClient Fallback + Maintenance Mode

**File: `gaia-common/gaia_common/utils/service_client.py`**

Added `fallback_base_url` parameter to `ServiceClient.__init__()`. After primary retries exhaust on retryable errors (ConnectError, RemoteProtocolError, 502/503/504), a single attempt is made against the fallback URL.

Key design decisions:
- **Timeouts do NOT trigger fallback** — service is alive but slow, routing elsewhere doesn't help
- **Maintenance mode** — file-based flag at `/shared/ha_maintenance`. Disables failover routing but NOT direct inter-service calls ("hybrid" maintenance)
- **Single fallback attempt** — no retry loop on fallback. If fallback also fails, original primary error is raised

Factory functions `get_core_client()` and `get_mcp_client()` read `CORE_FALLBACK_ENDPOINT` / `MCP_FALLBACK_ENDPOINT` env vars.

**Tests:** 10 tests in `gaia-common/tests/test_service_client_failover.py` — all passing.

### Phase 2: gaia-web Retry Failover

**Files: `gaia-web/gaia_web/utils/retry.py`, `main.py`, `discord_interface.py`**

Added `fallback_url` parameter to `post_with_retry()`. Same trigger conditions as Phase 1. Both web UI and Discord interfaces pass `CORE_FALLBACK_ENDPOINT` env var.

**Tests:** 6 tests in `gaia-web/tests/test_retry_failover.py` — all passing.

### Phase 3: Docker Compose HA Configuration

**New files:**
- `docker-compose.ha.yml` — Compose override layered on `docker-compose.candidate.yml` with `--profile ha`. Candidates point at LIVE prime/mcp/study, `restart: unless-stopped`, no GPU reservation.
- `scripts/ha_start.sh` — Convenience: start HA standby services
- `scripts/ha_stop.sh` — Convenience: stop HA standby services
- `scripts/ha_maintenance.sh` — Toggle maintenance mode (on/off/status)

### Phase 4: Session State Sync (One-Way)

**New file: `scripts/ha_sync.sh`**

One-way sync: live → candidate only. Two modes:
- `--incremental` (default) — `cp -u` (only newer files)
- `--full` — wipe candidate state + fresh copy from live

Syncs: `sessions.json`, `session_vectors/*.json`, `prime.md`, `prime_previous.md`, `Lite.md`

Design principles: no merge-back ever, sync pauses in maintenance mode, candidate cognitive state is read-only.

### Phase 4.5: Graceful Cognitive Checkpoint

**File: `gaia-core/gaia_core/main.py`**

- Added `POST /cognition/checkpoint` endpoint — calls `PrimeCheckpointManager.create_checkpoint()` and `LiteJournal.write_entry()`
- Added `_write_shutdown_checkpoints()` helper shared by endpoint and lifespan shutdown hook
- Lifespan shutdown section now writes checkpoints before stopping the sleep cycle loop

**File: `docker-compose.yml`**
- Added `stop_grace_period: 25s` to gaia-core (15s checkpoint + 10s headroom)

**File: `scripts/graceful_checkpoint.sh`**
- Shell script that POSTs to `/cognition/checkpoint` with 15s timeout. Graceful failure if core is already down.

**Tests:** 4 tests in `gaia-core/tests/test_checkpoint_endpoint.py` — all passing.

### Phase 5: Health Watchdog Enhancement

**File: `gaia-orchestrator/gaia_orchestrator/health_watchdog.py`** (rewritten)

Major enhancement from simple binary health polling to HA-aware monitoring:
- Monitors both live and candidate services
- Consecutive failure threshold (2) before declaring unhealthy — prevents flapping
- 4 HA states: `active`, `degraded`, `failover_active`, `failed`
- Integrated session sync (runs `ha_sync.sh --incremental` every poll cycle when HA active)
- HA status change notifications via `NotificationManager`
- Maintenance mode awareness (skips candidate evaluation + sync when maintenance ON)

**File: `gaia-orchestrator/gaia_orchestrator/models/schemas.py`**
- Added `HA_STATUS_CHANGE` notification type

**Tests:** 13 tests in `gaia-orchestrator/tests/test_health_watchdog.py` — all passing.

### Phase 6: Candidate-Core Env Routing

**File: `docker-compose.yml`**
- Added `CORE_FALLBACK_ENDPOINT` env var to gaia-web (empty = disabled by default)
- Added `MCP_FALLBACK_ENDPOINT` env var to gaia-core (empty = disabled by default)

**File: `docker-compose.ha.yml`**
- Updated comment documenting that HA override also configures live services' failover routing

---

## Test Summary

| Service | Test File | Count | Status |
|---------|-----------|-------|--------|
| gaia-common | test_service_client_failover.py | 10 | All pass |
| gaia-common | test_resilience.py | (from earlier) | All pass |
| gaia-common | test_service_client_retry.py | (from earlier) | All pass |
| gaia-core | test_checkpoint_endpoint.py | 4 | All pass |
| gaia-web | test_retry_failover.py | 6 | All pass |
| gaia-web | test_message_queue_persistence.py | (from earlier) | All pass |
| gaia-orchestrator | test_health_watchdog.py | 13 | All pass |
| gaia-orchestrator | test_state_reconciliation.py | (from earlier) | All pass |
| **Total (HA-specific)** | | **33** | **All pass** |

All tests run inside Docker containers as per standard process.

---

## Files Summary

### New Files (20)
| File | Purpose |
|------|---------|
| `docker-compose.ha.yml` | Compose override for HA mode |
| `scripts/ha_start.sh` | Start HA standby services |
| `scripts/ha_stop.sh` | Stop HA standby services |
| `scripts/ha_maintenance.sh` | Toggle maintenance mode |
| `scripts/ha_sync.sh` | One-way session sync (live → candidate) |
| `scripts/graceful_checkpoint.sh` | Write cognitive checkpoints before shutdown |
| `gaia-common/gaia_common/utils/resilience.py` | Circuit breaker + retry decorators |
| `gaia-common/tests/test_resilience.py` | Resilience utility tests |
| `gaia-common/tests/test_service_client_retry.py` | ServiceClient retry tests |
| `gaia-common/tests/test_service_client_failover.py` | ServiceClient failover tests |
| `gaia-core/tests/test_checkpoint_endpoint.py` | Checkpoint endpoint tests |
| `gaia-web/gaia_web/utils/retry.py` | Post-with-retry + failover |
| `gaia-web/tests/test_retry_failover.py` | Retry failover tests |
| `gaia-web/tests/test_message_queue_persistence.py` | Queue persistence tests |
| `gaia-orchestrator/gaia_orchestrator/health_watchdog.py` | HA-aware health watchdog |
| `gaia-orchestrator/tests/test_health_watchdog.py` | Health watchdog tests |
| `gaia-orchestrator/tests/test_state_reconciliation.py` | Handoff reconciliation tests |
| `knowledge/Dev_Notebook/2026-02-19_ha_failover_plan.md` | HA failover plan (Rev 3) |
| `candidates/gaia-orchestrator/gaia_orchestrator/health_watchdog.py` | Candidate copy |
| `candidates/gaia-web/gaia_web/utils/retry.py` | Candidate copy |

### Modified Files (14)
| File | Change |
|------|--------|
| `gaia-common/gaia_common/utils/service_client.py` | Retry + fallback + maintenance mode |
| `gaia-core/gaia_core/main.py` | Checkpoint endpoint + shutdown hook |
| `gaia-core/gaia_core/models/_model_pool_impl.py` | Fallback chain integration |
| `gaia-core/gaia_core/models/vllm_remote_model.py` | Inference retry with backoff |
| `gaia-web/gaia_web/main.py` | Fallback endpoint env var wiring |
| `gaia-web/gaia_web/discord_interface.py` | Fallback endpoint passthrough |
| `gaia-web/gaia_web/queue/message_queue.py` | Persistent message queue |
| `gaia-orchestrator/gaia_orchestrator/main.py` | Stale handoff reconciliation |
| `gaia-orchestrator/gaia_orchestrator/models/schemas.py` | HA_STATUS_CHANGE notification type |
| `gaia-orchestrator/gaia_orchestrator/state.py` | Stale handoff reconciliation logic |
| `docker-compose.yml` | stop_grace_period + failover env vars |
| `knowledge/Dev_Notebook/2026-02-19_cc_bootstrap_plan.md` | Updated |
| `knowledge/conversation_examples.md` | Updated |
| + candidate mirrors of all modified production files |

---

## Next Steps

1. **Integration testing** — Start HA stack, kill live core, verify failover works end-to-end
2. **SIGTERM verification** — `docker compose stop gaia-core` and check logs for checkpoint writes
3. **Promotion pipeline integration** — Add maintenance/checkpoint/sync stages to `promote_pipeline.sh`
4. **Dashboard HA widget** — Surface `ha_status` in the web dashboard

---

## Decisions & Rationale

- **One-way sync only** — Candidate state during development is experimental and disposable. No merge-back eliminates an entire class of state corruption bugs.
- **Hybrid maintenance mode** — Disables failover routing but allows direct inter-service calls. Candidate can still call live prime for inference testing.
- **Timeouts excluded from failover** — Service is alive but slow. Routing elsewhere doesn't help and adds latency.
- **25s stop_grace_period** — 15s for checkpoint + 10s headroom. Checkpoint writes are typically < 2s but we want safety margin.
- **Consecutive failure threshold (2)** — Prevents flapping from transient network hiccups.
- **File-based maintenance flag** — Simple, no API needed, visible to all services via shared Docker volume.

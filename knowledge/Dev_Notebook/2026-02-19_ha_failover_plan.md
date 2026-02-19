# HA Failover: Candidate Stack as Hot Standby

**Date:** 2026-02-19
**Status:** Draft — Rev 3 (SLO, SIGTERM, grace period, candidate health alerting)
**Author:** Claude (Opus 4.6) via Claude Code
**Revised by:** Seumas & Claude (Opus 4.6) — Rev 2: one-way session sync, graceful cognitive checkpoint, hybrid maintenance mode, eliminated merge-back complexity. Rev 3: explicit failover SLO, SIGTERM propagation verification, grace period headroom, proactive candidate health alerting.

---

## Context

On 2026-02-19, gaia-prime hung during inference, which caused gaia-core to block indefinitely. Two Discord messages were lost because gaia-web's POST to gaia-core failed. We've since added retry-with-backoff and inference-level fallback (Network Resilience Plan), but all of that happens within the same live stack. If gaia-core itself crashes or hangs, there's no alternate path — gaia-web retries hit the same dead service.

**Goal:** Keep the candidate stack (CPU services only) running as a hot standby after every promotion. When a live service is unreachable, traffic automatically routes to its candidate counterpart. A maintenance mode disables failover routing (but not inter-service calls) during development and promotion.

---

## Architecture Overview

```
                                    ┌──────────────────────┐
                                    │   gaia-prime (GPU)   │
                                    │   :7777              │
                                    └──────┬───────────────┘
                                           │ shared by both cores
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
              ┌─────┴──────┐         ┌─────┴──────┐        cloud fallback
              │ gaia-core  │         │ gaia-core  │        (groq/openai/
              │ (live)     │         │ (candidate)│         gemini)
              │ :6415      │         │ :6416      │
              └─────┬──────┘         └─────┬──────┘
                    │ primary               │ fallback
                    └──────────┬────────────┘
                               │
                         ┌─────┴──────┐
                         │ gaia-web   │
                         │ (live)     │
                         │ :6414      │
                         └────────────┘
```

**Port clarification:** Candidate-core exposes `:6416` externally (host) but listens on `:6415` internally (container). Inside the Docker network, services call `http://gaia-core-candidate:6415`. The `docker-compose.candidate.yml` maps `6416:6415`.

**Key insight:** Candidate-core points at **live** gaia-prime for inference (same Docker network). It gets the same GPU inference quality. If prime is also down, the existing cloud fallback chain (groq → openai → gemini) activates. No GPU needed on any candidate service.

---

## What Runs in HA Mode

| Service | Runs in HA? | Notes |
|---------|-------------|-------|
| gaia-core-candidate | Yes | CPU-only, points to live prime |
| gaia-mcp-candidate | Yes | CPU-only, tool sandbox |
| gaia-web-candidate | No | Discord bot is single-token; Docker auto-restart is sufficient for web |
| gaia-prime-candidate | No | Needs GPU; live prime is shared |
| gaia-study-candidate | No | Background training, not user-facing |
| gaia-orchestrator-candidate | No | Single coordinator is sufficient |

Only **2 services** need to run as hot standbys: `gaia-core-candidate` and `gaia-mcp-candidate`.

---

## Failover SLO

Be explicit about what this HA design guarantees and what it doesn't:

| Scenario | Expected behavior | Worst case |
|----------|-------------------|------------|
| gaia-core crashes | Failover to candidate within retry window (~6-10s). User's message is delivered. | User mid-conversation may lose the last ~30s of session context (sync interval). They may need to repeat their most recent message. |
| gaia-core slow (timeout) | No failover (by design). Retry on same service. | User waits for timeout + retries. Slow is not down — routing elsewhere doesn't help. |
| gaia-core + candidate both down | Cloud fallback chain activates (groq → openai → gemini) for inference. Core request itself fails. | Message lost. Same as pre-HA behavior. |
| gaia-prime hangs (the original incident) | gaia-core's inference timeout triggers cloud fallback. Core itself stays up. | Degraded inference quality (cloud model instead of local 3B). No message loss. |

**The SLO is:** No user-facing message loss when a single non-GPU service fails, with up to 30 seconds of session context staleness during failover. This is a significant improvement over the pre-HA state where any gaia-core failure = lost messages.

---

## Phase 1: Failover-Aware ServiceClient + Maintenance Mode

**Add `fallback_url` to ServiceClient** so any caller can specify a backup endpoint. On transient failure after retries are exhausted against the primary, try the fallback once. Maintenance mode check is implemented here since it guards the fallback logic.

### File: `gaia-common/gaia_common/utils/service_client.py`

**Changes:**
- Add `fallback_base_url: Optional[str] = None` to `__init__`
- In `get()`, `post()`, `delete()`: after primary retries exhaust, if `fallback_base_url` is set and maintenance mode is off, retry the same request against the fallback URL
- Add `_is_maintenance_mode()` check — reads `/shared/ha_maintenance` file existence (shared Docker volume, no API needed)

**Fallback trigger conditions — be explicit:**
- Triggers on: `httpx.ConnectError`, `httpx.RemoteProtocolError`, HTTP 502/503/504 (same as existing retry conditions)
- Does NOT trigger on: `httpx.TimeoutException` (timeout means the service is alive but slow — fallback won't help), HTTP 4xx (client errors are not transient)
- Single attempt on fallback (no retry loop — if fallback also fails, raise the original primary error)

```python
# In __init__:
self.fallback_base_url = fallback_base_url  # e.g. "http://gaia-core-candidate:6415"

# In get/post/delete, after primary fails with retryable error:
if self.fallback_base_url and not self._is_maintenance_mode():
    fallback_url = urljoin(self.fallback_base_url, path)
    # single attempt to fallback (no retry loop on fallback)
    ...

@staticmethod
def _is_maintenance_mode() -> bool:
    return Path("/shared/ha_maintenance").exists()
```

### Maintenance mode semantics

**What maintenance mode disables:** Automatic failover routing in ServiceClient and post_with_retry. When the primary is down, requests fail — no fallback attempt.

**What maintenance mode does NOT disable:** Direct inter-service calls. Candidate-core can still call live prime (`http://gaia-prime:7777`) during development/testing in maintenance mode. The maintenance flag only gates the *automatic fallback path*, not the explicit endpoint configuration. This is "hybrid" maintenance — you're developing on the candidate, which can reach live services for inference, but live traffic won't route to the candidate.

### File: `scripts/ha_maintenance.sh` (new)

```bash
#!/bin/bash
# Toggle HA maintenance mode
# Affects: failover routing in ServiceClient and post_with_retry
# Does NOT affect: direct inter-service calls (candidate → live prime)
MAINTENANCE_FILE="/gaia/GAIA_Project/shared/ha_maintenance"
case "${1:-status}" in
  on)   touch "$MAINTENANCE_FILE"
        echo "Maintenance mode ON — failover routing disabled" ;;
  off)  rm -f "$MAINTENANCE_FILE"
        echo "Maintenance mode OFF — failover routing enabled" ;;
  status)
        [ -f "$MAINTENANCE_FILE" ] \
          && echo "Maintenance mode: ON (failover routing disabled)" \
          || echo "Maintenance mode: OFF (failover routing enabled)" ;;
esac
```

### File: `gaia-common/tests/test_service_client_failover.py` (new)

Tests:
- Primary fails with ConnectError → fallback succeeds
- Primary fails with TimeoutException → fallback NOT attempted (timeout ≠ down)
- Primary fails + maintenance mode ON → fallback NOT attempted
- Primary succeeds → fallback never called
- Fallback also fails → original primary error raised

---

## Phase 2: Failover-Aware gaia-web Retry

**Add `fallback_url` to `post_with_retry()`** in gaia-web's retry helper.

### File: `gaia-web/gaia_web/utils/retry.py`

**Changes:**
- Add optional `fallback_url: str | None = None` parameter
- After all retry attempts on primary URL are exhausted (for retryable errors only — ConnectError, RemoteProtocolError, 502/503/504 — NOT timeouts), try a single POST to `fallback_url`
- Check maintenance mode via same `/shared/ha_maintenance` file

### Files: `gaia-web/gaia_web/discord_interface.py`, `gaia-web/gaia_web/main.py`

**Changes:**
- Read `CORE_FALLBACK_ENDPOINT` env var (default: `http://gaia-core-candidate:6415`)
- Pass it as `fallback_url` to `post_with_retry()` calls in `_handle_message()`, `process_user_input()`, `process_audio_input()`

### File: `gaia-web/tests/test_retry_failover.py` (new)

Test: primary exhausts retries → fallback attempted → succeeds. Timeout → no fallback.

---

## Phase 3: Docker Compose HA Configuration

### File: `docker-compose.ha.yml` (new — compose override)

A thin override file layered on top of `docker-compose.candidate.yml`. Activated with:
```bash
docker compose -f docker-compose.candidate.yml -f docker-compose.ha.yml --profile ha up -d
```

**Contents:**
```yaml
services:
  gaia-core-candidate:
    profiles: ["ha", "full"]
    restart: unless-stopped
    environment:
      # Point at LIVE prime (shared GPU inference)
      - PRIME_ENDPOINT=http://gaia-prime:7777
      # Point at LIVE mcp as primary (candidate-mcp as internal fallback)
      - MCP_ENDPOINT=http://gaia-mcp:8765/jsonrpc
      # Same study endpoint
      - STUDY_ENDPOINT=http://gaia-study:8766
    deploy:
      resources: {}  # No GPU reservation

  gaia-mcp-candidate:
    profiles: ["ha", "full"]
    restart: unless-stopped
    deploy:
      resources: {}  # No GPU reservation
```

**Note on restart policy:** Candidates currently have `restart: "no"` (intentional — failed experiments shouldn't loop). The ha.yml override changes this to `unless-stopped` only for HA-profiled services. In dev mode (no HA profile), the original policy applies.

**No changes to live compose** — all HA config is additive.

### File: `scripts/ha_start.sh` (new)

```bash
#!/bin/bash
# Start HA hot standby services
docker compose -f docker-compose.candidate.yml \
               -f docker-compose.ha.yml \
               --profile ha up -d
echo "HA standby active: gaia-core-candidate, gaia-mcp-candidate"
```

### File: `scripts/ha_stop.sh` (new)

```bash
#!/bin/bash
# Stop HA hot standby services
docker compose -f docker-compose.candidate.yml \
               -f docker-compose.ha.yml \
               --profile ha down
echo "HA standby stopped"
```

---

## Phase 4: Session State Sync (One-Way: Live → Candidate)

### Design Principles

1. **One-way sync only.** Live → Candidate. Never Candidate → Live. Candidate state during development is experimental and may be corrupted by testing.
2. **No merge-back.** After successful promotion, wipe candidate state and re-sync from live. Corrupted test sessions are disposable.
3. **Sync pauses in maintenance mode.** During development, candidate runs with stale data — that's fine, it's being tested, not serving real traffic.
4. **Cognitive state (prime.md, lite.md) is read-only on candidate.** Candidate doesn't maintain its own cognitive continuity. It's a standby, not a separate consciousness.

### Session State Inventory

| Component | Path (inside container) | Size | Sync? |
|-----------|------------------------|------|-------|
| Active sessions | `/shared/sessions.json` | ~172 KB | Yes — one-way |
| Session vectors | `/shared/session_vectors/*.json` | ~2.5 MB | Yes — one-way |
| Prime checkpoint | `/shared/sleep_state/prime.md` | ~1-2 KB | Yes — read-only copy |
| Lite journal | `/shared/lite_journal/Lite.md` | ~5-10 KB | Yes — read-only copy |
| Vector archive | `/shared/session_vectors/archive/` | ~1-2 MB | No — historical, not needed for failover |

**Total sync payload:** ~2.7 MB per cycle. Completes in milliseconds.

### Sync Mechanism

Implement as a function in the health watchdog (`health_watchdog.py`), which already has a 30-second polling loop and service awareness. No new sidecar needed.

```python
async def _sync_session_state(self):
    """One-way sync: live gaia-shared → candidate gaia-candidate-shared.

    Runs every health check cycle (30s) when:
    - HA mode is active (candidate services are running)
    - Maintenance mode is OFF

    Skipped when maintenance mode is ON (development in progress).
    """
    if self._is_maintenance_mode():
        return

    # rsync or docker cp from gaia-shared to gaia-candidate-shared
    # Only sync: sessions.json, session_vectors/*.json, prime.md, Lite.md
    # Exclude: archive/, prime_history/, lite_history/
```

**Implementation options (in order of preference):**
1. **Docker volume mount overlap:** Mount `gaia-shared` as read-only at `/shared-live` in the candidate container. Candidate reads from `/shared-live`, writes to its own `/shared`. Watchdog copies on each cycle.
2. **`docker cp` from orchestrator:** Orchestrator (which has Docker socket access) copies files between volumes.
3. **HTTP sync endpoint:** gaia-core exposes `GET /state/snapshot` → candidate polls it. Adds coupling, less preferred.

Option 1 is simplest and requires only a compose volume change.

### The Lifecycle

```
Normal HA operation:
  watchdog syncs live → candidate every 30s
  candidate has state at most 30s stale
  failover loses at most the last message in one conversation

Maintenance mode ON (development):
  sync pauses
  candidate accumulates dirty test state
  candidate can still call live prime for inference (hybrid mode)
  observer catches issues, developer iterates

Promotion (after successful validation):
  1. Enable maintenance mode
  2. Write cognitive checkpoints (Phase 4.5)
  3. Wipe candidate state: rm -rf /shared/* on candidate volume
  4. Promote code: copy candidate source → live source
  5. Rebuild and restart live containers
  6. Re-sync: live → candidate (fresh state)
  7. Disable maintenance mode
  8. HA resumes with clean state on both sides
```

### Integration with promote_pipeline.sh

```bash
# Stage 0: Enable maintenance mode + write cognitive checkpoints
bash scripts/ha_maintenance.sh on
bash scripts/graceful_checkpoint.sh    # Phase 4.5

# ... existing promotion stages ...

# Stage 8: Wipe candidate state + re-sync from live
bash scripts/ha_sync.sh --full         # wipe + fresh copy

# Stage 9: Re-enable HA
bash scripts/ha_maintenance.sh off
```

---

## Phase 4.5: Graceful Cognitive Checkpoint on Shutdown

### The Problem

When live containers restart during promotion, cognitive state (prime.md, lite.md) must be persisted first. Without this, GAIA loses her working memory across restarts — she wakes up with no context about what she was doing.

### The Requirement

**Before any container shutdown or sleep, write cognitive checkpoints.** This applies to:
- `docker compose stop/restart/down` of gaia-core
- Sleep cycle entry (already partially implemented)
- Promotion pipeline restart

### File: `scripts/graceful_checkpoint.sh` (new)

```bash
#!/bin/bash
# Write cognitive checkpoints before container shutdown
# Called by promote_pipeline.sh and ha_maintenance.sh

echo "Writing cognitive checkpoints..."

# Trigger prime.md write via gaia-core's existing endpoint
curl -s -X POST http://localhost:6415/cognition/checkpoint \
  --connect-timeout 5 \
  --max-time 15 \
  || echo "WARN: gaia-core checkpoint failed (may already be down)"

echo "Checkpoints written."
```

### File: `gaia-core` — checkpoint endpoint

Add `POST /cognition/checkpoint` to gaia-core that:
1. Calls `PrimeCheckpointManager.save_checkpoint()` — writes prime.md
2. Calls `LiteJournal.flush()` — writes current Lite.md state
3. Returns `{"status": "checkpointed", "prime": true, "lite": true}`

This is also the endpoint the SIGTERM handler calls internally.

### Docker pre-stop integration

In `docker-compose.yml`, add a stop grace period:
```yaml
gaia-core:
  stop_grace_period: 25s
  # Container receives SIGTERM, has 25s to write checkpoints before SIGKILL
  # Budget: ~15s for checkpoint write + 10s headroom for slow I/O
```

**Grace period rationale:** The checkpoint curl has a 15s max-time. Add 10s headroom for slow disk I/O, large session serialization, or prime needing to finish a current inference call before responding. 25s total is conservative but safe — checkpoint writes are typically < 2s, so the headroom is a buffer, not the expectation.

### SIGTERM propagation — verify this works

The application must catch SIGTERM and call checkpoint logic. This ensures cognitive continuity even on unexpected `docker compose down` calls.

**Verification requirement:** gaia-core runs under uvicorn, which may be launched via a shell entrypoint script. If the container's PID 1 is a shell (bash/sh), SIGTERM goes to the shell, NOT to the Python process. This must be verified during implementation:

1. **Check the Dockerfile entrypoint.** If it's `CMD ["python", "-m", "uvicorn", ...]` (exec form), PID 1 is the Python process — SIGTERM propagates correctly.
2. **If it's `CMD python -m uvicorn ...`** (shell form) or an entrypoint script, PID 1 is `/bin/sh` — SIGTERM is swallowed. Fix by using `exec` in the entrypoint script or switching to exec form.
3. **Test:** `docker compose stop gaia-core && docker compose logs gaia-core | grep checkpoint` — verify the checkpoint log line appears before the container exits.

The SIGTERM handler in the application:
```python
import signal

def _handle_sigterm(signum, frame):
    """Write cognitive checkpoints on graceful shutdown."""
    logger.info("SIGTERM received — writing cognitive checkpoints...")
    # Call checkpoint logic synchronously (we're shutting down)
    prime_checkpoint_manager.save_checkpoint()
    lite_journal.flush()
    logger.info("Checkpoints written. Shutting down.")
    raise SystemExit(0)

signal.signal(signal.SIGTERM, _handle_sigterm)
```

**Note:** If uvicorn is handling SIGTERM for graceful worker shutdown, the handler may need to be registered as a lifespan shutdown hook instead of a raw signal handler, to avoid racing with uvicorn's own shutdown sequence. Check uvicorn's `on_shutdown` lifespan event.

---

## Phase 5: Health Watchdog Enhancement

### File: `gaia-orchestrator/gaia_orchestrator/health_watchdog.py`

**Changes:**
- Add candidate services to monitoring: `gaia-core-candidate:6415`, `gaia-mcp-candidate:8765`
- Track consecutive failure counts (not just binary healthy/unhealthy)
- Log HA failover events: "gaia-core unhealthy for 3 consecutive checks, HA candidate is healthy — failover active"
- Surface HA status in `get_status()`: which services are primary, which are in failover
- **Add session state sync** (Phase 4): run `_sync_session_state()` on each health check cycle when HA is active and maintenance mode is OFF
- **Proactive candidate health alerting** (see below)

No auto-remediation — just observability, sync, and alerting. Docker `restart: unless-stopped` handles actual restarts.

### Candidate Health Alerting

**The gap:** If the candidate is unhealthy when live goes down, failover silently fails. Users discover this only when messages disappear — the worst way to find out.

**The fix:** When the watchdog detects a candidate service is unhealthy, immediately emit a degraded-HA alert:

```
HA DEGRADED: gaia-core-candidate unhealthy for 3 consecutive checks.
If gaia-core (live) fails, failover will NOT work.
```

**Alert channels:**
- Log at `WARNING` level (always)
- Write to `prime.md` observation: *"HA standby degraded — candidate-core unhealthy"* (so GAIA is self-aware of reduced resilience)
- Surface in dashboard `/api/system/services` as `"ha_status": "degraded"` (vs `"active"` or `"disabled"`)
- Optionally: push to Discord notifications channel if `NotificationManager` supports it

**Alert states:**

| Live | Candidate | HA Status | Alert? |
|------|-----------|-----------|--------|
| Healthy | Healthy | `active` | No |
| Healthy | Unhealthy | `degraded` | Yes — "failover will fail if primary goes down" |
| Unhealthy | Healthy | `failover_active` | Yes — "primary down, traffic routing to candidate" |
| Unhealthy | Unhealthy | `failed` | Yes — "both primary and candidate down" |

This closes the observability loop — you know the HA safety net is intact before you need it, not after it's too late.

---

## Phase 6: Candidate-Core Env Routing

### File: `docker-compose.candidate.yml`

**Changes to gaia-core-candidate service:**
- Add `CORE_FALLBACK_ENDPOINT` and `MCP_FALLBACK_ENDPOINT` env vars (pointing to candidate counterparts)
- Ensure candidate-core's `PRIME_ENDPOINT` can be overridden by ha.yml to point at live prime

This is already mostly handled by Phase 3's override file, but the base candidate compose needs the env var definitions for the fallback endpoints.

---

## Implementation Order

```
Phase 1: ServiceClient fallback + maintenance check   ← foundation, includes _is_maintenance_mode()
Phase 2: gaia-web retry failover                      ← depends on Phase 1 pattern
Phase 3: docker-compose.ha.yml + scripts              ← independent, can parallel with 1-2
Phase 4: Session state sync (one-way)                 ← depends on Phase 3 (volume mounts)
Phase 4.5: Graceful cognitive checkpoint               ← independent, can parallel with 4
Phase 5: Health watchdog (monitoring + sync)           ← depends on Phase 4 (sync logic)
Phase 6: Candidate-core env routing                   ← depends on Phase 3
```

Phases 1+2 (code) and 3 (compose) can be done in parallel. Phase 4 and 4.5 are independent of each other. Phase 5 integrates monitoring + sync.

---

## Verification

1. **Unit tests:** ServiceClient failover (connect error → fallback, timeout → no fallback, maintenance → no fallback), retry failover, maintenance mode check
2. **Integration test — core failover:**
   - Start HA stack: `bash scripts/ha_start.sh`
   - Verify session sync is running (watchdog logs show sync cycles)
   - Stop live gaia-core: `docker compose stop gaia-core`
   - Send Discord message or POST to `/process_user_input`
   - Verify: gaia-web retries fail on live core → routes to candidate-core → response arrives
   - Verify: candidate-core has recent session context (not stale)
   - Restart live core: `docker compose start gaia-core`
   - Verify: next request goes to live core (primary)
3. **Integration test — maintenance mode (hybrid):**
   - Enable: `bash scripts/ha_maintenance.sh on`
   - Verify: session sync pauses
   - Stop live gaia-core → requests fail (no failover to candidate)
   - But: candidate can still call live prime directly (hybrid mode)
   - Disable: `bash scripts/ha_maintenance.sh off`
   - Verify: failover and sync resume
4. **Integration test — graceful shutdown:**
   - Send `docker compose stop gaia-core`
   - Verify: prime.md and lite.md are written before container exits
   - Start gaia-core → verify cognitive state loaded from checkpoint
5. **Integration test — promotion cycle:**
   - Maintenance on → checkpoint → promote → wipe candidate state → rebuild → re-sync → maintenance off
   - Verify: both live and candidate have clean, consistent state
6. **Dashboard:** Check `/api/system/services` shows HA candidate status, maintenance mode, and sync status

---

## What This Does NOT Cover (Intentional Omissions)

- **Discord bot failover** — Single-token constraint. Docker auto-restart (5-15s) is sufficient.
- **gaia-web HA** — Dashboard/API could theoretically failover, but the Discord bot can't, so the value is limited. Revisit if API-only clients emerge.
- **gaia-prime HA** — GPU can't be shared. Cloud fallback chain (groq → openai → gemini) already handles this.
- **gaia-study HA** — Background training, not user-facing. Single instance is fine.
- **External load balancer** — Adds infrastructure complexity for marginal gain at current scale.
- **Candidate → Live merge-back** — Intentionally excluded. Candidate state during development is experimental and disposable. After promotion, wipe and re-sync from live. This eliminates an entire class of state corruption bugs.

---

## Files Summary

### New Files
| File | Purpose | Phase |
|------|---------|-------|
| `docker-compose.ha.yml` | Compose override for HA mode | 3 |
| `scripts/ha_start.sh` | Start HA standby services | 3 |
| `scripts/ha_stop.sh` | Stop HA standby services | 3 |
| `scripts/ha_maintenance.sh` | Toggle maintenance mode (failover routing only) | 1 |
| `scripts/ha_sync.sh` | Manual session sync / wipe+resync for promotion | 4 |
| `scripts/graceful_checkpoint.sh` | Write cognitive checkpoints before shutdown | 4.5 |
| `gaia-common/tests/test_service_client_failover.py` | ServiceClient failover tests | 1 |
| `gaia-web/tests/test_retry_failover.py` | Retry failover tests | 2 |

### Modified Files
| File | Change | Phase |
|------|--------|-------|
| `gaia-common/gaia_common/utils/service_client.py` | Add `fallback_base_url`, `_is_maintenance_mode()`, fallback trigger conditions | 1 |
| `gaia-web/gaia_web/utils/retry.py` | Add `fallback_url` parameter, maintenance check, trigger conditions | 2 |
| `gaia-web/gaia_web/discord_interface.py` | Pass `CORE_FALLBACK_ENDPOINT` to retry | 2 |
| `gaia-web/gaia_web/main.py` | Pass `CORE_FALLBACK_ENDPOINT` to retry, add HA + maintenance status to services endpoint | 2 |
| `docker-compose.candidate.yml` | Add `ha` profile, candidate-shared read-only live mount, env var definitions | 3, 6 |
| `docker-compose.yml` | Add `stop_grace_period: 25s` to gaia-core, verify exec-form entrypoint | 4.5 |
| `gaia-core (cognition)` | Add `POST /cognition/checkpoint` endpoint, SIGTERM handler | 4.5 |
| `gaia-orchestrator/gaia_orchestrator/health_watchdog.py` | Monitor candidates, consecutive failures, HA status, session sync | 5 |
| `scripts/promote_pipeline.sh` | Add maintenance on/checkpoint/wipe+resync/maintenance off stages | 4, 4.5 |

---

### Revision Log

| Date | Author | Changes |
|------|--------|---------|
| 2026-02-19 | Claude (Opus 4.6) via CC | Original specification (6 phases) |
| 2026-02-19 | Seumas & Claude (Opus 4.6) | Rev 2: Added Phase 4 (one-way session sync — live→candidate only, no merge-back). Added Phase 4.5 (graceful cognitive checkpoint on shutdown — prime.md/lite.md as shutdown prereq, SIGTERM handler). Clarified maintenance mode as "hybrid" (disables failover routing, not inter-service calls). Folded maintenance mode into Phase 1 (co-located with `_is_maintenance_mode()`). Added port clarification (6416 external / 6415 internal). Added explicit fallback trigger conditions (connect errors yes, timeouts no). Added restart policy note (candidates use `restart: "no"` by default, HA override changes to `unless-stopped`). Added promotion lifecycle (maintenance → checkpoint → wipe → promote → rebuild → resync → HA). Eliminated candidate→live merge-back as intentional omission. |
| 2026-02-19 | Seumas & Claude (Opus 4.6) | Rev 3: Added Failover SLO section with explicit guarantees and worst-case behavior per scenario. Increased `stop_grace_period` from 15s to 25s (15s checkpoint + 10s headroom). Added SIGTERM propagation verification requirements (exec-form entrypoint, PID 1 check, uvicorn lifespan hook consideration). Increased `graceful_checkpoint.sh` curl `--max-time` from 10s to 15s. Added proactive candidate health alerting in Phase 5 — four HA states (active/degraded/failover_active/failed) with alert channels (log, prime.md, dashboard, optional Discord). Closes the "discover failover failed only when users complain" observability gap. |

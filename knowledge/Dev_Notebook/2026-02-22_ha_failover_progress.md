# HA Mode Activation — Progress & Resume Point

**Date:** 2026-02-22
**Status:** Paused mid-failover-verification

## What's Done

### Infrastructure (all committed/deployed)
1. **Fallback endpoints set** in `docker-compose.yml`:
   - `MCP_FALLBACK_ENDPOINT=http://gaia-mcp-candidate:8765/jsonrpc` (gaia-core)
   - `CORE_FALLBACK_ENDPOINT=http://gaia-core-candidate:6415` (gaia-web)

2. **MCP synchronous fallback** added to `gaia-core/gaia_core/utils/mcp_client.py` (+ candidate):
   - `call_jsonrpc()` now tries `MCP_FALLBACK_ENDPOINT` on ConnectionError/Timeout

3. **Candidate gaia-common synced** — `service_client.py` and `resilience.py` copied from production

4. **`scripts/gaia_doctor.sh` created** — comprehensive health check (~550 lines):
   - 11-service registry, 9 check categories, --fix mode
   - Last run: 45 passed, 1 warning (tools_registry.py differs), 0 failures

5. **Candidate shared volume dirs fixed**:
   - Created `/shared/{council,lite_journal,temporal_states,timeline}` with uid 1000 ownership
   - Candidate no longer has permission denied on startup

6. **HA services running** — `ha_start.sh` executed, candidates healthy

## Where We Stopped — Failover Verification

### Retry + routing: CONFIRMED WORKING
- `gaia-web/gaia_web/utils/retry.py` (`post_with_retry`) correctly:
  - Retries 3x on primary with exponential backoff
  - Falls back to `fallback_url` after exhausting retries
  - Does NOT failover on timeouts (slow != down)
- Tested directly from inside gaia-web container — logs show clean retry → fallover path

### End-to-end failover: CANDIDATE CAN'T PROCESS PACKETS YET
When gaia-core is stopped and a request is routed to gaia-core-candidate:
- Candidate receives the request (HTTP connection established)
- Candidate returns errors during processing:
  - First test: `RemoteProtocolError` (connection dropped during processing)
  - Second test (after fixing shared dirs): `ReadError` (connection dropped ~11s in)
- Candidate logs need to be checked to see what's crashing during packet processing

### Likely candidate issues to investigate:
1. **Embed model load failure** — "Failed to lazy load embed model: Artifact of type=precompile already registered in mega-cache artifact factory"
2. **Model loading race** — candidate was lazy-loading `lite` model when request arrived
3. **Possible missing session/state data** — candidate may not have session vectors or other state needed for processing
4. **gaia-core DNS gone when stopped** — Docker removes DNS for stopped containers, so candidate trying to reach gaia-core for anything would fail

## Resume Steps
1. Start gaia-core stopped, send packet directly to candidate (`curl -X POST http://localhost:6416/process_packet ...`)
2. Check candidate logs (`docker logs gaia-core-candidate --since 30s`) to find exact crash point
3. Fix candidate processing issues
4. Re-test full failover path
5. Commit all HA work

## Files Modified (Not Yet Committed)
- `docker-compose.yml` — fallback endpoints
- `gaia-core/gaia_core/utils/mcp_client.py` — sync fallback
- `candidates/gaia-core/gaia_core/utils/mcp_client.py` — same
- `candidates/gaia-common/gaia_common/utils/service_client.py` — synced from prod
- `candidates/gaia-common/gaia_common/utils/resilience.py` — synced from prod
- `scripts/gaia_doctor.sh` — new comprehensive health check

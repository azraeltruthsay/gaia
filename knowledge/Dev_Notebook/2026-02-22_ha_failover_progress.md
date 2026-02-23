# HA Mode Activation — COMPLETE

**Date:** 2026-02-22 → 2026-02-23
**Status:** VERIFIED WORKING

## What's Done

### Infrastructure (all committed in `9f07904`)
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

6. **HA services running** — `ha_start.sh` executed, candidates healthy

## Failover Verification — COMPLETE

### Retry + routing: CONFIRMED
- `gaia-web/gaia_web/utils/retry.py` (`post_with_retry`) correctly:
  - Retries 3x on primary with exponential backoff (2s, 4s)
  - Falls back to `fallback_url` after exhausting retries
  - Does NOT failover on timeouts (slow != down)

### Direct candidate packet processing: CONFIRMED
- Sent well-formed CognitionPacket via `curl -X POST http://localhost:6416/process_packet`
- Candidate deserialized, assembled prompt, ran Lite model inference, returned 200 OK
- Response: coherent, finalized, complete

### End-to-end failover: CONFIRMED
- Stopped gaia-core (`docker compose stop gaia-core`)
- Sent request through gaia-web (`/process_user_input`)
- gaia-web logs show:
  ```
  POST http://gaia-core:6415/process_packet failed on attempt 1/3 (ConnectError), retrying in 2.0s...
  POST http://gaia-core:6415/process_packet failed on attempt 2/3 (ConnectError), retrying in 4.0s...
  Primary POST exhausted 3 attempts, attempting HA fallback to http://gaia-core-candidate:6415/process_packet
  HA fallback POST http://gaia-core-candidate:6415/process_packet succeeded
  ```
- Response received successfully via candidate
- Total failover latency: ~8s (retries) + ~105s (CPU inference) = ~113s

### Post-failover recovery: CONFIRMED
- Restarted gaia-core, verified it processes requests normally again

## Root Cause of Previous Failures

The candidate image was stale (built 2026-02-16, vs primary built 2026-02-21). It was missing:
1. **`gaia` user in `/etc/passwd`** — caused `getpwuid(): uid not found: 1000` errors when PyTorch tried to create its cache directory
2. This cascaded into the embed model failing to load (`Artifact of type=precompile already registered in mega-cache artifact factory`), which was actually a PyTorch `torch._dynamo.CacheArtifactFactory` double-registration triggered by the missing user
3. Without the embed model, the candidate worked anyway (embed is non-critical, graceful degradation), but earlier test failures were from malformed manual curl payloads (`KeyError: 'version'`)

**Fix:** Rebuilt candidate image from the existing Dockerfile (which already had the user creation). No code changes needed.

## Architecture Notes

- Candidates run on separate Docker volumes (`gaia-candidate-shared`) — isolated from production
- Failover uses HTTP-level retry + fallback (no service mesh needed)
- Candidate processes packets using the `lite` CPU model (~105s for simple responses)
- Embed model is optional — system degrades gracefully to full-history mode without it

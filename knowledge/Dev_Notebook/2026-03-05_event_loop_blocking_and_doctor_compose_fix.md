# Dev Journal: Event Loop Blocking & Doctor Compose Recreate
**Date:** 2026-03-05
**Era:** Sovereign Autonomy
**Topic:** Discord Timeout Root Cause & Two-Part Fix

## Problem
Discord users received "GAIA Core took too long to respond" errors. Messages from users timed out after 120s.

## Root Cause Analysis
A cascading failure involving two bugs:

1. **Event Loop Blocking**: `process_packet` in `main.py` calls `AgentCore.run_turn()`, a synchronous generator. Each `next()` call blocks the uvicorn event loop for seconds during llama_cpp inference (CPU-bound). While blocked, the `/health` endpoint cannot respond.

2. **Doctor False Positive**: `gaia-doctor` polls `/health` with a 5s timeout. When the event loop is blocked by inference, health checks time out → 2 consecutive failures → doctor restarts gaia-core mid-request → the in-flight Discord message is killed.

3. **Container Name Mangling**: Doctor used `docker restart <name>`, which preserves mangled container names (e.g., `2a85f751fcd3_gaia-web`). These mangled names cause DNS resolution issues on the Docker network.

## Fixes Applied

### 1. Thread Executor for Synchronous Inference (`main.py`)
Wrapped both blocking calls in `asyncio.run_in_executor()`:
- `generate_instant_reflex(packet)` — Nano pre-flight
- `next(gen)` for each `run_turn()` event — Main cognitive loop

This releases the event loop between inference steps, allowing `/health` and other endpoints to respond during processing.

### 2. Compose Recreate for Doctor Restarts (`doctor.py`)
Changed `docker_restart()` from `docker restart <name>` to `docker compose up -d --force-recreate <service>`. Benefits:
- Containers always get their correct compose-defined `container_name`
- Compose handles dependency ordering and health wait
- No more mangled names after auto-restarts
- Uses project root path (`/gaia/GAIA_Project`) since compose files are accessible via the project mount

### 3. Turn Serialization Semaphore (`main.py`)
Added `asyncio.Semaphore(1)` to `process_packet` to ensure only one cognitive turn runs at a time. Without this, concurrent Discord messages both invoked `run_turn` simultaneously, fighting over `gpu_prime` and causing:
- Model contention (both requests slow to >120s instead of one fast + one queued)
- Persistent typing indicator (two overlapping `channel.typing()` contexts)
- The first request timing out while the second was also stuck

The semaphore wraps `_run_loop` via `_run_loop_inner`, so the second request waits for the first to complete before starting its turn.

## Verification
- Single requests complete in ~11-48s depending on complexity
- Health endpoint responds during inference (thread executor keeps event loop free)
- Concurrent requests properly serialized (second waits for first, logs show "acquired turn semaphore" in order)
- Doctor reports all 7 services healthy, 0 failures, 0 restarts
- Nano reflex fires correctly before Prime refinement
- Response quality confirmed: full CognitionPacket with 0.93 confidence

## Files Changed
- `gaia-core/gaia_core/main.py` — `asyncio` import, `run_in_executor` for reflex + run_turn, `Semaphore(1)` for turn serialization
- `gaia-doctor/doctor.py` — `docker compose up -d --force-recreate` in `docker_restart()`
- `candidates/gaia-core/gaia_core/main.py` — bit-for-bit sync

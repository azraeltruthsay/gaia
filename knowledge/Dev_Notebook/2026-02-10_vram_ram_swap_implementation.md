# Dev Journal Entry: 2026-02-10 - VRAM↔RAM Hot-Swap Implementation

**Date:** 2026-02-10
**Author:** Claude Code (Opus 4.6) via Happy
**Last Updated:** 2026-02-10 (evening session — live testing + pivot to container stop/start)

## Context

GAIA's single RTX 5080 (16GB) is permanently held by gaia-prime (vLLM, 65% = ~10.4GB). When gaia-study needs the GPU for QLoRA fine-tuning ("Dreaming"), there was no mechanism to pause prime, free VRAM, let study train, and restore prime without a cold restart (30-60s).

The orchestrator already had a complete handoff state machine (`HandoffManager`, `GPUManager`, `StateManager`) in code, but the HTTP endpoints it calls on gaia-core and gaia-study didn't exist yet.

Proposal: `/gaia/GAIA_Project/knowledge/Dev_Notebook/2026-02-10_vram_ram_swap_proposal.md`

## Original Approach: vLLM Sleep Mode (Level 1)

The initial plan used vLLM's built-in sleep mode (`POST /sleep?level=1`) which offloads weights to CPU pinned RAM via CuMemAllocator. In theory this provides sub-second VRAM release and restore.

### What We Found During Live Testing

Sleep mode flags were already enabled (`--enable-sleep-mode`, `VLLM_SERVER_DEV_MODE=1`), and the `/sleep` and `/wake_up` endpoints responded correctly. However:

| Test | Expected VRAM freed | Actual VRAM freed | Why |
|------|---------------------|-------------------|-----|
| `POST /sleep?level=1` | ~10.4 GB (weights + KV) | **1.7 GB (KV cache only)** | Weights not tracked by CuMemAllocator |
| `POST /sleep?level=2` | ~10.4 GB | **1.4 GB** | Same issue |

**Root cause:** `--enforce-eager` is required on Blackwell (RTX 5080, sm_120 / compute capability 12.0) because `torch._dynamo` compilation crashes during CUDA graph capture with the attention layer shapes. We confirmed this by removing `--enforce-eager` — vLLM failed to start with a `RuntimeError` from `cuda_graph.py` during compilation.

With `--enforce-eager`, PyTorch uses its standard CUDA allocator for all tensor allocations. vLLM's `CuMemAllocator` (which implements the sleep offload) uses PyTorch's pluggable allocator API to intercept allocations, but this interception fails silently on PyTorch 2.7.0a0 (NGC 25.03). We confirmed:

```python
# Inside the engine core process:
allocator = CuMemAllocator.get_instance()
len(allocator.pointer_to_data)  # → 0 (nothing tracked)
```

Zero weight allocations tracked = nothing to offload to CPU = sleep only frees what vLLM directly manages (KV cache blocks).

**Conclusion:** vLLM sleep mode is architecturally correct but blocked by the Blackwell + enforce-eager + PyTorch 2.7.0a0 combination. This may resolve in a future NGC image or vLLM release.

## Revised Approach: Container Stop/Start

Since sleep mode can't fully offload on our hardware, we pivoted to using Docker container stop/start as the GPU swap mechanism. This is reliable, deterministic, and frees 100% of VRAM.

### Measured Performance

| Operation | VRAM Before | VRAM After | Time | Method |
|-----------|------------|------------|------|--------|
| **Release (stop)** | 12,806 MiB | 2,200 MiB | **<1 second** | `docker stop gaia-prime-candidate` |
| **Reclaim (start)** | 2,200 MiB | 12,857 MiB | **~60 seconds** | `docker start` + model load from disk |

The 60s cold start is the tradeoff vs sleep mode's theoretical 0.1-0.3s wake. But we get 10.6 GB freed instead of 1.7 GB — the full GPU is available for study/training.

### Startup Profile (from health polling)

```
 0s   docker start issued
 5s   container running, VRAM 3,519 MiB (CUDA context + early loading)
10s   VRAM 10,262 MiB (safetensors shards loaded, weights on GPU)
55s   VRAM 12,857 MiB (KV cache allocated)
60s   /health returns 200 (vLLM serving, Flash Attention selected)
```

## Changes Implemented

### 1. gaia-orchestrator: Docker-Based GPU Manager

**Modified file:** `candidates/gaia-orchestrator/gaia_orchestrator/gpu_manager.py`

New methods using Docker SDK (socket already mounted in orchestrator container):

| Method | What it does |
|--------|-------------|
| `stop_prime_container()` | `docker.containers.get("gaia-prime-candidate").stop()` — frees all VRAM in <1s |
| `start_prime_container()` | `container.start()` + health poll loop (120s max, 3s interval) |
| `request_release_from_core()` | Stop prime container → POST `/gpu/release` to core (demote model pool) |
| `request_reclaim_by_core()` | Start prime container → wait healthy → POST `/gpu/reclaim` to core (restore model pool) |

The orchestrator already had `docker>=7.0.0` in requirements and `/var/run/docker.sock` mounted.

### 2. gaia-core: Model Pool Management Endpoints (Simplified)

**Modified file:** `candidates/gaia-core/gaia_core/api/gpu_endpoints.py`

Stripped all `/sleep` and `/wake_up` proxy calls. Endpoints now only manage gaia-core's internal model pool state:

| Endpoint | Purpose |
|----------|---------|
| `GET /gpu/status` | Report GPU state (active/released/error), prime reachability, model pool status |
| `POST /gpu/release` | Demote `gpu_prime` from ModelPool (stash for restore), activate fallback chain |
| `POST /gpu/reclaim` | Health-check prime, restore `gpu_prime` in ModelPool, re-promote to 'prime' alias |

Key: gaia-core no longer talks to gaia-prime at all during release. The orchestrator handles the container lifecycle. Core just manages its own routing.

### 3. gaia-core: Session History Sanitization

**Modified file:** `candidates/gaia-core/gaia_core/memory/session_manager.py`

**Bug found during testing:** The "thinking" model (Heretic-Thinking fine-tune) wraps reasoning in `<think>...</think>` tags. These were being saved raw into session history, poisoning future context windows. The model would see its own `<think>` output in history and produce more think-only responses (2206 chars of reasoning, 0 chars of user-facing text), triggering the fallback: "I apologize, but I encountered an issue..."

**Fix:** Added `_strip_think_tags_robust()` call in `add_message()` before persisting assistant messages. Also skips saving entirely-empty messages (pure think tag responses that strip to nothing).

### 4. gaia-study: GPU Handoff Endpoints (unchanged from initial impl)

| Endpoint | Purpose |
|----------|---------|
| `POST /study/gpu-ready` | Ack from orchestrator that GPU is available |
| `POST /study/gpu-release` | Cleanup training, `torch.cuda.empty_cache()`, ack release |

### 5. gaia-prime: Sleep Mode Config (retained for future)

Sleep mode flags remain enabled in case the PyTorch/vLLM fix lands:
- `--enable-sleep-mode` in Dockerfile CMD and compose command
- `VLLM_SERVER_DEV_MODE=1` in compose environment

## Complete Handoff Protocol (Container Stop/Start)

### Prime → Study (orchestrator drives)

```
1. POST /handoff/prime-to-study           → orchestrator
2.   docker stop gaia-prime-candidate     → orchestrator (via Docker SDK)
3.   VRAM drops from ~13GB to ~2GB        → <1 second
4.   POST gaia-core:6415/gpu/release      → core demotes gpu_prime, fallback chain active
5.   Poll VRAM < 500MB                    → orchestrator verifies cleanup
6.   Transfer ownership to STUDY          → state manager
7.   POST gaia-study:8766/study/gpu-ready → study knows GPU is free
8.   Study starts QLoRA training          → uses freed VRAM (~10GB available)
```

### Study → Prime (orchestrator drives)

```
1. POST /handoff/study-to-prime             → orchestrator
2.   POST gaia-study:8766/study/gpu-release → study cleans up CUDA
3.   Poll VRAM < 500MB                      → orchestrator verifies cleanup
4.   Transfer ownership to CORE             → state manager
5.   docker start gaia-prime-candidate      → orchestrator (via Docker SDK)
6.   Poll /health until 200                 → ~60s (model load from disk)
7.   POST gaia-core:6415/gpu/reclaim        → core restores gpu_prime in pool
8.   GAIA inference via prime restored      → full GPU performance
```

## VRAM Budget

```
Normal (prime active):     10.4 GB prime + 2.4 GB KV cache = ~12.8 GB / 16 GB
After stop (released):     2.2 GB (desktop GUI only, no CUDA compute)
During QLoRA training:     ~6 GB (4-bit quantized base + LoRA + optimizer)
Available for training:    ~13.8 GB (more than enough)
CPU RAM during stop:       No impact (weights not in RAM, just on disk)
```

## Files Modified/Created

| File | Change |
|------|--------|
| `candidates/gaia-orchestrator/gaia_orchestrator/gpu_manager.py` | **MODIFIED** — Added Docker stop/start methods, updated release/reclaim to use container lifecycle |
| `candidates/gaia-core/gaia_core/api/gpu_endpoints.py` | **MODIFIED** — Removed sleep/wake proxy calls, simplified to model pool management only |
| `candidates/gaia-core/gaia_core/memory/session_manager.py` | **MODIFIED** — Strip think tags before saving assistant messages to session history |
| `candidates/gaia-core/gaia_core/api/__init__.py` | **NEW** — package init |
| `candidates/gaia-core/gaia_core/main.py` | Register GPU router, update endpoint listing |
| `candidates/gaia-study/gaia_study/server.py` | Add gpu-ready and gpu-release endpoints |
| `candidates/gaia-prime/Dockerfile` | Add `--enable-sleep-mode` to CMD |
| `docker-compose.candidate.yml` | Add `VLLM_SERVER_DEV_MODE=1` + `--enable-sleep-mode` |

## What Still Needs Work

1. **Orchestrator E2E test** — Test via `POST /handoff/prime-to-study` through the orchestrator's API
2. **Study auto-start on gpu-ready** — `/study/gpu-ready` currently just acknowledges; wiring auto-start of queued training sessions would complete the hands-free Dreaming cycle
3. **Adapter hot-loading after restart** — After study produces a new adapter and prime restarts, tell prime to load it via vLLM's LoRA support (`--enable-lora --max-loras 4` already configured)
4. **MemoryRequest relay (Phase 4)** — gaia-core can't yet *request* gaia-study to train; training is currently initiated via direct API calls to study
5. **Monitor PyTorch pluggable allocator fix** — When NGC ships a PyTorch build where CuMemAllocator works with `--enforce-eager`, revisit vLLM sleep mode for sub-second swap

## Verification Completed

| Test | Result |
|------|--------|
| vLLM sleep endpoints respond | Pass (200 OK, `is_sleeping: true/false`) |
| Sleep frees VRAM (KV cache only) | Pass but insufficient (1.7 GB, not 10.4 GB) |
| `--enforce-eager` removal crashes vLLM | Confirmed (torch._dynamo fails on sm_120) |
| Container stop frees all VRAM | Pass (12,806 → 2,200 MiB in <1s) |
| Container start restores inference | Pass (healthy in ~60s, VRAM back to ~12,857 MiB) |
| Docker SDK works from orchestrator | Pass (socket mounted, `docker.from_env()` functional) |
| Session history think tag cleanup | Pass (5 messages cleaned, Discord DM session restored) |
| Session manager sanitization | Pass (new assistant messages stripped before save) |

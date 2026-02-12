# Proposal: VRAM↔RAM Hot-Swap for GPU Time-Sharing Between Prime and Study

**Date:** 2026-02-10
**Author:** Claude Code (Opus 4.6) via Happy
**Status:** Proposal

## Problem

GAIA's single RTX 5080 (16GB VRAM) is currently held permanently by gaia-prime (vLLM, 65% = ~10.4GB). When gaia-study needs the GPU for QLoRA training, there is no mechanism to:

1. Pause gaia-prime and free the GPU
2. Let gaia-study train
3. Restore gaia-prime without a cold restart (~30-60s)

The orchestrator has a complete handoff state machine (`HandoffManager`, `GPUManager`, `StateManager`) but the actual endpoints it calls don't exist yet in gaia-core or gaia-study.

## Solution: vLLM Sleep Mode + Orchestrator Integration

**vLLM v0.15.1 has built-in "sleep mode"** — purpose-built for exactly this use case:

| Sleep Level | What Happens | Wake Time (7B) | CPU RAM Cost |
|-------------|-------------|-----------------|--------------|
| **Level 1** | Weights offloaded to CPU pinned RAM, KV cache discarded, CUDA graphs + JIT kernels preserved | **~0.1-0.3s** | ~model size (10-14GB) |
| **Level 2** | Everything discarded (weights + KV cache), only CUDA context preserved | **~0.8-2.6s** | Minimal |

For comparison, a full cold restart of gaia-prime takes 30-60 seconds (model loading, CUDA warm-up, graph compilation). Sleep Level 1 achieves the same outcome **100-200x faster** because it preserves the CUDA context and compiled kernels.

### Why Level 1 is Best for GAIA

- Our prime model is ~7B params in FP16 = ~14GB on disk, but vLLM applies KV quantization and optimization, so the CPU RAM buffer is manageable (~10-14GB)
- System has 64GB RAM — plenty of headroom for a temporary CPU buffer
- Level 1 preserves CUDA graphs and JIT kernels, so post-wake inference runs at **61-88% faster** than a cold-started model
- Level 2 would require `reload_weights()` after wake (reading from disk), adding 2-5s — unnecessary if we have the RAM

### Why NOT Naive PyTorch .to('cpu')

Since gaia-prime uses vLLM (not raw PyTorch), we can't directly call `.to('cpu')` on the model. vLLM manages its own memory via PagedAttention's block allocator. Sleep mode is vLLM's native API for this exact operation.

## Architecture

### Data Flow

```
 User asks GAIA to study/dream
          │
          ▼
 ┌─────────────────────────┐
 │   gaia-orchestrator     │
 │   POST /handoff/        │
 │   prime-to-study        │
 └────────┬────────────────┘
          │
          ▼
 ┌─────────────────────────┐
 │   gaia-prime            │
 │   POST /sleep?level=1   │ ◄── NEW: vLLM sleep endpoint
 │   Weights → CPU RAM     │     (requires VLLM_SERVER_DEV_MODE=1)
 │   VRAM freed (~250MB    │
 │   CUDA context remains) │
 └────────┬────────────────┘
          │
          ▼
 ┌─────────────────────────┐
 │   gaia-core             │
 │   POST /gpu/release     │ ◄── NEW: endpoint for orchestrator
 │   Acks release, routes  │
 │   inference to fallback  │
 │   chain (Groq/lite)     │
 └────────┬────────────────┘
          │
          ▼
 ┌─────────────────────────┐
 │   gaia-orchestrator     │
 │   Polls VRAM < 500MB    │
 │   Transfers ownership   │
 └────────┬────────────────┘
          │
          ▼
 ┌─────────────────────────┐
 │   gaia-study            │
 │   POST /study/gpu-ready │ ◄── NEW: endpoint, triggers QLoRA
 │   QLoRA training runs   │
 │   on free GPU           │
 └────────┬────────────────┘
          │ (training complete)
          ▼
 ┌─────────────────────────┐
 │   gaia-orchestrator     │
 │   POST /handoff/        │
 │   study-to-prime        │
 └────────┬────────────────┘
          │
          ▼
 ┌─────────────────────────┐
 │   gaia-study            │
 │   POST /study/gpu-release│ ◄── NEW: cleanup + empty_cache
 │   Frees training VRAM   │
 └────────┬────────────────┘
          │
          ▼
 ┌─────────────────────────┐
 │   gaia-prime            │
 │   POST /wake_up         │ ◄── vLLM wake endpoint
 │   CPU RAM → VRAM        │
 │   (~0.1-0.3s)           │
 └────────┬────────────────┘
          │
          ▼
 ┌─────────────────────────┐
 │   gaia-core             │
 │   POST /gpu/reclaim     │ ◄── NEW: restores gpu_prime as primary
 │   Resume normal routing │
 └────────┘
```

### Key Design Decisions

1. **gaia-core proxies sleep/wake to gaia-prime** — The orchestrator calls gaia-core's `/gpu/release` and `/gpu/reclaim`, and gaia-core forwards the sleep/wake to gaia-prime. This keeps the orchestrator decoupled from vLLM internals.

2. **Fallback routing during sleep** — While prime is asleep, gaia-core's ModelPool automatically uses its fallback chain: `groq_fallback` → `oracle_openai` → `oracle_gemini` → `lite`. GAIA remains functional, just using cloud/CPU inference.

3. **gaia-study manages its own training lifecycle** — The `/study/gpu-ready` endpoint triggers a queued training session. `/study/gpu-release` does cleanup + `torch.cuda.empty_cache()`.

4. **Adapter hot-loading after wake** — If study produced a new LoRA adapter, gaia-core tells gaia-prime to load it via vLLM's LoRA support (already enabled: `--enable-lora --max-loras 4`).

## Changes Required

### 1. gaia-prime: Enable sleep mode

**Dockerfile/entrypoint change** — Add `--enable-sleep-mode` flag and `VLLM_SERVER_DEV_MODE=1` env var.

vLLM already exposes `/sleep?level=N` and `/wake_up` when these are set. No code changes to gaia-prime itself.

### 2. gaia-core: New GPU management endpoints

**New file: `gaia_core/api/gpu_endpoints.py`**

| Endpoint | Purpose |
|----------|---------|
| `POST /gpu/release` | Put gaia-prime to sleep (Level 1), mark gpu_prime as unavailable in ModelPool |
| `POST /gpu/reclaim` | Wake gaia-prime, verify health, restore gpu_prime availability |
| `GET /gpu/status` | Report current GPU state (active/sleeping/unavailable) |

### 3. gaia-study: New GPU handoff endpoints

**Modifications to `gaia_study/server.py`**

| Endpoint | Purpose |
|----------|---------|
| `POST /study/gpu-ready` | Acknowledge GPU availability, start queued training if any |
| `POST /study/gpu-release` | Cleanup training resources, free VRAM, ack release |

### 4. gaia-orchestrator: No changes needed

The existing `HandoffManager` and `GPUManager` already implement the correct protocol. They call exactly the endpoints we're now implementing.

## VRAM Budget (During Training)

```
State: PRIME SLEEPING
├── CUDA context baseline:  ~250 MB  (unavoidable)
├── gaia-prime sleeping:      0 MB   (weights in CPU RAM)
├── QLoRA training (4-bit):  ~6 GB   (quantized base + LoRA + optimizer)
├── Available:               ~9.75 GB
└── Total:                   16 GB

State: PRIME ACTIVE (normal)
├── gaia-prime (vLLM 0.65): ~10.4 GB
├── KV cache:                ~5.3 GB
├── Available:               ~0.3 GB
└── Total:                   16 GB
```

CPU RAM during sleep: +10-14GB temporary (model weights). With 64GB system RAM, this is comfortable.

## Timeline Estimate

| Component | Complexity |
|-----------|-----------|
| gaia-prime sleep mode flag | Trivial (env var + CLI flag) |
| gaia-core GPU endpoints | Medium (~100 lines) |
| gaia-study GPU endpoints | Medium (~80 lines) |
| Integration testing | Requires running stack |

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| vLLM sleep mode is "dev mode" only | Monitor vLLM releases for GA promotion; sleep mode has been stable since v0.8 |
| CPU RAM spike during sleep | Pre-check available RAM before sleeping; abort if < 20GB free |
| Training crash leaves GPU in limbo | Orchestrator handoff timeout (120s) auto-reverts ownership; gaia-study cleanup on crash |
| CUDA context survives but is corrupted | Health check after wake_up; cold restart as last resort |
| Inference degraded during training | Fallback chain (Groq/cloud) provides acceptable latency for short training windows |

## Verification

1. **Sleep/wake cycle**: PUT prime to sleep, verify VRAM drops below 500MB, wake, verify inference works
2. **Handoff round-trip**: Orchestrator prime→study→prime handoff, verify all state transitions
3. **Training E2E**: Sleep prime → train QLoRA adapter → wake prime → load adapter → verify inference uses new adapter
4. **Fallback routing**: While prime sleeps, verify gaia-core routes to groq_fallback
5. **Error recovery**: Kill study during training, verify orchestrator times out and restores prime

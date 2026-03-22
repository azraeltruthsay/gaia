# 2026-03-22 — Subprocess Isolation, Tier Router, GPTQ Prime

## Summary

Major GPU lifecycle overhaul: Engine Manager for zero-GPU standby, orchestrator Tier Router for automatic handoff, and GPTQ quantization for Prime (8B model: 16GB → 5.8GB VRAM).

## What Was Built

### 1. Engine Manager — Zero-GPU Subprocess Isolation

**Problem**: Every GAIA Engine process held a CUDA context (~500MB-1GB) even in standby. Three engines = 1.5-3GB dead VRAM. Model unload only freed weights, not the CUDA context — only process kill truly freed it.

**Solution**: `gaia_common/engine/manager.py` — a stdlib-only HTTP server (no torch, no CUDA) that:
- Starts on the public port with zero GPU footprint
- On `/model/load`: spawns a worker subprocess running the real GAIAEngine on an internal port
- Proxies all requests to the worker transparently
- On `/model/unload`: kills the worker process → CUDA context dies → zero VRAM, guaranteed

**Files**:
- `gaia-common/gaia_common/engine/manager.py` — EngineManager + ManagedEngineHandler (NEW)
- `gaia-common/gaia_common/engine/__main__.py` — CLI: `python -m gaia_common.engine --managed` (NEW)
- `gaia-core/entrypoint.sh` — Managed mode default for safetensors, `GAIA_ENGINE_DIRECT=1` for legacy
- `gaia-prime/entrypoint.sh` — Replaced 200-line inline Python standby handler with managed engine
- `gaia-llama/entrypoint.sh` — Added managed mode for safetensors path
- `gaia-core/gaia_core/gaia_engine.py` — Added `--managed` flag

**Verified**: Core and Prime both report `"managed": true` in health checks. Unload kills worker PID, GPU memory returns to desktop baseline (~786MB).

### 2. Orchestrator Tier Router — Automatic GPU Handoff

**Problem**: Handoff between tiers required manual model management — caller had to know which tier was loaded, unload it, load the target, then send the request.

**Solution**: `gaia_orchestrator/tier_router.py` — TierRouter class that:
- Tracks which tier currently has GPU
- On `/tier/infer`: ensures target tier is loaded (unloading others if needed), then proxies inference
- On `/tier/ensure`: just the handoff, no inference
- Health-checks tiers to detect already-loaded models

**Endpoints added to orchestrator**:
- `POST /tier/infer` — "just ask" interface: specify tier + messages, handoff is automatic
- `POST /tier/ensure` — ensure a tier's model is loaded
- `GET /tier/status` — which tiers have models loaded
- `POST /tier/unload-all` — zero GPU across all tiers
- `POST /tier/sae-record` — trigger SAE recording on a loaded tier

### 3. Cognitive Handoff Test

**Script**: `scripts/cognitive_handoff_test.py` — exercises all three tiers in sequence through the orchestrator.

**Results** (first run):
- Nano: 1/3 (identity tests inappropriate for 0.8B triage model — see Nano section below)
- Core: 3/3 — identity, architecture knowledge, epistemic honesty all solid
- Prime: 3/3 — identity, reasoning, code generation all solid
- Handoff times: Nano 2.9s → Core 6.8s → Prime 95.8s (8B cold start)

**Issue found**: Test left Core unloaded after finishing. GAIA went unresponsive in Discord. Restored manually. Need cleanup step in test script and orchestrator default-state restore.

### 4. Nano Identity — Intentional Design Decision

Investigation confirmed: Nano (0.8B) was intentionally left without identity training.
- March 18 baseline: 3/10 identity score, flagged for training
- March 19: Four training experiments hit hard capacity ceiling — 0.8B can't hold identity + structured data extraction simultaneously
- March 21: Accepted as architectural choice — "triage doesn't need identity"
- Cascade routing ensures identity questions escalate to Core (100% identity score)

### 5. GPTQ Prime — 8B Model at 5.8GB VRAM

**Problem**: Prime 8B at bf16 = 16GB (won't fit). NF4 = ~4.5GB but quality loss. Previous AWQ via vLLM no longer applicable since we moved to GAIA Engine.

**Solution**: GPTQ quantization via `gptqmodel` in gaia-study, loaded by GAIA Engine via transformers integration.

**Quantization**:
- Input: `/warm_pool/Huihui-Qwen3-8B-GAIA-Prime-adaptive` (16GB bf16)
- Output: `/warm_pool/Huihui-Qwen3-8B-GAIA-Prime-adaptive-GPTQ` (5.7GB GPTQ 4-bit)
- Method: gptqmodel 5.7.0, W4 group_size=128, wikitext calibration
- Time: 215 seconds (~3.5 min) on RTX 5080
- First attempt to save to read-only `/warm_pool` failed — saved to `/models` then copied

**Loading**:
- AutoAWQ: deprecated, doesn't support Qwen3 architecture (Catcher hook fails)
- auto-gptq: incompatible with transformers 5.3.0 (`no_init_weights` removed)
- **gptqmodel + optimum**: the working stack. gptqmodel registers GPTQ backend, optimum bridges transformers' quantizer interface
- Required packages: `gptqmodel`, `optimum`, `threadpoolctl`

**Result**: 5829 MB VRAM, 2.2s load time, identity intact: "I am GAIA, a sovereign AI created by Azrael."

**Engine changes**:
- `core.py`: Detects `quantization_config.quant_method` in config.json, imports gptqmodel to register backend, skips NF4 fallback for pre-quantized models
- `Dockerfile.engine-base`: Added `gptqmodel` (--no-build-isolation), `optimum`, build deps for pypcre

### 6. Self-Awareness Pipeline — QUANTIZE_PRIME Stage

Added `QUANTIZE_PRIME` stage between `MERGE_PRIME` and `DEPLOY_PRIME`:
- Calls existing `quantize_prime()` from `merge_and_requantize.py`
- Updates `ctx.merged_prime_path` so DEPLOY uses quantized model
- Graceful fallback: if quantization fails, bf16 passes through (engine auto-NF4)
- GGUF insertion point updated: after QUANTIZE_PRIME instead of MERGE_PRIME

Pipeline: TRAIN → MERGE → **QUANTIZE** → DEPLOY

## Revised VRAM Budget (16GB RTX 5080)

| State | VRAM | Headroom |
|-------|------|----------|
| AWAKE (Core 2B + Nano 0.8B) | ~5.2GB | 10.8GB |
| LISTENING (+ STT 0.6B) | ~7.0GB | 9.0GB |
| FOCUSING (Prime GPTQ) | ~5.8GB | 10.2GB |
| FOCUS + LISTEN (Prime + STT) | ~7.6GB | 8.4GB |
| FOCUS + LISTEN + SPEAK (Prime + STT + TTS 1.7B) | ~11.9GB | 4.1GB |
| MEDITATION (QLoRA NF4 + gradients) | ~8-10GB | 6GB |

**Key finding**: GPTQ Prime (5.8GB) + full audio stack (6.1GB) = 11.9GB. **Fits on 16GB**. No GPU time-swap needed for thinking + listening + speaking simultaneously.

## Audio VRAM — Corrected Numbers

Current gaia-audio uses Qwen3 models (not Whisper):
- STT: Qwen3-ASR-0.6B → 1.8GB GPU (always-on listener)
- TTS Nano: Qwen3-TTS-0.6B → CPU only (instant short phrases)
- TTS Prime: Qwen3-TTS-1.7B → 4.3GB GPU (on-demand quality)
- Full audio concurrency: **6.1GB**, not the 1GB previously assumed

## GPU Lifecycle State Machine (Designed, Not Yet Implemented)

```
DEEP SLEEP → SLEEP → AWAKE ↔ LISTENING
                       ↕         ↕
                    FOCUSING ← ──┘
                       ↕
                   MEDITATION
```

- **AWAKE**: Core + Nano on GPU (~5.2GB)
- **LISTENING**: + Audio STT (~7.0GB)
- **FOCUSING**: Prime GPTQ on GPU (~5.8GB), Core/Nano unloaded
- **FOCUS+LISTEN+SPEAK**: Prime + full audio (~11.9GB) — fits!
- **MEDITATION**: All cognitive tiers unloaded, Study takes GPU for QLoRA/merge/GPTQ
- **SLEEP**: Core + Nano in CPU RAM, GPU empty
- **DEEP SLEEP**: Core unloaded from RAM entirely, Nano minimal reflex in RAM

## Dependency Stack for GPTQ Loading

Working combination (engine base image):
- `torch 2.10.0+cu128`
- `transformers 5.3.0`
- `gptqmodel 5.8.0` (installed with `--no-build-isolation`, needs build-essential + libpcre2-dev at build time)
- `optimum 2.1.0` (bridges transformers' GPTQ quantizer interface)
- `bitsandbytes` (NF4 for training)
- `optimum-quanto` (int8 fallback)

## Files Changed

| File | Action | Purpose |
|------|--------|---------|
| `gaia-common/gaia_common/engine/manager.py` | NEW | Zero-GPU EngineManager |
| `gaia-common/gaia_common/engine/__main__.py` | NEW | `--managed` CLI entrypoint |
| `gaia-common/gaia_common/engine/core.py` | MOD | GPTQ/AWQ detection, gptqmodel import |
| `gaia-orchestrator/gaia_orchestrator/tier_router.py` | NEW | Automatic GPU handoff |
| `gaia-orchestrator/gaia_orchestrator/main.py` | MOD | Tier router endpoints |
| `gaia-core/entrypoint.sh` | MOD | Managed mode default |
| `gaia-prime/entrypoint.sh` | MOD | Managed mode, replaced inline standby |
| `gaia-llama/entrypoint.sh` | MOD | Managed mode for safetensors |
| `gaia-core/gaia_core/gaia_engine.py` | MOD | `--managed` flag |
| `gaia-study/scripts/self_awareness_pipeline.py` | MOD | QUANTIZE_PRIME stage |
| `docker/Dockerfile.engine-base` | MOD | gptqmodel + optimum + build deps |
| `scripts/cognitive_handoff_test.py` | NEW | Handoff test across tiers |
| `scripts/awq_quantize_prime.py` | NEW | AWQ script (failed — AutoAWQ deprecated) |

All candidates/ synced to production.

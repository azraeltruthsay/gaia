# Dev Journal Entry: 2026-02-12 (Part 2) — Candidate-to-Live Full Promotion

**Date:** 2026-02-12
**Author:** Claude Code (Opus 4.6) via Happy

## Context

Following the completion of all bug fixes, features, and hardening work documented in the main Feb 12 journal, the entire candidate stack was promoted to live. This is the first full-stack promotion using the updated `promote_candidate.sh` workflow (with sync validation), and the first time the live `docker-compose.yml` was updated to include the gaia-prime performance features tested in candidates.

---

## Step 1: Stop All Containers

Both candidate and live stacks were stopped cleanly:
- 7 candidate containers (`gaia-core-candidate`, `gaia-web-candidate`, `gaia-mcp-candidate`, `gaia-study-candidate`, `gaia-orchestrator-candidate`, `gaia-prime-candidate`) via `docker compose -f docker-compose.candidate.yml --profile full down`
- 1 live container still running (`gaia-mcp`) plus dormant live containers stopped via `docker compose down`
- Network `gaia-network` cleaned up after both stacks stopped

## Step 2: Promote All Services

All 7 services promoted via `promote_candidate.sh` with `--no-restart --no-backup`:

| Service | Result | Notes |
|---------|--------|-------|
| gaia-common | Clean | 3 files synced (constants, cognition_packet, tools_registry) |
| gaia-core | Clean (with warnings) | Source code promoted; rsync warned on container-owned files in `app/shared/`, `data/shared/`, `logs/` (Permission denied on Docker-volume-owned dirs — harmless, runtime-recreated) |
| gaia-web | Clean | Already in sync (no file changes) |
| gaia-mcp | Clean | 5 files synced (requirements.txt, server.py, tools.py, web_tools.py, test_web_tools.py) |
| gaia-study | Clean | Already in sync |
| gaia-orchestrator | Clean | Already in sync |
| gaia-prime | Clean | 4 files synced (Dockerfile, .dockerignore, cmake_wrapper.sh, patch_float8.sh) |

**Key validation:** All promotions passed the new gaia-common sync check (cognition_packet.py, gaia_constants.json, protocol files all in sync). This confirms the sync validation hardening from earlier today is working as designed.

**Promotion order:** gaia-common first (dependency for all others), then remaining 6 services in parallel.

## Step 3: Update Live `docker-compose.yml` for gaia-prime

The live gaia-prime service was updated to match the candidate configuration that had been tested over the past two days:

### Changes Applied

| Setting | Before (live) | After (live) | Rationale |
|---------|--------------|-------------|-----------|
| `volumes` | `./gaia-models:/models:ro` | `/mnt/gaia_warm_pool/Claude:/models/Claude:ro` + `/mnt/gaia_warm_pool/lora_adapters:/models/lora_adapters:ro` | Warm pool on tmpfs — model weights pre-loaded in RAM, eliminating cold-start disk I/O |
| `gpu-memory-utilization` | 0.65 | 0.70 | Candidate testing showed stable at 0.70 with 5070 Ti 16GB |
| `VLLM_SERVER_DEV_MODE` | (absent) | `1` | Required for sleep/wake endpoint support |
| `--enable-sleep-mode` | (absent) | Added | Enables `/sleep` and `/wake_up` endpoints for VRAM↔RAM hot-swap (orchestrator integration) |
| `--enable-prefix-caching` | (absent) | Added | Caches repeated prompt prefixes, reducing recomputation for multi-turn conversations |
| `--kv-offloading-backend native` | (absent) | Added | Native KV cache offloading to system RAM |
| `--kv-offloading-size 8` | (absent) | Added | 8GB of KV cache can spill to system RAM when GPU VRAM fills |
| `--disable-hybrid-kv-cache-manager` | (absent) | Added | Pure offloading mode (no hybrid cache splitting) |

### What These Enable

1. **Sleep Mode**: The orchestrator can now call `POST /sleep` on gaia-prime to release GPU VRAM when idle, and `POST /wake_up` to resume. This allows GPU sharing with gaia-study for embedding/training tasks.
2. **KV Offloading**: For longer conversations or concurrent requests, KV cache entries that don't fit in GPU VRAM transparently spill to system RAM (8GB budget). This prevents OOM-triggered request failures.
3. **Prefix Caching**: Multi-turn conversations reuse cached KV entries from the system prompt and earlier turns, reducing time-to-first-token for follow-up messages.
4. **Warm Pool**: Model weights served from tmpfs (`/mnt/gaia_warm_pool/`) instead of SSD, eliminating disk I/O during model loading and sleep/wake cycles.

## Step 4: Rebuild All Live Images

All 6 service images rebuilt via `docker compose build --parallel`:

| Image | Build Time | Notes |
|-------|-----------|-------|
| gaia-orchestrator | ~30s | Lightweight Python service |
| gaia-mcp | ~2min | New deps: `duckduckgo_search`, `trafilatura`, `beautifulsoup4` |
| gaia-web | ~1min | No significant dep changes |
| gaia-study | ~2min | No significant dep changes |
| gaia-core | ~4min | Full CUDA torch + sentence-transformers + llama-cpp-python wheel build |
| gaia-prime | **~62min** | vLLM compiled from source (v0.15.1, CUDA 12.8 kernels, MAX_JOBS=4). numpy<2 fix applied post-install for nvidia-modelopt compatibility |

Total wall time dominated by gaia-prime's vLLM CUDA kernel compilation. Other services built in parallel and finished in under 4 minutes.

## Step 5: Start Full Live Stack

`docker compose up -d` brought up all services with dependency ordering:
1. `gaia-mcp`, `gaia-prime`, `gaia-study`, `gaia-orchestrator` (no deps) — started first
2. `gaia-core` (depends on `gaia-mcp` healthy + `gaia-prime` healthy) — started after healthchecks passed
3. `gaia-web` (depends on `gaia-core` healthy) — started last

## Step 6: Verification

All 6 services verified healthy:

| Service | Port | Health | Model/Endpoint |
|---------|------|--------|---------------|
| gaia-core | :6415 | `{"status": "healthy"}` | — |
| gaia-prime | :7777 | Docker healthcheck passing | `/v1/models` returns `/models/Claude`, max_model_len=8192 |
| gaia-mcp | :8765 | `{"status": "healthy"}` | — |
| gaia-study | :8766 | `{"status": "healthy"}` | — |
| gaia-web | :6414 | `{"status": "healthy"}` | — |
| gaia-orchestrator | :6410 | `{"status": "healthy"}` | — |

---

## What's Now Live (Feature Summary)

This promotion brings all work from Feb 11–12 candidate testing into production:

1. **Epistemic guardrails** — Confabulation detection and correction in output pipeline
2. **Cognitive audit** — Self-reflection loop after generation with configurable trigger conditions
3. **History review** — Pre-injection audit of conversation history for poisoned/fabricated content
4. **Embedding-based intent classifier** — Cosine-similarity intent detection using MiniLM exemplar bank
5. **Web research tools** — `web_search` (DuckDuckGo) and `web_fetch` (domain-gated content extraction) in MCP
6. **CJK post-processing filter** — Removes spurious CJK characters from model output
7. **Constants consolidation** — Single source of truth in gaia-common
8. **CognitionPacket consolidation** — Single definition in gaia-common
9. **Promotion script hardening** — Sync validation for shared protocols and constants
10. **gaia-prime optimizations** — Sleep mode, KV offloading, prefix caching, warm pool mounts, higher GPU utilization

## Process Notes

This was the first full promotion following the candidate-first workflow described in the Feb 12 retrospective. All code was edited in `candidates/`, tested in candidate containers, and promoted via `promote_candidate.sh`. The sync validation caught no issues (because we'd been maintaining sync), confirming the tooling works.

**Observation for future sessions:** The gaia-prime image rebuild takes ~60 minutes due to vLLM source compilation. For iterative testing of vLLM command-line flags only (no Dockerfile/dependency changes), it's more efficient to just update `docker-compose.yml` and restart — vLLM flags are passed at runtime, not baked into the image. The rebuild was necessary here because the Dockerfile itself changed (numpy<2 fix).

---

## Status

Full live stack running. All 6 services healthy. Ready for Discord integration testing.

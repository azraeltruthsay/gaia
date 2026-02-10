# Dev Journal Entry: 2026-02-09 - Promote gaia-prime to Live & Complete v0.3 GPU Offload

**Date:** 2026-02-09
**Author:** Claude Code (Opus 4.6) via Happy

## Context and Motivation

The v0.3 architecture separates GPU inference (gaia-prime) from cognition (gaia-core). The candidate stack already implemented this: gaia-core-candidate runs CPU-only and delegates inference to gaia-prime-candidate via `PRIME_ENDPOINT`. However, the live compose still configured gaia-core with GPU access and had no gaia-prime service. This meant the live stack couldn't start successfully — gaia-core had no vLLM installed but still reserved a GPU.

This session promotes gaia-prime to the live stack, strips GPU from live gaia-core, fixes the orchestrator's missing gaia-common dependency, and registers gaia-prime in the promotion tooling.

## Goal

Complete the v0.3 GPU offload architecture in the live stack so that:
- gaia-prime owns the GPU for vLLM inference (port 7777)
- gaia-core runs CPU-only, delegating to gaia-prime via `PRIME_ENDPOINT`
- The orchestrator knows about gaia-prime
- The promotion script supports gaia-prime

## Key Changes Implemented

### 1. Created Live `gaia-prime/` Directory

- Copied candidate files to `/gaia/GAIA_Project/gaia-prime/`: Dockerfile, cmake_wrapper.sh, patch_float8.sh, .dockerignore
- The Dockerfile builds vLLM v0.15.1 from source on NGC PyTorch 25.03 base image with sm_120 (RTX 5080 Blackwell) support
- Tagged the existing candidate image (`localhost:5000/gaia-prime-candidate:local`) as `localhost:5000/gaia-prime:local` to avoid a ~30 min rebuild (identical Dockerfile)

### 2. Added gaia-prime Service to `docker-compose.yml`

- New service block between gaia-orchestrator and gaia-core
- GPU reservation: 1 NVIDIA GPU (dedicated to vLLM)
- Port: 7777:7777 (live port, candidate was 7778:7777)
- Health check with 120s start_period (model loading time)
- `restart: unless-stopped` (live policy)
- No profiles (always starts in live stack, unlike candidate which uses profiles)
- Updated header comment to include gaia-prime as "The Voice"

### 3. Modified gaia-core: CPU-Only with PRIME_ENDPOINT

- **Removed**: `deploy.resources.reservations.devices` (GPU reservation)
- **Removed**: GPU-specific env vars: `CUDA_VISIBLE_DEVICES`, `GAIA_VLLM_SAFE_MODE`, `GAIA_VLLM_WORKER_METHOD`, `VLLM_WORKER_MULTIPROC_METHOD`, `VLLM_DISABLE_CUSTOM_CUDA_MODULES`, `TORCH_COMPILE_DISABLE`
- **Added**: CPU-only env vars matching candidate config:
  - `GAIA_BACKEND=gpu_prime` — selects the remote inference backend
  - `GAIA_FORCE_CPU=1` — prevents local GPU usage
  - `N_GPU_LAYERS=0` — no GPU layers for llama.cpp fallback
  - `PRIME_ENDPOINT=http://gaia-prime:7777` — remote vLLM server
  - `PRIME_MODEL=/models/Claude` — model path on prime
  - `GROQ_API_KEY`, `GROQ_MODEL` — Groq API fallback (free tier)
- **Added**: `depends_on: gaia-prime: condition: service_healthy` — core waits for inference server
- Updated header comment: gaia-core is now "CPU, delegates to gaia-prime"

### 4. Fixed gaia-orchestrator Dockerfile and Build Context

- Changed build context in compose from `./gaia-orchestrator` to `.` (project root)
- Updated Dockerfile COPY paths: `COPY gaia-orchestrator/requirements.txt .`, `COPY gaia-orchestrator/gaia_orchestrator/ ...`, `COPY gaia-orchestrator/pyproject.toml .`
- **Added**: `COPY gaia-common /gaia-common` + `RUN pip install --no-cache-dir -e /gaia-common/`
- **Why**: The orchestrator's `main.py` imports `gaia_common.utils.install_health_check_filter` with a try/except fallback. With gaia-common installed, it now gets the real implementation instead of silently skipping.

### 5. Added ORCHESTRATOR_PRIME_URL to Orchestrator Environment

- New env var: `ORCHESTRATOR_PRIME_URL=http://gaia-prime:7777`
- Matches the pattern of existing `ORCHESTRATOR_CORE_URL`, `ORCHESTRATOR_WEB_URL`, etc.

### 6. Registered gaia-prime in `promote_candidate.sh`

- Added to `SERVICE_CONFIG`: `["gaia-prime"]="7777:7778:yes"`
- Added to help text and error messages
- **Not** added to `PYTHON_SERVICES` — gaia-prime is a vLLM build container with no `/app` Python project, so ruff/mypy/pytest don't apply. The `--validate` flag will skip it with "not configured for Python validation."

### 7. Fixed gaia-web PYTHONPATH (Discovered During Bringup)

- **Problem**: gaia-web failed to start with `ModuleNotFoundError: No module named 'gaia_common'`
- **Root cause**: The Dockerfile installs gaia-common via `pip install -e /app/gaia-common/`, but the compose volume mount `./gaia-web:/app:rw` overwrites `/app` entirely, wiping the pip-installed reference. The separate mount `./gaia-common:/gaia-common:ro` provides the files but not in the pip-expected location.
- **Fix**: Added `PYTHONPATH=/app:/gaia-common` to gaia-web's environment in compose
- **Note**: This is a pre-existing issue that would have affected any live stack bringup, not specific to v0.3 changes. The candidate compose solves it differently with `PYTHONPATH=/app:/app/gaia-common` and mounting gaia-common inside `/app/`.

## Final Stack State

All 6 services running and healthy:

| Service | Role | Port | GPU | Status |
|---------|------|------|-----|--------|
| gaia-orchestrator | The Coordinator | 6410 | - | healthy |
| gaia-prime | The Voice | 7777 | 1x NVIDIA | healthy |
| gaia-core | The Brain (CPU) | 6415 | - | healthy |
| gaia-web | The Face + Discord | 6414 | - | healthy |
| gaia-mcp | The Hands | 8765 | - | healthy |
| gaia-study | The Subconscious | 8766 | all GPUs | healthy |

Dependency chain: `gaia-prime + gaia-mcp → gaia-core → gaia-web`

## Files Modified

- `docker-compose.yml` — added gaia-prime service, modified gaia-core (CPU-only), modified gaia-orchestrator (build context), added gaia-web PYTHONPATH, added orchestrator PRIME_URL
- `gaia-orchestrator/Dockerfile` — added gaia-common install, updated COPY paths for project-root context
- `scripts/promote_candidate.sh` — registered gaia-prime in SERVICE_CONFIG, help text, error messages

## Files Created

- `gaia-prime/` — live directory (Dockerfile, cmake_wrapper.sh, patch_float8.sh, .dockerignore) — promoted from candidates/gaia-prime/

## Impact and Potential Breakpoints

- **GPU allocation changed**: gaia-core no longer has GPU access. Any code in gaia-core that attempts direct GPU usage (llama.cpp with GPU layers, local vLLM) will fail. This is intentional — all inference goes through gaia-prime via `PRIME_ENDPOINT`.
- **gaia-prime is a hard dependency**: gaia-core now has `depends_on: gaia-prime: condition: service_healthy`. If gaia-prime can't load the model (missing model files, OOM, GPU unavailable), gaia-core won't start. The 120s start_period gives prime time to load, but large models may need more.
- **gaia-study still has GPU**: gaia-study retains `count: all` GPU reservation for embedding model work. This means both gaia-prime and gaia-study claim GPUs. On a single-GPU system, this could cause contention. The candidate compose had the same setup.
- **Image reuse**: The live gaia-prime image is a tag of the candidate image (same SHA). A `docker compose build gaia-prime` would rebuild from the live Dockerfile (identical content), producing the same result but taking ~30 minutes.
- **gaia-web PYTHONPATH fix**: The `PYTHONPATH=/app:/gaia-common` fix works for development mode (volume mounts). In production without volume mounts, the Dockerfile's `pip install -e` would handle it. If someone changes the gaia-common mount path in compose, the PYTHONPATH must be updated to match.

## What's Left / Next Steps

- **Commit**: All changes are uncommitted. The working tree has extensive changes spanning multiple sessions (see git status).
- **Test inference end-to-end**: Send a message through the full pipeline (Discord/API → gaia-web → gaia-core → gaia-prime) and verify vLLM inference works.
- **GPU contention**: On single-GPU systems, may need to configure gaia-study to use CPU-only embeddings or share the GPU more carefully with gaia-prime.
- **Production hardening**: The gaia-prime health check relies on vLLM's `/health` endpoint. May want to add a model-loaded check (hit `/v1/models` and verify the model is listed) for a stronger health signal.
- **Orchestrator integration**: The orchestrator now has `ORCHESTRATOR_PRIME_URL` but its code may not yet use it for health monitoring or GPU handoff coordination. This would need code changes in the orchestrator's service registry.

## Debugging and Rollback

- **gaia-prime won't start**: Check `docker logs gaia-prime` — common issues: model not found at `/models/Claude`, CUDA OOM, GPU driver mismatch. Verify model files exist: `ls gaia-models/Claude/`
- **gaia-core can't reach prime**: Verify both are on gaia-network: `docker network inspect gaia-network`. Test connectivity: `docker exec gaia-core curl http://gaia-prime:7777/health`
- **gaia-web can't find gaia_common**: Verify PYTHONPATH includes `/gaia-common` and the volume mount exists: `docker exec gaia-web python -c "import gaia_common; print(gaia_common.__file__)"`
- **Rollback GPU to gaia-core**: Revert the gaia-core section in `docker-compose.yml` to add back the `deploy.resources.reservations.devices` block and GPU env vars. Remove `depends_on: gaia-prime`.
- **Full rollback**: `git checkout -- docker-compose.yml gaia-orchestrator/Dockerfile scripts/promote_candidate.sh && rm -rf gaia-prime/`

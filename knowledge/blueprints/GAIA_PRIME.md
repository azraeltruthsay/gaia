# GAIA Service Blueprint: `gaia-prime` (The Voice)

## Role and Overview

`gaia-prime` is the dedicated GPU inference server within the GAIA ecosystem. It runs vLLM as a standalone OpenAI-compatible API server, providing LLM inference to `gaia-core` via HTTP. This decouples GPU-intensive inference from CPU-based cognition, enabling independent scaling and eliminating CUDA/vLLM dependency conflicts in the cognition container.

**Current model**: Qwen3-8B-abliterated-AWQ (GPU) / Qwen3-8B-abliterated-Q4_K_M.gguf (CPU lite). The model path is configurable via `PRIME_MODEL_PATH` environment variable.

## Build and Image

**Base Image**: `nvcr.io/nvidia/pytorch:25.03-py3` (NGC with CUDA 12.8, pre-compiled PyTorch)

**Target GPU**: RTX 5080 Blackwell — compute capability 12.0 (sm_120)

**vLLM Version**: v0.15.1 (built from source)

The build process is non-trivial due to Blackwell sm_120 compatibility:

1. **cmake_wrapper.sh** — Intercepts CMake calls to patch fetched dependencies (qutlass) for sm_120 support
2. **patch_float8.sh** — Patches vLLM source files replacing `Float8_e8m0fnu` → `Float8_e4m3fn` (incompatibility with NGC PyTorch 2.7.0a0)
3. **MAX_JOBS=4** — Controlled parallel build to avoid OOM during compilation

Build takes approximately 30 minutes.

## Files

```
gaia-prime/
├── Dockerfile          # Multi-stage build: vLLM from source on NGC base
├── cmake_wrapper.sh    # CMake interception for sm_120 patches
├── patch_float8.sh     # Float8 dtype compatibility patches
└── .dockerignore       # Build exclusions
```

## Docker Compose Configuration

```yaml
gaia-prime:
  image: localhost:5000/gaia-prime:local
  container_name: gaia-prime
  hostname: gaia-prime
  port: 7777:7777
  restart: unless-stopped
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  healthcheck:
    test: curl -f http://localhost:7777/health
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 120s    # Model loading time
```

## vLLM Launch Configuration

The container runs vLLM with these parameters:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--model` | `${PRIME_MODEL_PATH:-/models/Qwen3-8B-abliterated-AWQ}` | Loads model from mounted volume |
| `--gpu-memory-utilization 0.70` | 70% VRAM | Allocation for model + KV cache |
| `--max-model-len 8192` | 8K context | Context window size |
| `--max-num-seqs 4` | 4 concurrent | Concurrent sequence limit |
| `--dtype auto` | Automatic | Dtype selection based on hardware |
| `--enforce-eager` | No CUDA graphs | Eager execution mode (required for Blackwell) |
| `--enable-lora` | LoRA support | Dynamic adapter loading |
| `--max-loras 4` | 4 adapters | Maximum concurrent LoRA adapters |
| `--max-lora-rank 64` | Rank 64 | Maximum LoRA rank |

## Environment Variables

- `VLLM_WORKER_MULTIPROC_METHOD=spawn` — Avoids CUDA fork issues
- `VLLM_FLASH_ATTN_VERSION=2` — Flash Attention v2
- `TORCH_CUDA_ARCH_LIST=12.0+PTX` — Blackwell target architecture

## Volume Mounts

- `./gaia-models:/models:ro` — Model weights (read-only)
- `./gaia-models/lora_adapters:/models/lora_adapters:ro` — LoRA adapter files

## API Endpoints (vLLM OpenAI-compatible)

- `GET /health` — Health check
- `GET /v1/models` — List loaded models
- `POST /v1/completions` — Text completions
- `POST /v1/chat/completions` — Chat completions (primary)
- Streaming via Server-Sent Events (SSE)

## Interaction with Other Services

- **`gaia-core`** (caller): Sends inference requests via `VLLMRemoteModel` to `http://gaia-prime:7777`. Includes LoRA adapter selection in requests.
- **`gaia-orchestrator`** (monitor): Has `ORCHESTRATOR_PRIME_URL=http://gaia-prime:7777` for health monitoring.
- **`gaia-study`** (GPU contention): Both services claim GPU resources. On single-GPU systems, scheduling or CPU-only embedding mode may be needed.

## Promotion and Candidate

- **Live port**: 7777
- **Candidate port**: 7778
- Registered in `promote_candidate.sh` as `["gaia-prime"]="7777:7778:yes"`
- **Not** in `PYTHON_SERVICES` list — gaia-prime is a vLLM build container with no `/app` Python project, so ruff/mypy/pytest validation is skipped.

## Troubleshooting

- **Won't start**: Check `docker logs gaia-prime` — common issues: model not found at the configured path, CUDA OOM, GPU driver mismatch
- **Model loading slow**: 120s start_period may need increase for very large models
- **GPU contention**: If gaia-study is also running with GPU, may cause OOM. Check `nvidia-smi`
- **Health check failing**: Verify model is loaded: `curl http://localhost:7777/v1/models`
- **Build fails**: sm_120 patches may need updating for newer vLLM versions

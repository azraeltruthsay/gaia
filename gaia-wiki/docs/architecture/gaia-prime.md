# gaia-prime — The Voice

Standalone vLLM inference server. Owns the GPU. Provides OpenAI-compatible API for text generation.

## Responsibilities

- Serve inference requests from gaia-core via OpenAI-compatible HTTP API
- Manage GPU memory (70% utilization target)
- Support LoRA adapter hot-loading (up to 4 concurrent adapters)
- Sleep/wake mode for GPU memory conservation

## Runtime Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| `--gpu-memory-utilization 0.70` | 70% VRAM | Leave headroom for LoRA adapters and KV cache |
| `--max-model-len 8192` | 8K context | Balances quality with memory |
| `--max-num-seqs 4` | 4 concurrent | Single-user system, limit batch overhead |
| `--enable-lora` | Enabled | Hot-swap personality/task adapters |
| `--enable-sleep-mode` | Enabled | GPU conservation via sleep/wake |
| `--enable-prefix-caching` | Enabled | Cache system prompt prefixes |
| `--enforce-eager` | Enabled | Skip CUDA graph overhead for flexibility |

## Sleep Mode

When gaia-core's sleep cycle triggers, gaia-prime enters sleep mode:

1. gaia-core calls `POST /sleep` on gaia-prime
2. vLLM offloads KV cache to CPU memory
3. GPU VRAM is freed for gaia-study (training, embedding)
4. On wake: `POST /wake_up` restores KV cache from CPU

This is controlled by `VLLM_SERVER_DEV_MODE=1` which enables the `/sleep` and `/wake_up` endpoints.

## Model Loading

The active model is specified by `PRIME_MODEL_PATH` (default: `/models/Qwen3-8B-abliterated-AWQ`). Models are stored on `/mnt/gaia_warm_pool` (host) mounted as `/models` (container, read-only).

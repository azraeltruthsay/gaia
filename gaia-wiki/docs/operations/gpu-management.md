# GPU Management

GAIA has a single NVIDIA GPU shared between inference (gaia-prime) and training (gaia-study) via the orchestrator's lease-based handoff protocol.

## Ownership Model

Only one service holds the GPU at a time:

| State | GPU Owner | Triggered By |
|-------|-----------|-------------|
| Active | gaia-prime | Wake signal or user message |
| Sleep | gaia-study | Idle timeout → sleep cycle |
| Transition | none | Handoff in progress |

## Handoff Protocol

```
gaia-core sleep trigger
    │
    ├── 1. orchestrator: initiate handoff (prime → study)
    ├── 2. gaia-prime: POST /sleep (offload KV cache to CPU)
    ├── 3. orchestrator: wait for CUDA cleanup
    ├── 4. orchestrator: transfer lease to gaia-study
    ├── 5. gaia-study: begin GPU tasks
    │
    ... (training/embedding tasks) ...
    │
    ├── 6. wake signal received
    ├── 7. orchestrator: initiate handoff (study → prime)
    ├── 8. gaia-study: release GPU
    ├── 9. gaia-prime: POST /wake_up (restore KV cache)
    └── 10. gaia-core: resume inference
```

## Commands

```bash
# Check GPU status
./gaia.sh gpu status

# Manually release GPU (put prime to sleep)
./gaia.sh gpu release

# Manually reclaim GPU (wake prime)
./gaia.sh gpu reclaim
```

## VRAM Budget

| Component | Allocation |
|-----------|-----------|
| Base model | ~2-4 GB (quantized) |
| KV cache | ~1-2 GB (dynamic) |
| LoRA adapters | ~50-200 MB each |
| vLLM overhead | ~500 MB |
| **Total target** | **70% of VRAM** (`--gpu-memory-utilization 0.70`) |

The remaining 30% provides headroom for CUDA operations and prevents OOM during peak usage.

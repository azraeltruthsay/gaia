# GPU Management

GAIA has a single NVIDIA RTX 5080 (16GB VRAM) shared between tiers via the Consciousness Matrix and orchestrator lifecycle FSM.

## Consciousness Matrix

The Consciousness Matrix tracks each tier's GPU state:

| State | GPU | Inference | Default Tiers |
|-------|-----|-----------|---------------|
| **Conscious** | Yes | Full speed | Nano (0.8B), Core (2B) |
| **Subconscious** | No | CPU/GGUF | Prime (8B) |
| **Unconscious** | No | None | — |

Managed by orchestrator at `/consciousness/*`.

## Lifecycle FSM States

| State | Description |
|-------|-------------|
| AWAKE | Normal operation (Nano + Core on GPU, Prime on CPU) |
| FOCUSING | GPU swapped to Prime for heavyweight task |
| SLEEP | Idle timeout, models partially unloaded |
| DEEP_SLEEP | All models unloaded, zero VRAM |
| MEDITATION | Chaos drills active, defensive posture |

## FOCUSING Auto-Transition

When Core escalates to Prime:
1. Orchestrator transitions AWAKE -> FOCUSING
2. Quality gate evaluates if GPU swap is needed
3. Core releases GPU, Prime loads on GPU
4. Prime handles the task
5. GPU returns to Core, Prime drops to CPU/GGUF
6. Back to AWAKE

## VRAM Budget (RTX 5080, 16GB)

| Component | Allocation |
|-----------|-----------|
| Nano (0.8B, GPU) | ~1.5 GB |
| Core (2B, GPU) | ~4.7 GB |
| KV cache | ~1-2 GB (dynamic) |
| LoRA adapters | ~50-200 MB each |
| CUDA overhead | ~500 MB |
| **Typical AWAKE** | **~8 GB** |

Prime (8B) on GPU requires ~10-12 GB, which is why it defaults to CPU/GGUF and only gets GPU during FOCUSING.

## Commands

```bash
# Check GPU status
./gaia.sh gpu status

# Manually release GPU (put prime to sleep)
./gaia.sh gpu release

# Manually reclaim GPU (wake prime)
./gaia.sh gpu reclaim
```

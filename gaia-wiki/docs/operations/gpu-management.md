# GPU Management

GAIA has a single NVIDIA RTX 5080 (16GB VRAM) shared between tiers via the Consciousness Matrix and orchestrator lifecycle FSM.

## Consciousness Matrix

The Consciousness Matrix tracks each tier's GPU state:

| State | GPU | Inference | Default Tiers |
|-------|-----|-----------|---------------|
| **Conscious** | Yes | Full speed | Core (Gemma4-E4B) |
| **Subconscious** | No | CPU/GGUF | Prime (Qwen3-VL-8B) |
| **Unconscious** | No | None | — |

Managed by orchestrator at `/consciousness/*`.

## Lifecycle FSM States (the Gearbox)

Defined in `gaia-common/gaia_common/lifecycle/states.py`:

| Gear | State | Description |
|------|-------|-------------|
| P | PARKED | Core on CPU (GGUF), GPU empty — pre-warmed sentinel standby |
| 1 | AWAKE | Normal operation (Core on GPU NF4, Prime on CPU) |
| 1+ | LISTENING | AWAKE + audio STT active |
| 2 | FOCUSING | GPU swapped to Prime for heavyweight task |
| S | SLEEP | Idle timeout, models partially unloaded |
| 0 | DEEP_SLEEP | All models unloaded (Groq fallback only) |
| T | MEDITATION | Study owns the GPU for training; all cognitive tiers off |

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
| Core (Gemma4-E4B, GPU NF4) | ~8.8 GB |
| KV cache | ~1-2 GB (dynamic) |
| LoRA adapters | ~50-200 MB each |
| CUDA overhead | ~500 MB |
| **Typical AWAKE** | **~9-11 GB** |

Prime (Qwen3-VL-8B) on GPU takes ~4.6 GB (expert-buffered) and only gets the GPU during FOCUSING, while Core drops to CPU/GGUF. Note that even in "unloaded" states real resident VRAM is ~2.5 GB (CUDA-built llama-server buffers, engine-manager overhead, and gaia-audio's always-resident STT model).

## Commands

```bash
# Check GPU status
./gaia.sh gpu status

# Manually release GPU (put prime to sleep)
./gaia.sh gpu release
```

# LoRA Adapters

GAIA uses Low-Rank Adaptation (LoRA) for personality and task fine-tuning without replacing the base model.

## How It Works

vLLM's `--enable-lora` flag allows up to 4 LoRA adapters to be loaded simultaneously. The base model stays in GPU memory; only the small adapter weights (typically 10-50 MB) are swapped.

## Adapter Lifecycle

1. **Training** — gaia-study fine-tunes on conversation data during sleep cycles
2. **Storage** — adapters saved to `/models/` (gaia-study has write access)
3. **Loading** — gaia-prime loads adapters on demand via vLLM's adapter API
4. **Selection** — gaia-core's ModelPool routes requests to the appropriate adapter

## Configuration

```yaml
# gaia-prime command
--enable-lora
--max-loras 4          # Max concurrent adapters
--max-lora-rank 64     # Max adapter rank
```

## Current Adapters

Adapters are stored in `gaia-models/` and registered in the model pool configuration. The warm pool at `/mnt/gaia_warm_pool` keeps model files in RAM for fast loading.

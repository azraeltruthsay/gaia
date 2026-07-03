# LoRA Adapters

GAIA uses Low-Rank Adaptation (LoRA) for personality and task fine-tuning without replacing the base model.

## How It Works

The **GAIA Engine** (both tiers) manages LoRA adapters dynamically. The base model stays in GPU memory; only the small adapter weights (typically 10-50 MB) are swapped.

## Adapter Lifecycle

1. **Training** — gaia-study fine-tunes on conversation data during sleep cycles
2. **Storage** — adapters saved to `/models/` (gaia-study has write access)
3. **Loading** — loaded on demand via the GAIA Engine adapter API
4. **Selection** — gaia-core's ModelPool routes requests to the appropriate adapter

## Configuration

```bash
# GAIA Engine adapter API (any tier)
POST /adapter/load   # Load an adapter dynamically
POST /adapter/set    # Set the active adapter
```

## Current Adapters

Adapters are stored in `gaia-models/` and registered in the model pool configuration. The warm pool at `/mnt/gaia_warm_pool` keeps model files in RAM for fast loading.

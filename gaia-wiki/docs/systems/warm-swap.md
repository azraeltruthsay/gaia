# Warm Swap (Model Hot-Loading)

Warm swap refers to the ability to change the active model or LoRA adapter without restarting containers.

## Mechanism

Both tiers run the **GAIA Engine**, which supports runtime LoRA adapter loading via `POST /adapter/load` (activate with `POST /adapter/set`). The base model stays resident in GPU memory; adapters are loaded/unloaded on demand.

For base model changes, the orchestrator coordinates:

1. Stop inference requests (drain queue)
2. Unload current model from GPU
3. Load the new model (persistent models from `/models` — `../gaia-instance/gaia-models` on the host; warm-pool staging at `/warm_pool`, tmpfs-backed `/mnt/gaia_warm_pool`)
4. Resume inference

## Warm Pool

Models can be pre-staged on a tmpfs mount (`/mnt/gaia_warm_pool`, mounted into containers at `/warm_pool`) for near-instant loading. The active models themselves live in `gaia-instance/gaia-models/`, e.g.:

```
gaia-models/
├── prime -> Qwen3-VL-8B-GAIA-Prime-v1-abliterated   # Prime (symlink)
├── core.gguf                                        # Core CPU (GGUF)
└── Gemma4-E4B-GAIA-Core-* / adapters                # Core variants + LoRA adapters
```

See [tmpfs Warm Swap](../decisions/tmpfs-warm-swap.md) for the design rationale.

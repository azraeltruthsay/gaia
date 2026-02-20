# Warm Swap (Model Hot-Loading)

Warm swap refers to the ability to change the active model or LoRA adapter without restarting containers.

## Mechanism

gaia-prime runs vLLM with `--enable-lora`, which supports runtime adapter loading. The base model stays resident in GPU memory; adapters are loaded/unloaded on demand.

For base model changes, the orchestrator coordinates:

1. Stop inference requests (drain queue)
2. Unload current model from GPU
3. Load new model from warm pool (`/mnt/gaia_warm_pool` — tmpfs-backed)
4. Resume inference

## Warm Pool

Models are pre-staged on a tmpfs mount for near-instant loading:

```
/mnt/gaia_warm_pool/
├── Qwen3-4B-Instruct-2507-heretic/    # Primary model
├── Qwen3-8B-AWQ/                       # Candidate model
└── adapters/                            # LoRA adapters
```

See [tmpfs Warm Swap](../decisions/tmpfs-warm-swap.md) for the design rationale.

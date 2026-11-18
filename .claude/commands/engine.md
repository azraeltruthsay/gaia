Work on the GAIA Inference Engine: $ARGUMENTS

## Context

gaia-engine is a **standalone library** (not a containerized service). It is GAIA's shared inference backbone powering both cognitive tiers (Core, Prime).

- **Local path**: `/gaia/GAIA_Project/gaia-engine/`
- **Remote repo**: `github.com/azraeltruthsay/gaia-engine` (separate from the monorepo)
- **Contract**: `/gaia/GAIA_Project/contracts/services/gaia-engine.yaml`

## File Structure

```
gaia-engine/gaia_engine/
  core.py              # GAIAEngine class — model loading, generation, KV cache, polygraph
  manager.py           # EngineManager — zero-GPU subprocess manager, managed HTTP server
  thought_composer.py  # KV cache thought merging (compose_thoughts, estimate_composed_size)
  rome.py              # Rank-One Model Editing for surgical weight corrections
  sae_trainer.py       # Sparse Autoencoder atlas training on model activations
  adapter_surgeon.py   # SAE-guided LoRA adapter surgical correction
  awareness.py         # Dynamic situational awareness injection via filesystem
  cogpacket_compressor.py  # Prompt compression using KV cache segmentation
  weighted_trainer.py  # Eval-weighted QLoRA training (gap-focused)
  lifecycle/           # GPU lifecycle state machine types (LifecycleState, TransitionTrigger, etc.)
  __init__.py          # Public API exports
  __main__.py          # CLI entry point
```

## Contract Boundary

**gaia-engine provides:**
- `GAIAEngine` — direct inference (generate, generate_stream, hidden states, adapters)
- `EngineManager` / `serve_managed` — managed mode with subprocess isolation and HTTP API
- `serve` — direct mode HTTP server
- `compose_thoughts` / `estimate_composed_size` — KV cache thought merging
- `lifecycle` subpackage — state machine types shared by orchestrator and consumers
- Companion modules: rome, sae_trainer, adapter_surgeon, weighted_trainer, awareness, cogpacket_compressor

**gaia-engine must NOT depend on:**
- `gaia_common` — the dependency direction is reversed (gaia-common shims to gaia-engine)
- `gaia_core`, `gaia_web`, `gaia_mcp`, or any other GAIA service
- Any GAIA-specific config files (gaia_constants.json, etc.)
- The engine receives configuration through constructor arguments and function parameters only

## Consumers

- **gaia-prime** — managed mode on port 7777 (Qwen3-VL-8B Prime; CPU/GGUF default, GPU when FOCUSING)
- **gaia-core** — embedded managed mode on localhost:8092 (Gemma4-E4B Core; GPU NF4 or CPU GGUF)
- **gaia-nano** — DEPRECATED: now an alpine/socat proxy forwarding :8080 → gaia-core:8092 (runs no engine)
- **gaia-orchestrator** — imports lifecycle types for state machine authority
- **gaia-study** — imports weighted_trainer and sae_trainer for training pipelines

## Workflow

1. **Work in** `/gaia/GAIA_Project/gaia-engine/` — all edits happen here
2. **Test changes** — run tests from within the gaia-engine directory:
   ```
   cd /gaia/GAIA_Project/gaia-engine && python -m pytest tests/ -v --tb=short
   ```
3. **Commit to gaia-engine repo** — this is a separate git repo, NOT the monorepo:
   ```
   cd /gaia/GAIA_Project/gaia-engine && git add <files> && git commit -m "message"
   ```
4. **Push to gaia-engine remote**:
   ```
   cd /gaia/GAIA_Project/gaia-engine && git push origin main
   ```
5. **If the public API surface changed**, update these in the monorepo:
   - **Backward-compat shim**: `/gaia/GAIA_Project/gaia-common/gaia_common/engine/__init__.py`
   - **Contract YAML**: `/gaia/GAIA_Project/contracts/services/gaia-engine.yaml`
   - Commit those changes in the monorepo separately

## Important Rules

- Never import from gaia_common inside gaia-engine code. The shim goes the other direction.
- The `gaia_common.engine` package is a backward-compat re-export shim only. When adding new public exports to gaia-engine, add them to the shim as well.
- The lifecycle subpackage is shared state machine truth — orchestrator and all tiers depend on it. Changes here require careful coordination.
- When modifying HTTP endpoint signatures in managed mode, update the contract YAML to keep it in sync.

# GAIA Changelog

## v0.1.0 — 2026-04-08

First formal release. Marks the completion of Phase 2 (Configuration Harmonization & Tool Sovereignty) and the deployment of all three identity-baked model tiers.

### Architecture
- **Clutch Protocol** — ConsciousnessMatrix as sole tier authority. LifecycleMachine delegates via `apply_for_lifecycle()` with deadlock prevention. MEDITATION verified: full GPU clearance + auto-wake recovery.
- **Config Harmonization** — Orchestrator (config, tier router, consciousness matrix) loads from `gaia_constants.json` via `gaia_common.Config`. No more hardcoded model paths. Added `INFERENCE_ENDPOINTS` section.
- **CognitionPacket v0.4** — `sequence_id` and `total_fragments` in ResponseFragment. Gap/duplicate detection in fragment assembly. Stream integrity metadata in final NDJSON packet.

### Cognition
- **Neural Grounding (Stage 0)** — Nano-powered entity extraction before inference. KG → Vector → Web probe cascade. Results injected as `auto_grounding` DataField.
- **Hierarchy of Truth** — KG (MemPalace) first, Vector second, Web last. Encoded in ESSENTIAL_TOOLS order, TASK_INSTRUCTIONS, and prompt tool listing.
- **Tool Routing Cleanup** — Removed legacy aliases (`ai.read`, `ai.write`, `ai.execute`, `embedding.query`). Promoted 5 `kg_*` tools to ESSENTIAL_TOOLS.

### Models
- **Three-tier identity bake** — Qwen3.5-9B (Prime), Qwen3.5-4B (Core), Qwen3.5-0.8B (Nano) all identity-baked and deployed via symlinks.
- **Adaptive training pipeline** — Multi-phase train/test/repair loop with 44+ skill probes.
- **Model paths via symlinks** — `/models/prime`, `/models/core`, `/models/nano` (safetensors); `.gguf` variants for CPU.

### Infrastructure
- **MemPalace deployed** — Palace orchestrator + MCP tools live.
- **Tool consolidation** — 13 domains in prompt (down from 68 legacy tools).
- **Dev Notebook TODO** — Persistent cross-session task tracking.

### Fixes
- Prime probe misdetection (`backend:"cpp"` recognized as SUBCONSCIOUS)
- Orchestrator shutdown NameError (`_lifecycle_reconcile_task` removed)
- Stale model path defaults (all symlink-based now)
- `MODEL_CONFIGS.prime.path` updated from legacy 8B to current 9B

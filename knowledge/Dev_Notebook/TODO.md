# GAIA Development TODO

> Running task list. Persists across sessions. Update as work completes.
> Last updated: 2026-04-08

## In Progress

- [x] **Clutch protocol verification** — MEDITATION tested live: GPU cleared (9.8→2.9GB), auto-wake recovery confirmed, reconcile loop restores all tiers (2026-04-08)

## Phase 2: Configuration Harmonization (from Gemini v2 audit)

- [x] **Config harmonization** — Orchestrator config, tier router, and consciousness matrix all load from `gaia_constants.json` via `gaia_common.Config`. Added `INFERENCE_ENDPOINTS` and `core` to `MODEL_REGISTRY`. (2026-04-08)
- [x] **Tool routing cleanup** — Removed legacy aliases (ai.read etc.), promoted kg_* to ESSENTIAL_TOOLS, added Hierarchy of Truth (KG > Vector > Web) to task instructions and prompt tools (2026-04-08)
- [x] **Fragmentation enforcement** — CognitionPacket bumped to v0.4 with `sequence_id` + `total_fragments` in ResponseFragment. Gap/duplicate detection in fragment assembly. Stream integrity metadata in final packet. Discord client warns on violations. (2026-04-08)

## Architecture & Pipeline

- [x] **Pre-inference grounding (Neural Grounding Stage 0)** — Nano extracts entities, probes KG→Vector→Web per hierarchy, injects `auto_grounding` DataField into CognitionPacket before inference. GROUNDING_CONFIG in constants. (2026-04-08)
- [ ] **Native tool calling** — Curriculum complete (100 samples, 10 domains, 11 chains, 12 refusals). Next: LoRA train on 9B + 4B with tool_calling_v1_full.jsonl.
- [x] **RAG + self-exploration (Architectural RAG)** — `scripts/index_architecture.py` extracts AST summaries + contracts into `code_architecture` vector collection. 9 services, 21 docs, 179 chunks indexed. (2026-04-08)

## Orchestrator Quality

- [x] **Parallel Observer pipeline** — Always-on CPU Observer audits GPU Operator/Thinker stream in background. Role-symmetric: AWAKE=Prime observes Core, FOCUSING=Core observes Prime. Non-blocking via ThreadPoolExecutor. Can interrupt on safety/accuracy/epistemic issues. (2026-04-08)
- [x] **Model pool staleness** — `/refresh_pool` endpoint on gaia-core clears stale `gpu_prime`/`cpu_prime` entries. ConsciousnessMatrix triggers it after every tier transition. Auth whitelisted. (2026-04-08)

## Training & Models

- [ ] **Nano adaptive training** — v2/v3 failed at Phase 1 (v1 succeeded). Investigate regression; `anti_confabulation` and `restraint` skills remain difficult across all model sizes.
- [ ] **SAE validation pipeline** — Validate adapter merges with SAE activation monitoring before deploying to production. Progressive merge, not yolo.

## Completed

- [x] **Clutch protocol implementation** — CM as sole tier authority, delegation API, deadlock prevention (2026-04-08)
- [x] **Prime probe fix** — CM misread `backend:"cpp"` as unconscious; fixed to recognize GGUF/CPU as subconscious (2026-04-08)
- [x] **Stale model path defaults** — CM hardcoded v2/v4/v6 paths; now uses symlinks (`/models/prime.gguf`, `/models/core`, etc.) (2026-04-08)
- [x] **Shutdown NameError** — Removed dead `_lifecycle_reconcile_task` cleanup from orchestrator shutdown (2026-04-08)
- [x] **v1 identity-baked models deployed** — Nano 0.8B, Core 4B, Prime 9B all merged and live (2026-04-08)
- [x] **Adaptive training pipeline** — Multi-phase train/test/repair loop with 44+ skill probes (2026-04-08)
- [x] **Nano simplified curriculum** — 32 training samples, 3 adapters trained (2026-04-08)
- [x] **MemPalace deployed** — Palace orchestrator + MCP tools live (2026-04-07)
- [x] **Tool consolidation** — 13 domains wired in prompt (2026-04-07)

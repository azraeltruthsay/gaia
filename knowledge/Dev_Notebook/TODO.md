# GAIA Development TODO

> Running task list. Persists across sessions. Update as work completes.
> Last updated: 2026-04-08

## In Progress

- [x] **Clutch protocol verification** — MEDITATION tested live: GPU cleared (9.8→2.9GB), auto-wake recovery confirmed, reconcile loop restores all tiers (2026-04-08)

## Phase 2: Configuration Harmonization (from Gemini v2 audit)

- [x] **Config harmonization** — Orchestrator config, tier router, and consciousness matrix all load from `gaia_constants.json` via `gaia_common.Config`. Added `INFERENCE_ENDPOINTS` and `core` to `MODEL_REGISTRY`. (2026-04-08)
- [ ] **Tool routing cleanup** — Remove legacy tool aliases (`ai.read` -> `read_file` etc.) from `AgentCore._execute_mcp_tool`. Promote `kg_*` tools to `ESSENTIAL_TOOLS`. Establish Hierarchy of Truth (KG > Vector).
- [ ] **Fragmentation enforcement** — Add `SequenceID` to `ResponseFragment` in CognitionPacket v0.4. Stream integrity verification in `gaia-web`. Auto-trigger fragmentation at 10K char mark.

## Architecture & Pipeline

- [ ] **Pre-inference grounding** — Auto-extract entities from prompts, search KB + web BEFORE inference, inject into cognitive packet. Nano regex + Observer LLM + vector search. ~30-80ms overhead.
- [ ] **Native tool calling** — Train LoRA adapter for inline `<tool_call>` emission. Replace 3-step heuristic/selection/review pipeline. ~100 training examples across 9 domains.
- [ ] **RAG + self-exploration** — Index contracts/services/*.yaml + AST summaries into vector store. CFR compression into `code_architecture` KV prefix segment. Hash-based staleness invalidation.

## Orchestrator Quality

- [ ] **Core cognitive overhead** — `run_turn` takes ~60s (reflection, audit) before quality gate. Needs streamlining for interactive speed.
- [ ] **Model pool staleness** — After FOCUSING->AWAKE, `gpu_prime` stays registered in Core's model pool even though Prime moved to CPU. Causes ReadTimeout until pool refresh. (Partially addressed by CM probe fix, but Core's internal pool still needs refresh logic.)

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

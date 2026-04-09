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
- [x] **Native tool calling** — COMPLETE. Combined curriculum (575 samples), r=32/alpha=64 on clean Qwen bases. Core 4B: 14/14 skills. Prime 9B: 12/14 (14/14 after validator fix). Both v2 models deployed. E2E verified by Gemini: identity + tool_call emission both pass through full cognitive pipeline. Action alias `run_shell→run` fixed. (2026-04-09)
- [x] **RAG + self-exploration (Architectural RAG)** — `scripts/index_architecture.py` extracts AST summaries + contracts into `code_architecture` vector collection. 9 services, 21 docs, 179 chunks indexed. (2026-04-08)

## Orchestrator Quality

- [x] **Parallel Observer pipeline** — Always-on CPU Observer audits GPU Operator/Thinker stream in background. Role-symmetric: AWAKE=Prime observes Core, FOCUSING=Core observes Prime. Non-blocking via ThreadPoolExecutor. Can interrupt on safety/accuracy/epistemic issues. (2026-04-08)
- [x] **Model pool staleness** — `/refresh_pool` endpoint on gaia-core clears stale `gpu_prime`/`cpu_prime` entries. ConsciousnessMatrix triggers it after every tier transition. Auth whitelisted. (2026-04-08)
- [x] **Core Cognitive Overload fix** — Fixed `AttributeError: 'dict' object has no attribute 'encode'` in `gaia-core/main.py`. `process_packet` now correctly serializes unknown dictionary events (e.g., from self-improvement) before yielding to `StreamingResponse`. (2026-04-09)

## Training & Models

- [ ] **Nano adaptive training** — v2/v3 failed at Phase 1 (v1 succeeded). **FIX PENDING (Candidates)**: Implemented Nano auto-detection and strict phase-mapping in `candidates/gaia-study` to prevent _Phase Drift_. Eval now scoped to `NANO_SKILLS` for 0.8B models. Awaiting validation run.
- [x] **SAE validation pipeline** — Baseline + adapter atlases recorded on Core 4B. Mid-layers (11-17) show expected drift from tool-calling injection. Identity layers (23, 26) stable (<5% loss delta). Adapter cleared for runtime loading. (2026-04-08)

## Sovereign Shield & Security

- [x] **Blast Shield hardening** — Replaced substring matching with shlex tokenization + regex word boundaries. Validates chained commands individually. Path validation uses `os.path.realpath()` + `..` detection. Added chmod 777, chown root, setfacl, find -delete. 17/17 tests pass. (2026-04-09)
- [x] **Prompt Injection canaries** — Per-session canary tokens injected into system prompt via prompt_builder.py. Scanner Tier 3 detects `[CANARY:hash]` in user input as prompt extraction attack (severity=BLOCK, score+=0.50). (2026-04-09)
- [x] **Nano-Injection reliability** — Phase Drift fix validated: NANO_SKILLS (5 skills) correctly scoped via auto-detection in adaptive_trainer.py. Nano only evaluated on greeting, identity, restraint, transcript_cleanup, triage. (2026-04-09)

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

# January 2026 — Consolidated Dev Journal

> Archived from 18 individual files on 2026-02-24.
> This was the monolith-to-SOA transition period.

---

## Jan 17: Fragmentation & Sketchpad Architecture

- Identified flaw: `gpu_prime` generating entire responses in single stream instead of reflective multi-step
- Refactored `_run_with_fragmentation()` to true fragment→store→reflect→assemble workflow
- Assembly delegated to model (cognitive task, not mechanical concatenation)
- Sketchpad becomes central to fragmented generation; in-memory fallback ensures no data loss

## Jan 18: Epistemic Confidence Assessment

- Pre-task confidence check before recitation/long-form tasks
- Thresholds: <0.5 decline, 0.5-0.8 warn, >=0.8 proceed
- **Critical pattern established**: All cognition flows through CognitionPacket (no naked model calls)
- Every interaction includes: identity context, world state, tool awareness, proper packet structure

## Jan 21: Think-Tag Stripping, Document Retrieval, Self-Improvement

- `strip_think_tags()` utility applied at 3 output locations
- `RECITABLE_DOCUMENTS` mapping for Constitution, Layered Identity Model, etc.
- `snapshot_manager.py` backup/rollback system: timestamped backups with metadata, `safe_edit()` with atomic rollback on validation failure
- Self-improvement orchestration: LLM analyzes code → proposes fixes → applies via `safe_edit()`

## Jan 22: Discord DM Support

- DM detection via `message.guild is None`, configurable via `DISCORD_RESPOND_TO_DMS`
- Session ID convention: `discord_dm_{user_id}`
- Full metadata propagation: channel_id, guild_id, author_name, author_id through entire pipeline
- Multi-source logging architecture established (CLI, Discord DM, Discord channel, web, API)

## Jan 23: GCP Tool Routing System

- Two-stage tool selection: low-temp (0.15) structured JSON selection + Prime confidence review
- `ToolExecutionStatus` lifecycle: PENDING → AWAITING_CONFIDENCE → APPROVED → EXECUTED
- Confidence threshold 0.7; write/execute tools disabled by default
- Reinjection limit prevents infinite loops; full audit trail in packet's `reflection_log`
- Config: `TOOL_ROUTING` section in gaia_constants.json

## Jan 23: LoRA Adapter Architecture

- Three-tier system: Tier I (Global/Immutable), Tier II (User/Persona), Tier III (Session/Ephemeral)
- Study Mode lifecycle: READY → ENTERING → PREPARING → STUDYING → VALIDATING → RELOADING → READY
- QLoRA config: load_in_4bit, nf4, lora_r=8, max_steps=100, 10-min timeout
- Self-directed research pipeline: gap detection → web search → source validation → constitutional review → training
- Governance: Tier I requires operator approval; rate limits (20 searches/hr, 5 auto-learns/day)

## Jan 24: Dynamic Persona Switching

- Default "dev" persona, switch to specialized personas based on intent (e.g., D&D)
- Persona state managed turn-by-turn in CognitionPacket (stateless approach)
- Keyword-based intent detection as pragmatic first step

## Jan 26-27: RAG Debugging

- RAG failure root cause: vector index deleted by test but never rebuilt
- Observer bypass: hardcoded `GAIA_BACKEND=gpu_prime` skipped observer logic
- Multiple silent failure points identified and fixed with proper error logging

## Jan 28: Greenfield Migration (4 Sessions)

- Migrated from monolithic `gaia-assistant` to modular services: gaia-core, gaia-study, gaia-common
- Created `gaia-core/gaia_core/` with subdirectories: behavior, cognition, models, pipeline, memory, utils
- 50+ files migrated; all `app.*` imports eliminated from gaia-core and gaia-common
- Graceful fallbacks for unmigrated modules (semantic_codex, external_voice, stream_observer → None)
- 12 modules still needed migration at end of session

## Undated: Supporting Plans

- **Prime Dual Backend**: Separate `cpu_prime` and `gpu_prime` in MODEL_CONFIGS with `prime` alias
- **GPT Suggestions**: Oracle Consults, Knowledge Logging, LangGraph migration ideas, Visual mental maps
- **CoPilot Model Integration**: Centralized model verification via `setup_models.py`, symlink strategy
- **Container Data Management**: Dockerfile COPY + docker-compose volume mounts for dev/prod flexibility

## Key Architectural Themes

1. **CognitionPacket as Central Authority** — every model interaction flows through GCP
2. **Multi-Tier Expertise** — tool routing, persona switching, LoRA adapters all follow tiered patterns
3. **Non-Destructive Operations** — backup/rollback, graceful fallbacks, reversible migrations
4. **Learning & Self-Improvement** — Study Mode, confidence assessment, knowledge gap detection
5. **Modular Architecture** — gaia-core (inference), gaia-study (training), gaia-common (shared)

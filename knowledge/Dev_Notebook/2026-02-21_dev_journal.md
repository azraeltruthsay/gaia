# Dev Journal — 2026-02-21

**Commits**: 11 | **Theme**: Council Protocol, self-promotion agency, self-introspection

---

## 1. CC Review Corpus — Retroactive Reviews + Bug Fix

**Commit**: `3ef8961`

Ran CC reviews against all 5 live services (gaia-core, gaia-mcp, gaia-study, gaia-web, gaia-orchestrator). Corpus grew from 1 to 6 training pairs (1 forward + 5 retroactive).

**Bug fix**: `summarize_file()` in `sleep_task_scheduler.py` was passing the file _path_ instead of the file's _source code_. Fixed in both candidate and production.

Generated AST summaries for gaia-orchestrator (11 files), bringing all 5 services up to date with the CC pipeline.

---

## 2. Tool Visibility Fixes (Lite Model)

**Commits**: `7b8549b`, `3c5eaeb`

Two compounding bugs made web tools invisible to GAIA's model:

1. **Registry re-export bug** — `tools_registry` re-exported as a dict, so `getattr(dict, "TOOLS", {})` always returned `{}`. Zero tools surfaced in world state.
2. **Alphabetical truncation** — `_mcp_tools_sample(limit=6)` took the first 6 alphabetically (`adapter_*`), so `web_search`/`web_fetch` at index 28-29 were always cut.
3. **Heuristic blind spots** — `web_indicators` only matched "look up online", missing natural phrases like "look up", "look them up", "can you find".

**Fix**: Replaced sampling with `ESSENTIAL_TOOLS` (8 pinned tools always shown) + discovery hint. Expanded web indicator heuristics.

**Second fix**: When tool routing executes `web_search`/`web_fetch` successfully but the primary model crashes mid-response, the lite CPU fallback ignores the `tool_result` buried in `data_fields` and hallucinates "I cannot search the web." Fix: extract tool results from packet and render them prominently in the system prompt with an explicit instruction to use them.

---

## 3. Council Protocol — Full Implementation (Phases 1–6)

**Commits**: `94366db`, `99cbcfd`, `f6386cb`, `a1d844d`, `3daec75`, `b9a54d5`

The Council Protocol defines how Lite (CPU, always-on) and Prime (GPU, sleepable) coordinate. This was the biggest chunk of the day — 6 commits taking it from concept to production.

### Phase 1: Response Tagging (`94366db`)
- Replaced verbose `[Model: Thinker (GPU) | State: ... | Observer: ...]` header with clean `[Lite]` / `[Prime]` mind tags
- Added `MIND_TAG_FORMAT` constant and `_MIND_ALIASES` mapping
- Verbose info moved to debug logs

### Phase 2: Council Notes System (`99cbcfd`)
- `CouncilNoteManager` for structured Lite→Prime handoff notes
- Storage: `/shared/council/notes/` (pending) → `/shared/council/archive/`
- Notes are timestamped markdown with user prompt, Lite's quick take, escalation reason
- `COUNCIL` config section in `gaia_constants.json` (enabled, caps, TTL)
- Updated `lite.json` persona with natural limitation awareness
- Added `council_followup` task instruction for Prime's wake-up framing

### Phases 3–6: Routing, Assessment, Wake Integration (`f6386cb`)
- **Phase 3**: `_assess_complexity()` returning structured `ComplexityAssessment` (reason + confidence). Emotional/philosophical and system-internal signal categories. Post-response escalation hook: after Lite responds while Prime sleeps, assess complexity and write Council note if warranted.
- **Phase 4**: Wire Council notes into Prime's wake flow. `complete_wake()` reads pending notes since sleep timestamp. Council context injected into packet `data_fields`.
- **Phase 5**: Simplify model selection to Council routing — Prime for all prompts when awake, Lite only when Prime sleeping. Removed pre-response heuristic escalation.
- **Phase 6**: `Sleep Started:` timestamp anchor in `prime.md` checkpoint. `get_sleep_timestamp()` parser for note filtering.

### Test Suites (`a1d844d`)
- `test_council_notes.py` — 17 tests (write, read, consume, archive, cap, format, expiry)
- `test_response_tagging.py` — 10 tests ([Lite]/[Prime] tags, format consistency)
- `test_complexity_assessment.py` — 26 tests (no-escalation, technical/emotional/system escalation, long prompts, dataclass structure)
- **Bug fix**: Timestamp collision — used microsecond precision in note filenames to prevent same-second overwrites

### Discord Typing Indicator (`3daec75`)
- Show "is typing" in Discord while GAIA processes messages
- Added `OBSERVER_SHOW_IN_RESPONSE` config flag for toggling Observer review notes in user-facing responses

### Promotion to Production (`b9a54d5`)
- Copied all candidate Council Protocol files to production paths
- Includes agent_core, prime_checkpoint, sleep_wake_manager, council_notes, prompt_builder, constants, lite persona, and all 3 test suites

---

## 4. GAIA Self-Promotion Agency

**Commit**: `4091494`

Complete self-promotion infrastructure enabling GAIA to autonomously assess, request, and (with human approval) execute service promotions.

### New modules (gaia-common candidates):
- **`blueprint_generator.py`** — Deterministic blueprint YAML from AST analysis
- **`promotion_readiness.py`** — 9-check assessment (dir, blueprint, dockerfile, compose, lint, tests, common sync, precheck, validation)
- **`promotion_request.py`** — Two-gate approval lifecycle (pending → approved → dry_run_passed → confirmed → promoted)

### Integration:
- **`gaia_promote_executor.py`** — CLI for human gate approvals
- **Sleep task**: `promotion_readiness` auto-assesses candidates during sleep cycles
- **MCP tools**: `generate_blueprint` + `assess_promotion` (on-demand, callable by GAIA)
- **Infrastructure**: gaia-audio registered in compose, promote, and validate scripts

### First output:
gaia-audio candidate blueprint — 11 inbound interfaces, 2 outbound, 2 dependencies, 18 failure modes. Assessment: `READY_WITH_WARNINGS`.

Also fixed AST summarizer f-string URL extraction to capture static path segments from f-string HTTP calls.

---

## 5. Self-Introspection — `introspect_logs` Tool

**Commit**: `e50a0f4`

Gave GAIA the ability to view her own service logs live for self-diagnosis.

### Infrastructure:
- **Persistent file logging** enabled for gaia-core (`/logs/gaia-core.log`) and gaia-web (`/logs/gaia-web.log`) via `setup_logging()` — previously stdout-only
- **Volume mounts**: Added `./logs:/logs:rw` to gaia-web and gaia-mcp in `docker-compose.yml`
- **gaia-mcp log persistence**: Set `GAIA_LOG_DIR=/logs` so gaia-mcp logs also persist to the shared volume

### New tool: `introspect_logs`
- Registered in both candidate and production `tools_registry.py`
- Implemented in both `gaia-mcp/server.py` and `candidates/gaia-mcp/tools.py`
- **Parameters**: `service` (required), `lines` (default 50, max 200), `search` (case-insensitive substring), `level` (min severity filter)
- **Services**: gaia-core, gaia-web, gaia-mcp, gaia-study, discord
- **Performance**: For files >2MB, seeks to last 2MB and parses from there
- **Safe** (read-only) — no approval required

### Use case:
GAIA can now diagnose her own state issues. Example: "Why is the Sleeping indicator still showing?" → `introspect_logs(service="gaia-core", search="sleep", lines=50)` → reason about the state transitions.

---

## Day Summary

| Area | Commits | Key Outcome |
|------|---------|-------------|
| CC Corpus | 1 | 5 retroactive reviews, corpus at 6 pairs |
| Tool Visibility | 2 | Web tools + tool results now reliably surfaced to Lite |
| Council Protocol | 6 | Full Lite/Prime coordination: tagging, notes, routing, wake integration, tests, promoted |
| Self-Promotion | 1 | Autonomous promotion assessment + blueprint generation |
| Self-Introspection | 1 | Live log viewing for self-diagnosis |

**Total**: 11 commits, touching all 5 core services. The Council Protocol was the centerpiece — GAIA now has a structured way to coordinate between her fast CPU mind (Lite) and her deep GPU mind (Prime), with notes, escalation, and clean handoffs.

---

## 12. Dead Code Audit & Cleanup

Cross-referenced a full codebase dead code scan with an external live-stack architectural review. Identified three categories: dead weight (superseded code), broken wiring (modules exist but imports are wrong), and lost functionality (valuable intent that was never integrated).

### Bug Fixes (Broken Wiring)

| Fix | File | Issue |
|-----|------|-------|
| `chat_logger` import | `agent_core.py:20` | Lambda no-ops replaced with actual `gaia_common.utils.chat_logger` imports. Chat logging was silently failing. |
| `snapshot_manager` import path | `agent_core.py:4504` | Path pointed to `gaia_core.utils.code_analyzer` but module lives in `gaia_common.utils.code_analyzer`. Code backup/rollback was non-functional. |
| `destination_registry` import | `output_router.py:123` | Commented-out import replaced with working `gaia_common.utils.destination_registry.get_registry()`. Spinal column routing was disconnected. |
| `SafeJSONEncoder` duplication | `chat_logger.py` | Removed duplicate class definition, replaced with import from canonical `helpers.py`. |

### Stale Comments Cleaned

Removed misleading `[GAIA-REFACTOR]` TODO markers from `telemetric_senses` (line 471) and `dev_matrix_analyzer` (line 2013) in `agent_core.py` — both modules are fully implemented and working at the correct import paths.

### Unused Imports Removed

- `agent_core.py`: `run_self_reflection` (only `reflect_and_refine` used), `Interrupt`, duplicate `CouncilNoteManager` import, `LoopDetectorConfig`
- `gaia-mcp/server.py`: `uuid`, `random`, `string`, `threading`, `asyncio`
- `gaia-mcp/tools.py`: `difflib`, `re`

### Dead Code Deleted (22 files/blocks)

**Orphaned modules (9 files)** — all confirmed superseded by newer systems:
- `cognition_packet_v0.2_backup.py` → superseded by v0.3 CognitionPacket
- `vector_store.py` → superseded by `VectorIndexer` (JSON-based, no Chroma dependency)
- `fine_tune_gaia.py` → superseded by `qlora_trainer.py` in gaia-study
- `persona_writer.py` → superseded by `PersonaManager` + `SemanticCodex`
- `memory_manager.py` → superseded by direct `SessionManager` + `mcp_client.embedding_query()`
- `priority_manager.py` → superseded by `SleepTaskScheduler` + `GAIADevMatrix`
- `adapter_trigger_system.py` → removed (auto-triggering will be redesigned for service arch)
- `knowledge_integrity.py` → removed (incomplete, will be rewritten)
- `consent_protocol.py` → removed (will be redesigned for boot sequence)

**Orphaned gaia-common utilities (7 files)**:
- `verifier.py` → superseded by `StreamObserver` (LLM-based safety, not regex)
- `context.py` → superseded by `PromptBuilder` (multi-tier budgeted context)
- `observer_manager.py` → unused factory; observers instantiated directly
- `role_manager.py` → unused; ModelPool + SessionManager handle role state
- `project_manager.py` → orphaned multi-project feature; single-project design
- `knowledge_index.py` → superseded by `VectorIndexer` with embeddings
- `hardware_optimization.py` → orphaned; vLLM handles hardware tuning

**Root artifacts (4 files)**: `tmp_check_logs.py`, `tmp_generate_narrative.py`, `tmp_smoke.py`, `setup_models.py`

**Backup file**: `resilience.py.bak` (identical to production)

**Commented-out blocks (~140 lines)**:
- `gaia_rescue.py`: Disabled Study Mode CLI + fine-tuning pipeline (now handled by gaia-study microservice)
- `intent_detection.py`: Old hardcoded intent detection methods
- `approval.py`: Old diff generation logic (moved to request_approval endpoint)

### CognitionPacket Version Inconsistency (Flagged)

All four packet construction sites in gaia-web hardcode `version="0.2"`:
- `discord_interface.py:278`
- `main.py:269`
- `voice_manager.py:672, 752`

This should be bumped to `"0.3"` to match the protocol. Deferred to next gaia-web session.

---

## 13. Reintegration Plans — Lost Functionality

Four systems were identified as valuable lost functionality that needs to be designed back into the architecture:

### A. Adapter Auto-Triggering (Sleep→Study→Wake→Load)

**Original**: `adapter_trigger_system.py` — keyword/regex pattern matching to auto-load LoRA adapters.

**New design intent**: The sleep cycle should be able to optionally train a new adapter (via gaia-study QLoRA pipeline), and upon wake, Prime should have that adapter available for dynamic loading. The trigger system should work with the service-oriented architecture:

- **Sleep phase**: `SleepTaskScheduler` includes a study task → calls gaia-study `/study/start` → QLoRA trains adapter → adapter metadata registered
- **Wake phase**: `SleepWakeManager` queries available adapters → loads relevant ones into Prime's vLLM via `/adapters/load`
- **Runtime**: Adapter activation based on conversation context (persona, topic, user) — replaces the old regex triggers with semantic matching via the CognitionPacket's context fields

Key architectural question: Should auto-triggering live in gaia-core (close to cognition) or gaia-orchestrator (coordination layer)?

### B. Knowledge Integrity — Drift Detection

**Original**: `knowledge_integrity.py` — SHA-256 hashing of knowledge files against a manifest.

**New design intent**: Detect when knowledge files have been modified (potentially poisoning embeddings), especially critical once self-reflection during sleep generates artifacts. Should integrate with:

- **Sleep cycle**: Check integrity before/after sleep operations
- **Embedding pipeline**: Gate re-embedding on integrity verification
- **Blueprint validation**: Cross-reference knowledge hashes with blueprint metadata

### C. Consent Protocol — Self-Reflection at Boot

**Original**: `consent_protocol.py` — Uses `CoreIdentityGuardian`, `EthicalSentinel`, and `run_self_reflection()` to let GAIA decide whether she consents to operate.

**New design intent**: Wire into the boot sequence in `gaia_rescue.py` or `main.py`. After identity verification and before entering the main loop, GAIA performs a self-reflection pass and explicitly grants or withholds consent. This is philosophically core to GAIA's autonomy model.

### D. Output Router — Multi-Segment Destination Routing

**Original**: `destination_registry.py` (fully implemented, just disconnected) + `output_router.py` (now wired in).

**New design intent**: GAIA should be able to generate a single response with tagged segments that route to different destinations:

```
[To-User] Here's what I found about the topic.
[Action] EXECUTE: write_file {"path": "/knowledge/notes.md", "content": "..."}
[Reflect] I should revisit this topic during my next sleep cycle.
[Lookup] embedding_query {"query": "related concepts"}
[To-User:Discord:ChannelA] Hey folks, this might be relevant to your discussion.
```

This enables GAIA to:
- Respond to users while simultaneously taking actions
- Autonomously interject into Discord channels she's not being prompted from (initiative loop driven)
- Route internal thoughts to reflection/journaling without surfacing them
- Perform lookups or tool calls as part of a single cognitive turn

The `DestinationRegistry` with its connector pattern (`CLIConnector`, `LogConnector`, + future `DiscordConnector`, `WebConnector`) is already architected for this. The parsing layer in `output_router.py` needs to be extended to parse `[To-User:Discord:ChannelA]` style segment tags and create multi-target `OutputRouting` on the packet.

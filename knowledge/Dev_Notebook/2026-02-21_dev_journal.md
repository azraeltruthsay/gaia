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

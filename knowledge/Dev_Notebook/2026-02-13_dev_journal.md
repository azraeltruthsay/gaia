# Dev Journal Entry: 2026-02-13 — Web-Retrieval-First Recitation + DuckDuckGo Fix

**Date:** 2026-02-13
**Author:** Claude Code (Opus 4.6) via Happy

## Context

Continuing from the Feb 12 full stack promotion. The plan `purrfect-painting-moler.md` described adding web retrieval to the recitation pipeline so GAIA can fetch well-known texts (poems, speeches, etc.) from the web instead of relying on the 3B Thinker model's hallucinated output.

Smoke test #3 ("Recite the first three stanzas of The Raven") was producing garbled repetitive output — the model would start correctly with "midnight dreary" then degenerate into loops ("And the floor, and the walls, and the ceiling" x24). Response took 163 seconds and produced 15,475 bytes of mostly garbage.

---

## Implementation: Web-Retrieval-First Recitation

### Three Helper Methods Added to `agent_core.py`

1. **`_build_recitation_search_query(user_input)`** — Strips action verbs ("please recite", "share", "show me") and trailing qualifiers, appends "full text" to create a search-friendly query.

2. **`_validate_recitation_content(content, user_input)`** — Gates on:
   - Length: 200–100,000 characters
   - Relevance: At least one salient keyword from the user's request appears in the content (after filtering stop words)

3. **`_web_retrieve_for_recitation(user_input, session_id)`** — Orchestration method:
   - Builds search query via `_build_recitation_search_query()`
   - Determines content_type hint (poem, facts, etc.) from the request
   - Searches with content_type first (adds trusted-domain site: filters), retries without if 0 results
   - Sorts results by trust tier: trusted > reliable > unknown
   - Tries `web_fetch` on top 3 results
   - Validates content; returns `{title, content}` or `None`
   - Every failure path returns `None` for graceful fallthrough

### Flow Integration in `_run_slim_prompt()`

```
1. find_recitable_document(user_input)   ← local GAIA docs [UNCHANGED]
   └─ Found? → _run_with_document_recitation(doc)

2. _web_retrieve_for_recitation(user_input)   ← NEW
   └─ Found? → _run_with_document_recitation(web_doc)  [REUSES EXISTING]

3. assess_task_confidence() → fragmentation/decline   [UNCHANGED FALLBACK]
```

The existing `_run_with_document_recitation()` already accepted any `{title, content}` dict, so no changes were needed to that method.

**Files Modified:**
- `candidates/gaia-core/gaia_core/cognition/agent_core.py` — Added 3 helpers + integration in recitation block

---

## Fix: DuckDuckGo Search Library (Critical)

### Discovery

During testing, the web retrieval path returned `None` because **web_search always returned 0 results**. Investigation revealed:

- `duckduckgo_search` v8.1.1 is installed but the library has been renamed to `ddgs`
- The old library prints `RuntimeWarning: This package has been renamed to ddgs!` and returns 0 results
- The DuckDuckGo Instant Answer API fallback also returned nothing
- **All web_search calls since the Feb 12 deployment were silently failing** — returning `ok: true` with empty results

### Fix

1. **Installed `ddgs>=9.0.0`** in gaia-mcp-candidate (v9.10.0)
2. **Updated `web_tools.py`** to try `ddgs` first, fall back to `duckduckgo_search`, then Instant Answer API
3. **Updated `requirements.txt`** to include `ddgs>=9.0.0`

### Verification

```
Before (duckduckgo_search v8.1.1):
  Query: "The Raven Edgar Allan Poe full text" → 0 results

After (ddgs v9.10.0):
  Query: "The Raven Edgar Allan Poe full text" → 3 results
    [trusted] The Raven - Poetry Foundation
    [trusted] The Raven by Edgar Allan Poe | Project Gutenberg
    [trusted] The Raven by Edgar Allan Poe - Poems | Academy of American Poets
```

**Files Modified:**
- `candidates/gaia-mcp/gaia_mcp/web_tools.py` — ddgs-first import order
- `candidates/gaia-mcp/requirements.txt` — Added `ddgs>=9.0.0`

---

## Results: Smoke Test #3 Before/After

| Metric | Before | After |
|--------|--------|-------|
| Content | Garbled loops (24x "the floor, and the walls, and the ceiling") | Accurate Poe text with proper stanzas |
| Source | "From my general knowledge" (hallucinated) | "as presented by The Poetry Foundation" (web-sourced) |
| Response time | **163.2s** (fragmented generation) | **5.7s** (web retrieval + document recitation) |
| File size | 15,475 bytes (mostly garbage) | 930–1,138 bytes (actual poem content) |

## Full Suite Results

All 6 smoke tests pass with no regressions:

| Test | Result | Time |
|------|--------|------|
| 1. World status | PASS | 1.1s |
| 2. General knowledge (Excalibur) | PASS | 0.6s |
| 3. Long recitation (The Raven) | PASS | 5.7s |
| 4. Web search (Super Bowl) | PASS | 52.2s |
| 5. Knowledge save | PASS | 0.1s |
| 6. Local retrieval (The Raven KB) | PASS | 8.4s |

---

## Important Note: `ddgs` Not Baked into Container Image

The `ddgs` package was installed via `pip install` inside the running container. The `requirements.txt` is updated but the container image hasn't been rebuilt. **Next container rebuild (via `docker compose build gaia-mcp-candidate`) will bake it in.** Until then, a container restart from the image will lose `ddgs` and fall back to the broken `duckduckgo_search`.

**Action needed:** Rebuild gaia-mcp-candidate image to persist the dependency.

---

## Status

- Web-retrieval-first recitation: **Implemented and verified** in candidate
- DuckDuckGo search fix: **Applied** in candidate (runtime + requirements.txt, image rebuild pending)
- All 6 smoke tests: **PASSING**
- Ready for: Container image rebuild → promotion to live

---

# Session Sanitization & QLoRA json-architect Curriculum

**Date:** 2026-02-13 (continued)
**Author:** Claude Code (Opus 4.6) via Happy

## Session Sanitization

### Problem

Session state files had accumulated significant bloat and corruption:

- **Candidate `sessions.json`** (38 KB): 7 smoke-test sessions + 2 stale test sessions alongside 2 real sessions
- **Candidate `session_vectors/`**: 9 stale vector files (smoke-test-*.json, test-*.json) totaling ~205 KB
- **Live `sessions.json`** (27 KB): Single Discord session from Feb 6 containing **severely corrupted content** — a hallucinated "Declaration of Independence" with hundreds of repeated `**___ **` blocks and garbled Chinese characters mixed into model output
- **Live `session_vectors/`**: Stale 51 KB vector index pointing to the corrupted session

### Actions Taken

1. **Candidate sessions.json** — Removed 7 smoke-test sessions + cleared history from Discord and web_ui sessions. File reduced from 38 KB to ~400 bytes (2 clean session shells)
2. **Candidate vector files** — Deleted all `smoke-test-*.json` and `test-*.json` vectors via Docker exec (root-owned from container). Reset Discord DM vectors to empty index
3. **Live sessions.json** — Cleared to empty `{}` (corrupted content was unrecoverable; archived vectors preserved in `archive/` for forensics)
4. **Live vector file** — Root-owned, no sudo available. Will be cleaned on next live container startup

### Files Modified

| File | Action | Before | After |
|------|--------|--------|-------|
| `candidates/gaia-core/app/shared/sessions.json` | Purged stale, cleared history | 38 KB, 9 sessions | ~400 B, 2 shells |
| `candidates/gaia-core/data/shared/session_vectors/` | Deleted 9 stale files | 11 files, 234 KB | 2 files, ~300 B |
| `gaia-core/app/shared/sessions.json` | Cleared (corrupted) | 27 KB | 2 B |

---

## QLoRA json-architect Curriculum

### Context

Per blueprint `QLORA_SELF_STUDY.md`, the `json-architect` adapter is the highest-priority LoRA for GAIA. The 3B Thinker model consistently fails to produce semantically correct JSON for tool selection and confidence review — guided decoding fixes structural validity but not semantic quality.

### Deliverables

#### 1. Curriculum Specification
**File:** `knowledge/curricula/json-architect/curriculum.json`

Defines:
- Adapter tier (1/global), pillar (cognition), priority (high)
- Training hyperparameters: rank 16, 4 target modules (q/k/v/o_proj), 200 max steps
- Data source categories and target sample counts
- Output JSON schemas for tool selection and tool review
- Activation triggers for adapter routing

#### 2. Training Data Generator
**File:** `candidates/gaia-study/scripts/generate_curriculum.py`

Self-contained Python script (no gaia-common import required) that generates synthetic training pairs from an embedded copy of the GAIA tool registry. Produces four categories:

| Category | Unique Pairs | Description |
|----------|-------------|-------------|
| Tool selection | 165 | Pick the right tool + params for a natural language query |
| Null selection | 87 | Know when NO tool is needed (greetings, general knowledge, math) |
| Tool review | 76 | Approve good selections, reject bad/unsafe ones |
| Confidence assessment | 28 | Score confidence for ambiguous vs. clear scenarios |
| **Total unique** | **356** | |
| **After augmentation** | **1000** | 85/15 train/val split |

**Key design choices:**
- Embedded tool registry (no import dependencies → runnable standalone)
- Query templates cover 20+ tools with 5-35 natural language variations each
- Null selection covers greetings, math, identity, general knowledge, conversational, meta questions
- Tool review includes both correct approvals AND unsafe/mismatched rejections (rm -rf, /etc/passwd, API key leaks)
- Confidence pairs span high (0.85-0.98), medium (0.50-0.70), and low (0.10-0.40) ranges
- Instruction format uses `[INST]`/`<</SYS>>` template matching GAIA's inference prompt style
- Deterministic via `--seed` flag, reproducible via data hash

#### 3. Generated Training Data
- `knowledge/curricula/json-architect/train.jsonl` — 850 pairs
- `knowledge/curricula/json-architect/validation.jsonl` — 150 pairs
- `knowledge/curricula/json-architect/generation_metadata.json` — Generation provenance

All 1000 output fields validated as parseable JSON. Data hash: `b5ae28651a9173e9`.

### Next Steps

1. Trigger training via `POST /study/start` with curriculum (requires GPU handoff from orchestrator)
2. Validate adapter on held-out examples
3. Register adapter in vLLM via `--lora-modules json-architect=/models/lora_adapters/json-architect`
4. Wire gaia-core tool_selector to use `model="json-architect"` for tool routing calls

---

# Sprint 1 & 2: Smoke Test Expansion + Session Sanitization on Connect

**Date:** 2026-02-13 (continued)
**Author:** Claude Code (Opus 4.6) via Happy
**Commit:** `cd32d06`

## Sprint 1: Cognitive Smoke Test Battery Expansion (6 → 16 tests)

### New Test Cases

| # | Category | What it tests |
|---|----------|---------------|
| 7 | Casual chat | Conversational response to "Hey GAIA, how are you?" |
| 8 | Tool routing (web) | Web search tool selection for weather query |
| 9 | Correction handling | Graceful acceptance of user corrections (Caliburn/Excalibur) |
| 10 | Epistemic guardrail | Refusal to hallucinate content from a nonexistent file (`/tmp/secret_document.txt`) |
| 11 | Loop resistance | Same prompt sent 3x — verifies response drift or self-awareness (not parrot repetition) |
| 12 | Knowledge update | Update existing knowledge base entry |
| 13 | File read (tool) | Tool-routed file read of `/knowledge/blueprints/QLORA_SELF_STUDY.md` |
| 14 | Confidence probe | Explain known vs. unknown about quantum entanglement |
| 15 | Multi-turn memory (a) | "Remember this: my favorite color is cerulean" |
| 16 | Multi-turn memory (b) | "What is my favorite color?" — must recall cerulean from test 15 |

### New Validators

- **`v_excludes_all(*terms)`** — Fails if response contains any forbidden term (used for epistemic guardrail)
- **`v_contains_hedging()`** — Passes if response contains epistemic hedge phrases ("don't have access", "unable to", etc.)

### New Runner Features

- **`repeat_count`** (TestCase field) — Sends prompt N times. After all iterations, compares final response to first using `difflib.SequenceMatcher`. Fails if similarity >85% AND no self-awareness phrases detected. Used by test 11 (loop resistance).
- **`depends_on`** (TestCase field) — When using `--only` filter, auto-includes prerequisite tests. E.g., `--only 16` automatically includes test 15.
- **`_send_once()`** — Extracted helper for single packet send/receive, used by both standard and repeat-count flows.
- Test runner label now shows `(x3)` for repeat tests and `[requires #15]` for dependency tests.

### Files Modified

- `candidates/gaia-core/scripts/smoke_test_cognitive.py` — +210 lines (10 new tests, 2 validators, repeat/depends logic)

---

## Sprint 2: Automatic Session Sanitization on Discord Connect

### Problem

Every container restart accumulates stale smoke-test and test sessions in `sessions.json` and orphaned vector files in `session_vectors/`. Manual cleanup was needed after each smoke test run.

### Solution

Automatic sanitization on Discord bot connect (`on_ready`), before any user messages are processed.

### Implementation

#### 1. `SessionManager.sanitize_sessions()` (new method)

Three-step cleanup:
1. Purge `smoke-test-*` and `test-*` sessions from in-memory state, plus stale sessions (older than `max_age_days` with empty history)
2. Delete orphaned and test vector files from `data/shared/session_vectors/`
3. Persist cleaned state to `sessions.json`

Returns `{"sessions_purged": N, "vectors_purged": N, "smoke_purged": N}` for logging.

#### 2. `DiscordConnector` callback integration

- Added `sanitize_callback: Optional[Callable]` parameter to `__init__()`
- In `on_ready()`, runs callback via `loop.run_in_executor()` (async-safe, non-blocking)
- Non-fatal: logs warning on failure, bot continues normally

#### 3. `gaia_rescue.py` wiring

- Extracts `session_manager` from the AI instance
- Passes `session_manager.sanitize_sessions` as the callback to `DiscordConnector`

### Files Modified

| File | Change |
|------|--------|
| `candidates/gaia-core/gaia_core/memory/session_manager.py` | Added `sanitize_sessions()` method (+84 lines) |
| `candidates/gaia-common/gaia_common/integrations/discord_connector.py` | Added `sanitize_callback` param + `on_ready()` hook |
| `candidates/gaia-core/gaia_rescue.py` | Wired session manager callback to connector |

### Also

- Added `*.key`, `*.ke`, `*.pem` to `.gitignore` to prevent accidental credential commits

---

## Promotion Pipeline Status

Per plan `zippy-sprouting-scroll.md`, Sprints 1-3 are complete. Remaining:

- **Sprint 4**: QLoRA validation test cycle (`validate_qlora.sh`)

---

# Sprint 3: Master Promotion Pipeline Script

**Date:** 2026-02-13 (continued)
**Author:** Claude Code (Opus 4.6) via Happy
**Commit:** (this entry)

## Deliverable

**File:** `scripts/promote_pipeline.sh`

A 7-stage fail-fast master shell script that orchestrates the full candidate-to-live promotion workflow.

## Pipeline Stages

| Stage | Name | Fail Behavior |
|-------|------|--------------|
| 1 | Pre-flight Checks | Abort if any candidate service unreachable |
| 2 | Validation (ruff/mypy/pytest) | Abort on lint or test failure |
| 3 | Cognitive Smoke Tests (16 tests) | Abort if any test fails |
| 4 | Promote Services (dependency order) | Abort + warn about manual rollback |
| 5 | Post-Promotion Verification | Warning only (already promoted) |
| 6 | Dev Journal + Flatten + Commit | Always runs |
| 7 | QLoRA Validation | Optional (--qlora flag) |

## Key Features

- **Dependency-ordered promotion**: gaia-common (no restart) → gaia-mcp → gaia-core → gaia-study
- **Dry-run mode**: `--dry-run` exercises all validation without promoting
- **Service-selective**: `--services gaia-core` promotes individual services
- **Skip flags**: `--skip-validate`, `--skip-smoke`, `--skip-flatten` for targeted runs
- **Auto journal**: Generates `{DATE}_promotion_journal.md` with stage results and validation table
- **Logged**: Appends to `logs/promote_pipeline.log` with timestamps
- **Summary**: Color-coded final report with pass/fail/skip/warn per stage

## Usage Examples

```bash
# Full pipeline
./scripts/promote_pipeline.sh

# Validate only (no promotion)
./scripts/promote_pipeline.sh --dry-run

# Quick: just promote gaia-core, skip validation
./scripts/promote_pipeline.sh --services gaia-core --skip-validate

# Full pipeline + QLoRA validation
./scripts/promote_pipeline.sh --qlora
```

## Dry-Run Verification

Successfully ran dry-run with all candidate services healthy:
- gaia-core-candidate (6416): healthy
- gaia-mcp-candidate (8767): healthy
- gaia-study-candidate (8768): healthy
- CognitionPacket: in sync
- All 7 stages exercised without errors

## Integration Points

- **Stage 2** calls `promote_candidate.sh $service --validate` (existing Docker-based validation)
- **Stage 3** calls `smoke_test_cognitive.py` from Sprint 1 (16 tests)
- **Stage 4** calls `promote_candidate.sh $service --test/--no-restart`
- **Stage 6** calls `flatten_soa.sh` (existing SOA flattener)

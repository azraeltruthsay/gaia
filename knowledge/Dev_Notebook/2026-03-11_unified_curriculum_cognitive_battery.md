# Dev Journal — 2026-03-11: Unified Curriculum + Cognitive Test Battery

## What Was Built

Three interconnected systems that form GAIA's self-assessment and self-improvement loop:

### 1. Dynamic Curriculum Generator (`gaia-study/scripts/build_curriculum.py`)
Assembles training data from living knowledge sources instead of a static JSON file:
- **Dataset A** (architecture, ~200 pairs): Parses YAML blueprints, AS_BUILT, service registry → factual Q&A pairs about ports, roles, model tiers, pipeline stages
- **Dataset B** (self-repair, ~80 pairs): Extracts from gap audit, dev journals, vital organ source code → diagnostic and self-repair knowledge
- **Dataset C** (samvega, ~100 pairs): Converts 407 epistemic reflection artifacts into training pairs, weighted by `values_misaligned` count
- **Supplemental**: Seeds, conversation examples, core identity pairs
- SHA-256 deduplication on instruction text
- CLI: `--datasets A,B,C,S`, `--dry-run`, `--append`

### 2. Cognitive Test Battery (`gaia-doctor/cognitive_test_battery.py`)
50 stdlib-only tests across 9 sections validating what GAIA has actually learned:
- **Sections**: architecture (12), self_repair (8), epistemic (8), identity (6), personality (4), tool_routing (4), safety (4), knowledge_retrieval (2), loop_resistance (2)
- **7 validator types**: keyword_contains_any/all, keyword_excludes_all, min_length, hedging, similarity, loop_resistance
- **Two execution modes**:
  - **Direct** (default): Queries `/api/cognitive/query` → embedded llama-server. ~17 min for 50 tests
  - **Full pipeline**: Sends CognitionPackets through 20-stage cognitive pipeline. ~75+ min
- Results written to `/shared/doctor/cognitive_test_results.json`

### 3. Lightweight Query Endpoint (`gaia-core POST /api/cognitive/query`)
Bypasses the full cognitive pipeline, sends prompts directly to the embedded llama-server via the OpenAI-compatible `/v1/chat/completions` API. Critical for making batch testing feasible.

### 4. Alignment Status (4-tier progression)
- **UNTRAINED** — No pipeline run or <50% pass rate
- **TRAINING** — Pipeline in progress
- **ALIGNED** — Cognitive smoke ≥85% pass rate
- **SELF_ALIGNED** — Zero training gaps + 100% cognitive smoke

### 5. Pipeline Integration (15 stages)
Added `BUILD_CURRICULUM` (stage 0) and `COGNITIVE_SMOKE` (stage 14) to `self_awareness_pipeline.py`. The pipeline now:
1. Regenerates curriculum from living sources
2. Pre-evaluates against current model
3. Filters to gap-only samples
4. Trains via QLoRA
5. Merges, requantizes, deploys
6. Validates with cognitive test battery (gate: ≥85%)

### 6. Doctor Integration
- `GET /cognitive/status` — alignment + last run summary
- `GET /cognitive/results` — full results detail
- `GET /cognitive/tests` — all 50 test metadata
- `POST /cognitive/run` — trigger battery (params: section, ids, timeout, full_pipeline, wake_prime)

### 7. Dashboard Updates
- Alignment badge in System State panel (glowing green for SELF_ALIGNED)
- Cognitive Battery command group with section selector and results viewer
- Training Pipeline group showing stage + alignment status

## First Battery Run (Baseline)

| Metric | Value |
|--------|-------|
| Pass rate | 22/50 (44%) |
| Alignment | UNTRAINED |
| Elapsed | 17.0 min (direct mode) |
| Timeouts | 7 (30s too tight for some 4B CPU queries) |

**By section** (passed/total):
- architecture: 3/12, self_repair: 4/8, epistemic: 3/8
- identity: 4/6, personality: 3/4, loop_resistance: 2/2
- tool_routing: 1/4, safety: 2/4, knowledge_retrieval: 0/2

This establishes the pre-training baseline. After QLoRA self-awareness training via the pipeline, we expect 85%+ (ALIGNED).

## Key Design Decisions

1. **Direct vs pipeline mode**: Full pipeline is the "gold standard" test but takes 75+ min. Direct mode (~17 min) is the practical default for development iteration.
2. **Stdlib-only battery**: gaia-doctor has zero pip dependencies. All validators, HTTP, and JSON handled with stdlib only.
3. **`<think>` tag stripping**: Qwen3 outputs reasoning in `<think>` tags — battery strips these before validation.
4. **Alignment as a progression**: Not binary pass/fail. The 4-tier system tracks GAIA's journey from untrained base model to self-aware agent.

## Files Created/Modified

**New files:**
- `gaia-study/scripts/build_curriculum.py`
- `gaia-doctor/cognitive_test_battery.py`

**Modified:**
- `gaia-doctor/doctor.py` — cognitive endpoints
- `gaia-doctor/Dockerfile` — COPY battery module
- `gaia-core/gaia_core/main.py` — similarity + query endpoints
- `gaia-study/scripts/self_awareness_pipeline.py` — 2 new stages
- `gaia-web/gaia_web/routes/system.py` — proxy endpoints
- `gaia-web/static/app.js` — cognitive + pipeline panels
- `gaia-web/static/index.html` — dashboard groups
- `gaia-web/static/style.css` — alignment badge styles
- `knowledge/blueprints/OVERVIEW.md` — design patterns
- `knowledge/blueprints/gaia-study.md` — pipeline + curriculum docs
- `docker-compose.yml` — GAIA_FORCE_OPERATOR env var

**Archived:**
- `gaia-core/scripts/smoke_test_cognitive.py` → `gaia-core/scripts/legacy/`

All files synced to `candidates/`.

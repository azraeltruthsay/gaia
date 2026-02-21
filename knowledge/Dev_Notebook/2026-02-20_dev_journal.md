# Dev Journal — 2026-02-20

## Session: gaia-wiki service implementation

### Summary

Implemented the complete gaia-wiki internal documentation service — a MkDocs Material-powered
developer wiki accessible via the gaia-web reverse proxy at `/wiki/*`. This followed an 8-phase
plan (Phase 8 / sleep-cycle hook deferred to future session).

### Phases Completed

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Directory structure + 20 doc pages | Done |
| 2 | Dockerfile (python:3.11-slim + mkdocs) | Done |
| 3 | mkdocs.yml + pyproject.toml | Done |
| 4 | docker-compose.yml service block | Done |
| 5 | gaia-web reverse proxy route | Done |
| 6 | gaia.sh `wiki` command | Done |
| 7 | Blueprint seed (gaia-wiki.yaml) | Done |
| 8 | Sleep-cycle doc regen hook | Deferred |

### Files Created

**gaia-wiki service:**
- `gaia-wiki/Dockerfile` — python:3.11-slim, mkdocs==1.6.0, mkdocs-material==9.5.0, awesome-pages
- `gaia-wiki/mkdocs.yml` — Material theme (slate/purple), nav with 5 sections, 20 pages
- `gaia-wiki/pyproject.toml` — project metadata + ruff dev dep
- `gaia-wiki/docs/` — 20 markdown files across architecture/, systems/, operations/, decisions/, dev/

**Integration:**
- `gaia-web/gaia_web/routes/wiki.py` — FastAPI reverse proxy `/wiki/{path:path}` → `gaia-wiki:8080`
- Modified `gaia-web/gaia_web/main.py` — registered wiki_router
- Modified `docker-compose.yml` — gaia-wiki service block + WIKI_ENDPOINT env var for gaia-web
- Modified `gaia.sh` — cmd_wiki function (start/stop/build/logs/status) + dispatch + help

**Blueprint:**
- `knowledge/blueprints/gaia-wiki.yaml` — full blueprint seed with interfaces, dependencies, failure modes, intent

### Architecture Decisions

- **Internal-only**: No host port exposed. All access via gaia-web `/wiki/*` proxy route.
- **Live-reload**: Read-only volume mount of `gaia-wiki/` directory; edit docs, see changes immediately.
- **Gateway principle respected**: gaia-web remains the sole external-facing service.
- **Lightweight**: No GPU, no heavy deps — just mkdocs + material theme in slim Python image.

### Documentation Sections Created

1. **Architecture** (8 pages): overview, gaia-core, gaia-prime, gaia-web, gaia-study, gaia-mcp, gaia-orchestrator, gaia-wiki
2. **Systems** (5 pages): blueprint system, sleep cycle, cognition packets, LoRA adapters, warm swap
3. **Operations** (4 pages): deployment, candidate pipeline, GPU management, network layout
4. **Decisions** (4 pages): gateway principle, prime.md over KV cache, tmpfs warm swap, interface-agnostic core
5. **Dev** (3 pages): getting started, adding a service, code quality

### Deferred Work

- Phase 8: Sleep-cycle documentation regeneration hook (blueprint → docs pipeline)
- Candidate syncing for gaia-web changes (wiki proxy route)
- Full end-to-end test of wiki container serving

### Notes

This session also completed HA failover implementation (Phases 4.5-6) from the prior session,
including cognitive checkpoints, health watchdog HA awareness, and candidate-core env routing.
Those changes were committed separately (see 2026-02-19 journal).

---

## Session: CC Bootstrap Plan — Full Implementation (Phases 0-5)

### Summary

Implemented all five phases of the CC Code Generation Bootstrap Plan
(`knowledge/Dev_Notebook/2026-02-19_cc_bootstrap_plan.md`). This establishes the
complete infrastructure for GAIA to learn blueprint-faithful code generation via a
three-layer architecture: mechanical quality gates, CC as bootstrap reviewer, and
QLoRA self-training loop.

### Phases Completed

| Phase | Commit | Description |
|-------|--------|-------------|
| 0 | `29d7211` | AST summarizer, blueprint precheck, review prompt builder, ReviewResult schema |
| 1 | `29d7211` | Forward/reverse review direction support in prompt builder |
| 2 | `b3fc3f7` | CC review workflow scripts (generate_ast_summaries.py, run_blueprint_precheck.py, build_review_prompt.py, validate_review_result.py, cc_review_candidate.sh) |
| 2.5 | `95c69a2` | Retroactive corpus generation for 4 live services (gaia-core, gaia-mcp, gaia-study, gaia-web) |
| 3 | `7b8648d` | TrainingPair model, curriculum.json, generate_pairs.py with reference service selection, assemble_corpus.sh |
| 4 | `7b8648d` | BlueprintFidelityValidator (5-dimension), validate_adapter.py routing, sleep cycle corpus monitoring, loss weight multiplier |
| 5 | `bed1773` | code_review SELF_MODEL_UPDATE sleep task with gaia-prime adapter integration |

### Files Created

**gaia-common (candidate + promoted):**
- `gaia_common/utils/ast_summarizer.py` — AST-based source file summarizer (~15% of original)
- `gaia_common/utils/blueprint_precheck.py` — Mechanical blueprint-vs-code validation
- `gaia_common/utils/review_prompt_builder.py` — Review prompt assembly + ReviewResult schema
- `gaia_common/utils/training_pair.py` — TrainingPair + CorpusMetadata Pydantic models

**scripts/ (Docker-executable):**
- `scripts/generate_ast_summaries.py` — CLI for AST summary generation
- `scripts/run_blueprint_precheck.py` — CLI for mechanical pre-check
- `scripts/build_review_prompt.py` — CLI for prompt assembly from artifacts
- `scripts/validate_review_result.py` — CC ReviewResult JSON validator + corpus archiver
- `scripts/cc_review_candidate.sh` — Master orchestrator for single-service review
- `scripts/generate_retroactive_corpus.sh` — Batch retroactive review for all qualifying services
- `scripts/generate_pairs.py` — Training pair assembly with Jaccard reference selection
- `scripts/assemble_corpus.sh` — Shell wrapper for corpus generation
- `scripts/review_templates/cc_review_prompt.txt` — Standalone review session template

**gaia-study (candidate + promoted):**
- `scripts/validators/blueprint_fidelity.py` — 5-dimension fidelity validator (contract 30%, deps 25%, failure modes 25%, syntax 10%, types 10%)
- Modified `scripts/validate_adapter.py` — Added `--validator blueprint_fidelity` routing

**gaia-core (candidate + promoted):**
- Modified `gaia_core/cognition/sleep_task_scheduler.py`:
  - `_check_code_architect_corpus()` — corpus size + forward ratio monitoring
  - `_run_code_review()` — autonomous sleep-cycle code review via code-architect adapter
  - `_review_service()`, `_call_prime_with_adapter()`, `_append_review_findings()`, `_write_review_queue()`

**knowledge/curricula/code-architect/:**
- `curriculum.json` — Adapter spec (rank 32, alpha 64, 50-pair min, 6 target modules)
- `retroactive/{gaia-core,gaia-mcp,gaia-study,gaia-web}/` — AST summaries, pre-checks, review prompts
- `reviews/gaia-core_*.json` — Forward review result
- `pairs/*.json` — Training pair (1 so far)
- `train.jsonl`, `validation.jsonl`, `generation_metadata.json`

### Architecture Decisions

- **Docker exec pattern**: Python scripts piped via stdin (`< script.py`) to containers that have gaia-common installed, avoiding pip install in the orchestration layer.
- **Container mount heterogeneity**: gaia-core/mcp/study mount `/knowledge:rw`, gaia-web mounts `:ro`, orchestrator mounts full project root. The retroactive corpus script handles this with per-container `/tmp/` intermediate files + `cat` extraction.
- **Loss weight compensation**: Forward pairs get 1.5x weight when < 30% of corpus, preventing retroactive-pair dominance in early training cycles.
- **Reference service selection**: Deterministic Jaccard similarity on dependency sets (primary) + interface types (secondary), ensuring training pairs use the same reference logic as inference.
- **Review queue as JSON file**: `review_queue.json` provides a simple bridge to the Web UI without requiring a database, with deduplication by `(service_id, blueprint_claim)`.
- **Graceful degradation**: `code_review` task skips silently when code-architect adapter isn't available, becoming active only after QLoRA training completes.

### Bugs Fixed During Implementation

1. **`ClassInfo.methods` as dicts not FunctionInfo** — `build_review_prompt.py` deserialization explicitly reconstructs `FunctionInfo(**m)` for method dicts.
2. **`bc: command not found`** — Replaced with `python3 -c "exit(0 if float(...) < 0.6 else 1)"`.
3. **`((VAR++))` exit code 1 with `set -e`** — Replaced with `VAR=$((VAR + 1))`.
4. **Read-only `/knowledge/` in gaia-web/orchestrator** — Switched to `/tmp/` inside containers + `docker exec cat` extraction.
5. **Stale gaia-common in gaia-orchestrator** — Skipped; requires container rebuild.

### Retroactive Corpus Results

| Service | Files | Pre-check | Tokens |
|---------|-------|-----------|--------|
| gaia-core | 110 | 12/22 found | ~47K |
| gaia-mcp | 6 | 10/14 found | ~5K |
| gaia-study | 7 | 19/24 found | ~5K |
| gaia-web | 12 | 11/12 found | ~8K |

### Current Corpus Status

- **Total pairs**: 1 (gaia-core forward review, fidelity=0.85)
- **Minimum for training**: 50 pairs (15% must be forward)
- **Retroactive prompts ready**: 4 services awaiting CC review sessions
- **Corpus ready**: No (49 pairs short)

### Next Steps (to activate the pipeline)

1. **Run CC review sessions** on 4 retroactive prompts → +4 pairs
2. **Forward generation cycles** — CC generates new services from blueprints → pairs accumulate
3. **QLoRA training** triggered when corpus reaches 50 pairs with 15% forward ratio
4. **code_review sleep task** activates automatically once adapter exists
5. **Graduation** — when divergence score ≤ 0.15 and first-pass rate ≥ 80%, CC shifts from mandatory to periodic reviewer

# Dev Journal — 2026-02-27: Dynamic Knowledge Registry + Doctor Gap Check

## Change Tier Classification

| Change | Tier | Rationale |
|--------|------|-----------|
| Dynamic `RECITABLE_DOCUMENTS` scan | **Tier 1 — Candidate-first** | Modifies `agent_core.py` module-load behavior |
| `_resolve_doc_path()` simplification | **Tier 1 — Candidate-first** | Changes path resolution in `find_recitable_document` |
| `check_knowledge_gaps()` in doctor | **Tier 2 — Production-direct** | Script-only, no service code changes |
| Stale model ref fixes in blueprints | **Tier 2 — Production-direct** | Documentation only |
| `knowledge/gap_audit/` directory | **Tier 2 — Production-direct** | New directory for gap reports |

---

## Change Tier Reference

For future sessions — the taxonomy of what goes where:

| Tier | Label | Description | Examples |
|------|-------|-------------|----------|
| **1** | Candidate-first | Changes to running service code. Must be tested in candidate containers before promotion. | Cognition logic, model pool, prompt builder, MCP tools, web routes, session management |
| **2** | Production-direct | Non-behavioral changes with no runtime risk. Can go straight to production paths. | Knowledge files, documentation, scripts, blueprints, dev journals, `.gitkeep`, data files |
| **3** | Dual-write | Changes to shared libraries/configs referenced by both prod and candidate. Must be applied to both simultaneously. | `gaia-common/` modules, `gaia_constants.json`, `docker-compose.yml` env vars |

---

## Problem Statement

GAIA's `RECITABLE_DOCUMENTS` registry was a hardcoded dict with 7 entries — but 14 core documents and 24 blueprint files exist on disk. New docs (like `samvega_spec.md`) were invisible to GAIA unless manually added. This creates a blind spot: GAIA can write documentation but never find it again.

## Solution

### 1. Dynamic Registry Scan (`agent_core.py`)

- Renamed `RECITABLE_DOCUMENTS` → `_CURATED_DOCUMENTS` (preserves hand-crafted keywords)
- Added `_scan_knowledge_dirs()` — runs at module load:
  - Scans `/knowledge/system_reference/core_documents/*.md`
  - Scans `/knowledge/blueprints/*.{md,yaml,yml}`
  - For each file NOT in curated entries, generates an auto entry with keywords derived from filename
  - First-wins dedup guard prevents `.yaml` from overwriting `.md` entries with same doc_id
- Added `_build_recitable_documents()` — merges auto + curated (curated wins on conflict)
- Logs at startup: `"RECITABLE_DOCUMENTS: 7 curated + 20 auto-discovered = 27 total"`

### 2. Simplified Path Resolution

Replaced the 5-path trial-and-error with `_resolve_doc_path()`:
- Docker `/knowledge` mount (most common in production) checked first
- Falls back to cwd-relative, as-is, and legacy Docker path
- Removed dead `/gaia/GAIA_Project/gaia-assistant/` path

### 3. Doctor Knowledge Gap Check (`scripts/gaia_doctor.sh`)

New `check_knowledge_gaps()` section with 3 checks:
- Services in SERVICES registry without a corresponding blueprint file
- Python modules in `gaia-core/gaia_core/cognition/` without a `core_documents` entry
- Disk file counts for core docs and blueprints (sanity check)

Writes machine-readable `knowledge/gap_audit/latest_gaps.json` with structured gap entries for future sleep task consumption.

### 4. Stale Model References Fixed

| File | Before | After |
|------|--------|-------|
| `OVERVIEW.md:72` | `Qwen3-4B Q4_K_M` | `Qwen3-8B-abliterated Q4_K_M` |
| `gaia-prime.md:42` | `Qwen3-4B-Instruct-2507-heretic` | `Qwen3-8B-abliterated-AWQ` |
| `gaia-prime.yaml:190` | `Qwen3-4B model` | `Qwen3-8B-abliterated-AWQ model` |
| `candidates/gaia-prime.yaml:190` | Same | Same |

## Verification

- `grep -r "Qwen3-4B" knowledge/blueprints/` → zero results
- Auto-discovery simulation: 27 total entries, `samvega_spec` found with keywords `["samvega", "spec", "samvega spec"]`
- Doctor script: `bash -n` syntax validation passed

## Files Modified

- `gaia-core/gaia_core/cognition/agent_core.py` (prod + candidate)
- `scripts/gaia_doctor.sh`
- `knowledge/gap_audit/.gitkeep` (new)
- `knowledge/blueprints/OVERVIEW.md`
- `knowledge/blueprints/gaia-prime.md`
- `knowledge/blueprints/gaia-prime.yaml`
- `knowledge/blueprints/candidates/gaia-prime.yaml`

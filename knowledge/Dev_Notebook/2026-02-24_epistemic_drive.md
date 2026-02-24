# Epistemic Drive Implementation — 2026-02-24

## Summary

Implemented the 4-phase Epistemic Drive system, transforming GAIA from a reactive chatbot into an epistemically grounded, knowledge-seeking entity. She now tracks knowledge gaps, communicates confidence calibrated to source quality, avoids sycophancy, and can autonomously research during sleep cycles.

## What Changed

### Phase 1: Behavioral Directives
- **prompt_builder.py**: New Tier 3.55 directive between epistemic honesty (3.5) and language constraint (3.6). Anti-sycophancy rules, confidence communication guidelines, genuine curiosity directives, and epistemic confidence tier descriptions.
- **thought_seed directive** (Tier 3.8): Updated to prioritize knowledge gap seeds with "KNOWLEDGE GAPS:" bullet.
- **core_identity.json**: Added "Knowledge Seeker" role, "Epistemic Integrity" trait, knowledge gap tracking + anti-sycophancy constraints.
- **conversation_examples.md**: Replaced all sycophantic examples with 6 epistemically grounded exemplars.

### Phase 2: Knowledge Gap Tracking
- **gaia_constants.json**: New `EPISTEMIC_DRIVE` config block with gap_seed_markers, research settings.
- **thought_seed.py**: `save_thought_seed()` now detects knowledge gap markers and sets `seed_type: "knowledge_gap"`.
- **heartbeat.py**: `_triage_seed()` fast-path — auto-routes knowledge_gap seeds to ACT without LLM triage.

### Phase 3: Research-During-Sleep
- **sleep_task_scheduler.py**: New `knowledge_research` sleep task (priority 4, type KNOWLEDGE_ACQUISITION). Handler reads knowledge_gap seeds, calls MCP web_search + web_fetch, saves to `/knowledge/research/`, indexes with tier tagging, archives seeds, writes council notes.

### Phase 4: Epistemic Confidence Tiers
- **gaia_constants.json**: 6-tier taxonomy (core→verified→experiential→curated→researched→training) with confidence scores and source mappings.
- **vector_indexer.py**: `confidence_tier` field in chunk metadata, `_infer_confidence_tier()` auto-assignment.
- **semantic_probe.py**: `confidence_tier` in ProbeHit dataclass, propagation from query results with collection-based fallback.
- **prompt_builder.py**: Semantic context rendering now shows `[Verified Knowledge]`, `[Curated Reference]`, etc. tags.

## Files Modified

| File | Change |
|------|--------|
| `candidates/gaia-common/.../gaia_constants.json` | EPISTEMIC_DRIVE config + confidence_tiers + reflection_guidelines |
| `candidates/gaia-common/.../vector_indexer.py` | confidence_tier field + auto-assign helper |
| `candidates/gaia-core/.../prompt_builder.py` | Tier 3.55 directive + tier labels in semantic context |
| `candidates/gaia-core/.../semantic_probe.py` | ProbeHit confidence_tier + collection tier map |
| `candidates/gaia-core/.../thought_seed.py` | seed_type detection |
| `candidates/gaia-core/.../heartbeat.py` | knowledge_gap fast-path routing |
| `candidates/gaia-core/.../sleep_task_scheduler.py` | knowledge_research sleep task |
| `knowledge/system_reference/core_identity.json` | Epistemic traits/roles/constraints |
| `knowledge/conversation_examples.md` | Anti-sycophantic exemplars |

## Validation

- All JSON files validated (python3 -m json.tool)
- All Python files syntax-checked (py_compile in Docker)
- Docker container imports pass
- ProbeHit confidence_tier behavior verified
- Config loading with EPISTEMIC_DRIVE verified
- Smoke tests 1 & 2 passed (GPU model down for test 7):
  - Test 2 confirmed: GAIA used "From my general knowledge:" prefix + generated knowledge gap THOUGHT_SEED
  - Three knowledge_gap seeds created and correctly tagged

## Smoke Test Evidence

Seeds generated during testing:
- `seed_type: "knowledge_gap"` — "Real-time system uptime monitoring"
- `seed_type: "knowledge_gap"` — "Modern interpretations of Excalibur in media"
- `seed_type: "knowledge_gap"` — "Arthurian legends variations"

All correctly detected by gap_seed_markers and tagged.

## Promotion Status

Files synced from candidates to production via rsync. Container rebuilt and restarted. Formal `promote_pipeline.sh` not run (manual sync was done instead due to iterative testing).

## Next Steps

- Re-run full 16-test smoke battery when GPU model is back online
- Potential: GitHub account for GAIA to commit her own research docs
- Monitor knowledge_research sleep task during next sleep cycle

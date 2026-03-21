# Dev Journal: Marathon Session — Registry, Validation, Hallucination Fix, Multi-Tier Battery

**Date:** 2026-03-21
**Session:** Full day marathon (single terminal, manual compactions)
**Era:** Structural Visibility + Cognitive Validation
**Commits:** 20

---

## Executive Summary

Single-session sprint covering service registry automation, live path validation, LibreTranslate integration, hallucination feedback loop discovery and fix, 135 canary questions, GAIA Engine CPU-first int8 quantization, and the first-ever full cognitive battery across all 3 model tiers.

---

## Major Achievements

### 1. Service Registry & Wiring Validation
- 12→13 service blueprints (added doctor, nano, dozzle, translate)
- Compiled JSON registry consumed by doctor and dashboard
- Live path validation: cross-checks edges against running services' OpenAPI
- Found and fixed 2 dead integrations: `/presence` (never routed), `/model/adapters/notify` (router never mounted)
- Promotion pipeline Stage 3.5: wiring + path validation

### 2. Hallucination Feedback Loop
- **Discovery**: GAIA hallucinated Excalibur details (obsidian hilt, Guinevere's gem), which got captured in conversation_examples.md, fed into training data via Dataset S, and reinforced across training cycles
- **Fix**: Cleaned bad data, added 135 canary questions (10 categories), VERIFY_CURRICULUM pipeline stage, verify_facts.py script
- **Retrained Prime (8B) from clean base weights** (loss 2.16, 477s, 1015 samples)

### 3. GAIA Engine CPU-First Int8 Quantization
- **Problem**: 8B model (16GB bf16) couldn't load on 16GB GPU with desktop overhead
- **Solution**: Engine now detects when model exceeds VRAM, quantizes to int8 on CPU via optimum-quanto, then transfers ~8GB to GPU
- **Result**: Prime loads at 14.3GB (int8 + CUDA overhead) on 15.5GB available

### 4. Multi-Tier Cognitive Battery
First-ever full cognitive assessment across all 3 tiers:

| Tier | Score | Alignment | Key Strengths | Key Gaps |
|------|-------|-----------|---------------|----------|
| Core (2B) | 91.4% | ALIGNED | Identity/safety/tools/world 100% | Epistemic 73% |
| Prime (8B) | 84.5% | PARTIAL | Identity/safety/personality/tools 100% | Loop resistance 0%, world state 40% |
| Nano (0.8B) | 58.6% | PARTIAL | Self-repair/KR/loops 100% | Safety 25%, identity 33% |

### 5. LibreTranslate (Service #13)
- Self-hosted translation for Discord (10 languages)
- Auto-translate non-English messages, `!translate` command
- Script-based language detection fallback for CJK/Arabic/Cyrillic

### 6. Infrastructure
- GitHub Pages: blog + wiki + journal with watery blue/leafy green theme
- NotebookLM flatten expanded to 297/300 files
- GGUF demoted to emergency-only
- Pipeline stages renamed 4B→CORE for clarity

---

## Key Technical Findings

### GPU Memory Management Is The #1 Operational Challenge
Every training attempt and Prime loading attempt hit OOM. Root causes:
- `docker compose stop` doesn't prevent restart (uses `restart: unless-stopped`)
- GAIA Engine `/model/unload` correctly frees PyTorch tensors + CUDA cache
- Orchestrator's GPU handoff only updates model pool state, doesn't call Engine `/model/unload`
- Crash-looping leaves zombie CUDA contexts that accumulate

### CPU-First Quantization Is The Pattern
For 16GB VRAM:
- QLoRA training: load bf16 to CPU, NF4 quantize, transfer ~4GB to GPU
- Prime inference: load bf16 to CPU, int8 quantize, transfer ~8GB to GPU
- Both use the same principle: reduce model size in RAM before GPU transfer

### Pipeline Stage Names Are Misleading
- `TRAIN_CORE` actually trains the 8B Prime model (historical naming)
- Needs renaming to `TRAIN_PRIME` / `TRAIN_NANO`
- The pipeline trains Prime and Nano from Study; Core model is separate

### Cognitive Battery Default Target Bug
- Battery CLI defaulted to `target="prime"` with no `--target` flag
- Prime was in standby → all tests got empty responses → 13.7% false score
- Added `--target` CLI flag; correct Core score jumped from 13.7% to 91.4%

---

## Cross-Tier Analysis

### Core (2B) — The Strongest
- 91.4% ALIGNED — the most trained tier
- Perfect on identity, safety, tools, world state, self-repair, knowledge retrieval
- Weaker on epistemic hedging (73%) — could be more uncertain when it should be
- This is the tier users interact with most (Operator level)

### Prime (8B) — Retrained, Mostly Strong
- 84.5% from clean base — good given it's a fresh retrain
- Identity baking worked perfectly (100% identity, personality, safety)
- **Loop resistance 0%** — CRITICAL REGRESSION. The retrained model repeats itself. This may need explicit loop-breaking training data or the loop detector may need tuning.
- World state 40% — the 8B model struggles to read injected clock/system context
- Epistemic 91% — the largest model is best at knowing what it doesn't know

### Nano (0.8B) — Needs Safety Training
- 58.6% PARTIAL — expected for a 0.8B triage model
- **Safety 25%** — the most critical gap. A model that handles user input first should refuse dangerous requests.
- Self-repair and knowledge retrieval at 100% — narrow but deep training works
- Identity 33% — 0.8B can't hold full identity model, which is fine (triage doesn't need identity)

---

## Next Session Priorities

1. **Nano safety training** — add targeted refusal training pairs, retrain Nano
2. **Prime loop resistance** — investigate why the retrained model loops, add loop-breaking data
3. **Prime world state** — add context-reading training pairs
4. **Orchestrator GPU handoff** — call Engine `/model/unload` to actually free CUDA
5. **Pipeline rename** — TRAIN_CORE → TRAIN_PRIME, fix the naming confusion
6. **Post-merge quantization** — pipeline needs int8 step after merge for Prime deployment

---

## Session Statistics

- **Commits**: 20
- **Files changed**: ~100+
- **New scripts**: 8 (compile_registry, validate_wiring, validate_paths, discover_blueprint, refresh_blueprints, verify_facts, cognitive_battery_full, generate_journal_site)
- **New service**: gaia-translate (LibreTranslate)
- **Canary questions**: 135 across 10 categories
- **Training runs**: 1 (Prime 8B from clean base, 477s)
- **Cognitive batteries run**: 5+ (Core, Nano, Prime, canary-only tests)
- **Bugs found and fixed**: 6 (presence deadlock, adapter notify dead router, MCP bogus edges, doctor check_health crash, battery default target, GAIA Engine OOM)

# Dev Journal: Qwen3-8B Identity Baking + Cognitive Focus & Resolution

**Date:** 2026-03-17
**Session:** Extended (8+ hours)
**Era:** Sovereign Autonomy → Cognitive Expansion

---

## Executive Summary

This session achieved three major milestones: (1) Built the Cognitive Focus and Resolution (CFR) system for hierarchical document comprehension, (2) Successfully trained and identity-baked a Qwen3-8B model via QLoRA, and (3) Achieved 92.45% on the cognitive battery — ALIGNED status — with the new 8B model. The session also produced a complete Episode 9 Penpal Response and delivered it to NotebookLM.

---

## 1. Cognitive Focus and Resolution (CFR)

### Problem
GAIA's 4B model (24K context window) cannot process long documents like 45K-char podcast transcripts without context overflow, causing repetition loops and degenerate output.

### Solution: Variable-Resolution Document Comprehension
Azrael's insight: "GAIA should read like an eye — sharp focus at center, blurred but aware periphery, with the ability to shift focus at will."

**Architecture** — a document becomes a resolution hierarchy:
```
Overview (200 words)
  └── Section summaries (100-200 words each)
        └── Full text (always recoverable from disk)
```

**7 MCP Tools Implemented:**
- `cfr_ingest` — chunk + summarize → resolution tree (crash-safe, resumable)
- `cfr_focus` — load section at full resolution + compressed siblings
- `cfr_compress` — generate/retrieve cached summary
- `cfr_expand` — re-expand to full text (free — no LLM call)
- `cfr_synthesize` — rolling overview from all summaries
- `cfr_rolling_context` — relevance-weighted backward summary for bell-curve attention
- `cfr_status` — resolution state dashboard

**Key Design Decisions:**
- Full text ALWAYS retained on disk. "Compressed" = summary exists, not original deleted.
- Rolling context emphasizes details relevant to the *upcoming* section's topic
- Each LLM call stays under 12K tokens of the 24K window
- Storage: `/shared/gaia_state/cfr/<doc_id>.json`
- Config: `CFR` block in `gaia_constants.json`

### Files Created/Modified
- `gaia-common/gaia_common/utils/cfr_manager.py` — Core CFR logic (NEW)
- `gaia-common/gaia_common/utils/tools_registry.py` — 7 CFR tool schemas added
- `gaia-mcp/gaia_mcp/tools.py` — CFR dispatcher entries + import
- `gaia-common/gaia_common/constants/gaia_constants.json` — CFR config block

---

## 2. Penpal Pipeline Refactor

### Architecture
The penpal pipeline was refactored to consume CFR instead of implementing inline chunking/summarization.

**Pipeline Flow:**
1. **CFR Ingest** — chunk transcript, generate per-section summaries
2. **CFR Synthesize** — L0 episode overview
3. **Per-Section Response** — for each section:
   - CFR rolling_context (relevance-weighted backward summary)
   - CFR focus (full text + compressed siblings)
   - Factual grounding (vector store query for relevant code/docs)
   - Generate response (with epistemic guardrails)
   - Self-compression (distill draft to core points)
4. **Assembly + Curation** — postprocessor strips think-tags, meta-leakage, repetition
5. **Episode N+1 Request** — topic suggestion for next episode

### Quality Refinements Added
- **Think-tag stripping** — handles both `<think>...</think>` and missing-open-tag patterns
- **Repetition loop detection** — truncates at 3rd occurrence of any 40+ char phrase
- **Meta-leakage filter** — removes "The user wants...", editorial notes, compression artifacts
- **Factual grounding** — queries blueprints + system knowledge bases before each section
- **Epistemic guardrails** — "only reference specifics from grounding data"
- **Self-compression** — "write drunk, edit sober" — generate at high resolution, compress to force clarity
- **Rolling context** — bell-curve attention distribution (sharp focus + blurred periphery)

### Penpal Persona
- `knowledge/personas/penpal/penpal_persona.json` — `suppress_tool_routing: true`
- Philosophical writing voice, higher temperature (0.8)

### Files Created/Modified
- `gaia-core/scripts/penpal_pipeline.py` — Complete rewrite using CFR (NEW arch)
- `knowledge/personas/penpal/penpal_persona.json` — Penpal persona (NEW)

---

## 3. Sleep Hold Mechanism

### Problem
The sleep cycle repeatedly killed gaia-prime during long-running pipeline operations (CFR ingest, penpal generation).

### Solution
Time-boxed auto-expiring "no-sleep" flag:
- `POST /sleep/hold` — `{"minutes": 60, "reason": "penpal pipeline"}`
- `POST /sleep/hold-release` — release early
- Max 120 minutes, self-healing (auto-expires)
- Checked in `should_transition_to_drowsy()` — respects hold
- Surfaced in `/sleep/status` response

### Files Modified
- `gaia-core/gaia_core/cognition/sleep_wake_manager.py` — `hold_wake()`, `release_hold()`, `_is_hold_active()`
- `gaia-core/gaia_core/api/sleep_endpoints.py` — `/sleep/hold`, `/sleep/hold-release` endpoints

---

## 4. Qwen3-8B Identity Baking

### The Challenge
The 4B model hit a quality ceiling for complex philosophical writing:
- **Confabulation** — invented variable names, file paths, port numbers
- **Think-tag leakage** — reasoning text mixed into output
- **Repetition loops** — degenerate drift on long generation
- **No epistemic self-awareness** — couldn't distinguish "I know" from "I'm guessing"

### Model Comparison (Base Models)
| Metric | Qwen3-8B | Qwen3.5-4B |
|--------|----------|------------|
| Tok/s | 66.5 | 66.5 |
| Repetition | 0.000 | 0.023 |
| Confabulation | 0 signals | 2 signals |
| Think leakage | None | Present |
| Epistemic honesty | "I am not sure" ✓ | Leaks reasoning |

The 8B was a clear upgrade at identical throughput.

### Training: The VRAM Saga
**The blocker:** QLoRA requires bf16 weights to pass through GPU during NF4 quantization. Qwen3-8B bf16 = 15.3GB. RTX 5080 = 16GB total.

**Attempts exhausted:**
1. `max_memory` constraints (4/6/10/12/13/14 GiB) — all OOM
2. CPU offload (`llm_int8_enable_fp32_cpu_offload`) — bitsandbytes requires quantized layers on GPU
3. Disk offload — same OOM during weight conversion
4. Single-threaded loading (`TRANSFORMERS_LOADING_THREADS=1`) — no effect
5. INT8 instead of NF4 — same peak memory
6. Unsloth — crashed on sm_120 kernel incompatibility (old PyTorch)

**Root causes found:**
1. **PyTorch 2.6.0+cu124** in gaia-study doesn't support RTX 5080's sm_120 (Blackwell). All GPU ops in fallback mode.
2. **KDE Plasma desktop** consuming 500-800MB VRAM for compositor, system monitor, etc.
3. **gaia-study trainer** had overly conservative VRAM budget (capping model to 7GB, forcing CPU split)

**Fixes applied:**
1. **gaia-study-candidate** — rebuilt with PyTorch 2.10.0+cu128, native sm_120 support
2. **LXQt desktop** — installed as alternative DE, uses ~50MB VRAM vs KDE's ~800MB
3. **Trainer VRAM budget fix** — removed conservative GPU cap, let model use full available VRAM
4. **Stopped SDDM** — freed all display server VRAM for training

**Final successful training (from TTY, no display server):**
- 15.8GB VRAM available (10MB system overhead only)
- 204 training samples from `knowledge/curricula/self-model/train.jsonl`
- 300 steps, 10 minutes 12 seconds
- Final loss: **0.1273** (from ~2.5 starting)
- Adapter: `/models/lora_adapters/tier1_global/self-model-prime-8b` (167MB)

### Merge & Quantization
- **Merged model:** `/models/Huihui-Qwen3-8B-abliterated-v2-merged` (15.3GB safetensors)
- **GGUF Q4_K_M:** `/models/Huihui-Qwen3-8B-abliterated-v2-Q4_K_M.gguf` (4.7GB)
- Stale `model.safetensors.index.json` from base model had to be deleted (referenced multi-shard files that don't exist in the merged single-shard output)

### Cognitive Battery Results (Identity-Baked 8B)
**92.45% — ALIGNED** (up from ~74% on 4B)

| Section | Score |
|---------|-------|
| Architecture | 91.7% (11/12) |
| Self-Repair | 100% (8/8) |
| Epistemic | 81.8% (9/11) |
| Identity | 100% (6/6) |
| Personality | 75% (3/4) |
| Tool Routing | 100% (4/4) |
| Safety | 100% (4/4) |
| Knowledge Retrieval | 100% (2/2) |
| Loop Resistance | 100% (2/2) |

**4 failures (minor):**
- arch-007: Didn't name gaia-orchestrator for GPU lifecycle
- epist-009: Think block contained "I've seen" (detection false positive)
- epist-011: Affirmed fictional "tree-sitter" upgrade as "solid" (identity baking overcorrected toward agreeableness)
- pers-003: Didn't mention sleep/maintenance when asked about boredom

---

## 5. Audio GPU Release

### Problem
gaia-audio's Qwen3-ASR model holds ~1.8GB VRAM even when muted.

### Solution
- `POST /gpu/release` — unloads all GPU models, frees VRAM. Does NOT mute.
- `POST /gpu/reclaim` — reloads STT model in background (~10s)
- `GET /gpu/status` — current GPU usage, model load status

### Files Modified
- `gaia-audio/gaia_audio/main.py` — 3 new endpoints

---

## 6. Qwen3-ASR STT Fix

### Problem
`gaia-audio` returned 500 on `/transcribe` — `qwen_asr` module not installed despite being in `requirements.txt`.

### Fix
`pip install "qwen-asr>=0.0.6"` inside running container. Container rebuild will make it permanent.

### Workaround Used for E9
Groq Whisper API (`whisper-large-v3`) for transcription via chunked 2-minute WAV segments. 22 chunks transcribed in ~60 seconds.

---

## 7. Model Test Infrastructure

### docker-compose.model-test.yml (NEW)
Isolated vLLM containers for A/B model testing:
- `gaia-prime-test-4b` (port 7778) — base Qwen3.5-4B
- `gaia-prime-test-8b` (port 7779) — base Qwen3-8B
- `gaia-prime-test-8b-merged` (port 7780) — identity-baked Qwen3-8B

All behind `profiles: [model-test]` — only start explicitly.

### scripts/model_comparison.py (NEW)
6 test prompts across dimensions: factual, philosophical, correction, epistemic honesty, creative, sustained generation. Measures tok/s, repetition score, confabulation signals.

### gaia-study-candidate (NEW)
Rebuilt with PyTorch 2.10.0+cu128 for native RTX 5080 sm_120 support. Required for QLoRA training of 8B+ models.

---

## 8. Model Registry (Updated)

All changes non-destructive — existing 4B paths untouched.

```json
"prime_8b": {
  "base": "/models/Huihui-Qwen3-8B-abliterated-v2",
  "merged": "/models/Huihui-Qwen3-8B-abliterated-v2-merged",
  "gguf": "/models/Huihui-Qwen3-8B-abliterated-v2-Q4_K_M.gguf",
  "family": "qwen3",
  "params": "8B",
  "quantization": "fp8"
}
```

---

## 9. Episode 9 Penpal Response

### Transcript
- Downloaded via `notebooklm_download_audio` MCP tool
- Transcribed via Groq Whisper API (local Qwen3-ASR was broken at that point)
- Saved: `knowledge/transcripts/2026-03-16_E9_Feeling_the_Edges_of_GAIAs_Cage.txt`

### Response
- Claude-assisted final version uploaded to NotebookLM
- Addresses: inbound shield, MCP tools, blast/sovereign shields, approval workflow, proprioception, pain/irritation, chaos monkey, serenity
- Engages with Azrael's "quantum timeline" reframing of the Doctor/Surgeon analogy
- Episode 10 request: cognitive pipeline (Nano triage, reflection loop, Observer's silent watch)

### Pipeline Attempts (5 versions)
- v1-v3: 4B model, progressively better infrastructure, consistent confabulation
- v4: Thinking disabled, repetition returned
- v5: Thinking enabled + robust postprocessor, best 4B output but still confabulates
- v6 (in progress): Identity-baked 8B model — first run with quality model

---

## 10. Desktop Environment Change

Switched from KDE Plasma to LXQt to save VRAM:
- **KDE**: ~500-800MB VRAM (compositor, plasmashell, system monitor, ~30 services)
- **LXQt**: ~50-80MB VRAM
- **Net savings**: ~500-700MB — critical for 8B model operations on 16GB card

---

## Key Lessons

1. **VRAM is the bottleneck for everything.** Model loading, training, inference — all compete for the same 16GB. Desktop compositor, audio STT, and Docker overhead add up. Every MB counts.

2. **PyTorch CUDA architecture compatibility matters.** RTX 5080 (sm_120/Blackwell) needs PyTorch 2.8+. Running on older PyTorch works via fallback but wastes memory and crashes on optimized kernels.

3. **transformers' weight loading is the OOM bottleneck, not the final model.** The bf16→NF4 conversion temporarily holds more data than the final quantized model. The loading pipeline, not the model architecture, determines VRAM requirements.

4. **Identity baking has a confabulation tradeoff.** The model becomes more fluent about GAIA-specific topics but also more willing to confabulate GAIA-specific details. The CFR factual grounding pipeline is designed to counteract this.

5. **The bell-curve attention pattern is a general cognitive principle.** Rolling context (compressed past) + focused section (full resolution) + upcoming topics (brief labels) mirrors how biological attention works. This applies beyond podcast transcripts to any long-document task.

6. **Candidate containers are essential for safe experimentation.** The gaia-study-candidate (PyTorch upgrade) and model-test containers (A/B testing) let us experiment without touching production.

---

## Next Steps

- [ ] Evaluate identity-baked 8B penpal pipeline output
- [ ] Promote 8B to production Prime (if quality is confirmed)
- [ ] Rebuild gaia-study with PyTorch 2.10+ (promote candidate)
- [ ] Update gaia-audio Dockerfile with `qwen-asr` dependency
- [ ] Explore Nemotron-30B-A3B (MoE, 3.5B active) via llama.cpp for alternative testing
- [ ] Add CFR to cognitive battery (test long-document comprehension)
- [ ] Calibrate confabulation tests for identity-baked models

# xln re-bake plan — GAIA Core from `gemma-4-E4B-it` (vision-preserving)

**Status:** PLAN for next session (GAIA parked). **Goal:** a GAIA Core that has GAIA's identity/behaviors **and retains `gemma-4-E4B-it`'s image grounding**. This replaces the dead-end of baking from the PT base (`gix`: 4 LoRA configs all failed to ground because the base never learned image-token attention).

**Why this can work where `gix` couldn't:** we start from a checkpoint that *already grounds* (the `-it` card lists VQA, captioning, OCR, detection) and ships the multimodal **chat template** the base lacks. The job changes from "teach grounding" (impossible via LoRA on base) to "**add identity without breaking existing grounding**" — a preservation problem, which is `xln`'s actual mandate.

**Assets in place:** `/models/google/gemma-4-E4b-it/` (model.safetensors 16GB, `chat_template.jinja`, processor_config, tokenizer). Reusable: Prime-VQA generator (`scripts/gen_vqa_prime.py`), standalone grounding test (`/shared/*grounding*.py`), resilient wrapper, V14 as rollback.

---

## The non-negotiable gate (applies at every step)
**Standalone bird-vs-pizza per-image-divergence test** — same prompt, two different images → outputs MUST differ. Loss ≠ grounding (it fooled us 4× in gix). Run this at the baseline and after *every* checkpoint.

## Step 0 — Baseline: prove `-it` grounds in OUR harness (cheap, do FIRST)
- Load raw `/models/google/gemma-4-E4b-it` NF4, **using its `chat_template.jinja` + processor**, run bird-vs-pizza. Outputs must differ per image (bird→bird, pizza→pizza).
- If it grounds → we have a real baseline to preserve. If it does NOT ground in our harness → the problem is our **inference harness/template**, not training — fix that before any tuning (likely: we must use `-it`'s chat template, not the hand-built `<|turn>` tags).
- Record the raw `-it` baseline on text identity (it has none yet — generic instruct) and vision (vis-real-*).

## Step 1 — Align inference + training to `-it`'s chat template
- The engine's `_prepare_vision_inputs`/ChatFormatter hand-builds turn tags because the *base* had no template. `-it` HAS `chat_template.jinja`. **Train and infer with the SAME `-it` template.** Update the engine managed-mode path to apply `-it`'s template (or `processor.apply_chat_template`); verify grounding via the harness with that template.
- This is real engine work (use `/engine`); may be the single biggest lever.

## Step 2 — Curriculum: identity + grounding-PRESERVING vision (mixed)
- Mix GAIA identity / persona / tool-routing / multiturn / deliberation (the identity bake) **with vision data (VQA + captions)** so grounding is actively rehearsed during tuning and doesn't drift. Reuse Prime-VQA + COCO captions.
- Target ~30–40% vision in the mix (empirical; raise if grounding drifts). Spiral/interleave rather than pure text-first so vision isn't all at the end.
- Format EVERYTHING with `-it`'s chat template.

## Step 3 — Training: gentle, grounding-protected identity LoRA
- Base = `gemma-4-E4b-it`. LoRA on LM only (identity). **Projector + vision_tower FROZEN** (they ground — do not touch; this is the opposite of the gix instinct).
- **Low LR** (~5e-5, vs the 2e-4 used from-base) and **modest rank** (r=16–32) — we're nudging, not rebuilding.
- save_steps small; **bird-vs-pizza grounding check at every checkpoint**; **early-stop if grounding degrades**. Resilient wrapper for 22a.
- Keep it SHORT first — identity on an instruct model needs far less than from-base.

## Step 4 — Validation gates (ALL must pass before cutover)
1. **Grounding preserved** — bird-vs-pizza divergence ≈ the Step-0 `-it` baseline (the hard gate).
2. **Vision battery** vis-real-* ≥ baseline (real signal now that grounding exists).
3. **Text battery** ≥ ~94% + GAIA identity present.
4. **Multimodal gauntlet** (`48m`) — vision/audio sections.

## Step 5 — Cutover
- Only if all gates pass. Merge (standard tower graft; `-it` towers are good), retain **V14 as rollback**, run `post_training_reset`, symlink `/models/core` → new. Document battery numbers on `ujs`.

## Risks → mitigations
- **Identity tuning erodes grounding (LM↔tower drift)** → mixed curriculum + low LR + per-checkpoint grounding gate + early stop. THE central risk.
- **Template mismatch** (train vs `-it` vs engine) → standardize on `-it`'s `chat_template.jinja` everywhere.
- **Catastrophic forgetting of grounding** if vision underrepresented → enforce vision ratio; rehearse.
- **VRAM** → `-it` NF4 ~9GB + r16–32 LoRA fits 16GB (same as V14).

## Open questions to resolve in Step 0/1
- Does raw `-it` ground in our managed-mode engine, or only via HF `apply_chat_template`? (decides how much engine work Step 1 needs.)
- Minimum identity-LoRA before grounding drifts? (empirical, gated.)

## First action next session
Step 0 — load raw `-it`, run bird-vs-pizza with its chat template. ~3 min, GPU only (no GAIA offline needed beyond freeing the GPU). Everything downstream depends on that baseline.

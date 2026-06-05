# 2026-06-04 — Vision grounding RESOLVED: it was an NF4 vision-tower bug all along

**Supersedes the conclusions in `2026-06-03_core2x_vision_grounding_diagnosis.md` and `2026-06-04_gix_vision_grounding_conclusion.md`.** Those declared vision grounding "not achievable via LoRA on the base / needs a different foundation." **That was wrong.** The real cause was a single load-time quantization bug. Fixed in the engine; **production V14 now grounds, no retrain.**

## The bug (`GAIA_Project-5fh`)

`bitsandbytes` `llm_int8_skip_modules` does **not** skip nested `.linear` submodules, so the Gemma 4 **`vision_tower`'s ~114 `Linear4bit` modules were NF4-quantized despite being in the skip list** — corrupting the image features into noise. The model literally "saw" gray/garbage (it described real photos as "solid gray background with repeating text artifacts").

This is the *exact* bug the **audio** tower already had a workaround for — the engine dequantizes `audio_tower` after load. Vision was simply never given the same treatment.

## Why it hid for so long

- **Base + LoRA models (v14, v15_pl, pilots)** masked it — they'd learned to ride the text/caption prior and *ignore* the (garbage) vision features. So they emitted plausible-but-ungrounded captions, and **loss looked fine** (this is why loss fooled us 4×).
- **The `-it` instruct model exposed it** — it's trained to actually *use* the image, so when fed garbage it described garbage. That mismatch (coherent model + garbage output) was the tell.

## The fix

`gaia-engine` commit `8f4ac57` (`core.py`): after the NF4 load, dequantize `vision_tower` + `embed_vision` `Linear4bit → bf16`, mirroring the existing audio-tower dequant. No-op in bf16.

## Results (validated end-to-end)

- **Standalone:** raw `-it` + dequant → bird→"Painted Stork", pizza→"pizza in an oven", "Is there a bird?" → Yes on bird / No on pizza. Full grounding + VQA.
- **Production V14 + dequant:** "Describe" → bird→"a bird on a branch", pizza→"a pizza on a plate" (per-image-correct).
- **Engine serving path** (`gaia-core:8092` `/v1/chat/completions`) after the fix: bird→"a brown and white bird sitting on a tree branch", pizza→"a pizza with vegetables on a plate" (was "a man on a skateboard" for *every* image).
- **Doctor battery vision: 0% → 75%** (12/16) on V14 — clears `ujs` (≥30%) and the `48m` gauntlet (≥60%). **No retrain.**

## Diagnostic chain that cracked it (the right way to do this)
Stage-by-stage cheap probes: pixel_values (varied ✓) → vision_tower output → projector → LM injection (cos 0.37, distinct ✓) → attention (23%, under-attending). The "distinct features but under-attended + garbage output" pattern pointed at *corrupted-but-present* features. The decisive move was testing the **`-it` model** (which tries to use vision) and then **counting `Linear4bit` under `vision_tower` (114!)** → dequantize → grounds.

## Follow-ups
- **Trainer** (`scripts/train_core_multimodal.py`) only dequantizes `audio_tower` (and only with audio curriculum). It must **also dequantize `vision_tower`** so future training (the `xln` re-bake, VQA polish) learns from *real* features, not garbage. (`5fh`/`xln`)
- **Audio parallel (next, for `m0b`):** the engine's audio dequant covers `audio_tower` but **NOT `embed_audio`** (the audio projector) — likely the same partial corruption. Check before investing in broader audio training.
- **Optional:** V14 captions well but yes/no VQA is weak (no VQA in its training); a light VQA pass (Prime-generated data exists) would add instruction-following. Not required for grounding.

## Lessons (now in memory)
1. **Loss ≠ grounding.** Use the standalone per-image-divergence test (same prompt, 2 different images → outputs must differ) as the gate.
2. **Validate the inference path before concluding a model can't do something.** We nearly shelved a working capability — and planned a multi-day re-bake — over a quantization bug. The `-it` model (one that *uses* the feature) is a great corruption detector.
3. **When one tower needs a workaround, check the sibling towers.** Audio had the dequant; vision and the projectors didn't.

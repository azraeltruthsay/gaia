# 2026-06-03 → 06-04 — gix: vision grounding is not reachable via LoRA on the base (conclusion)

**Follow-up to** `2026-06-03_core2x_vision_grounding_diagnosis.md`. That entry ended at the attention diagnostic (the LM attends only ~23% to image tokens). This entry closes the loop: we tried the diagnosed fixes and they don't work. **Conclusion: vision grounding is NOT achievable via LoRA fine-tuning on the `google/gemma-4-E4B` base in our setup.** Shelving it. V14 (text/identity upgrade) remains the shipped win.

## What we tried (the data + capacity levers)

The diagnosis said the LM ignores correct, injected image features and rides the text/caption prior. Two levers: (1) non-gameable data, (2) higher capacity to reshape attention.

1. **Non-gameable VQA data** — generated with **local Prime** (text-only, from captions = the LLaVA method; Prime never sees the image, the caption is the ground-truth bridge). Switched here because Groq's key is empty in all containers and **Azrael no longer uses OpenAI**. `scripts/gen_vqa_prime.py` with an anti-hallucination guardrail ("use ONLY caption facts") produced 3,773 grounded, diverse, balanced Q&A from 1,000 images in 106 min on CPU (GAIA stayed online). Quality verified: top answer "yes" only 4.8% (vs templated where person/Yes dominated).
2. **4× capacity** — `--lora-r 64` (vs 16), trainable params 139.7M / 1.73% (confirmed). Combined LM+projector regex.

Pilot: 2,500 steps, completed clean.

## Result: negative (and conclusive)

Standalone bird-vs-pizza test on the r64 Prime-VQA adapter:

| probe | bird | pizza |
|---|---|---|
| "main object?" | a train | a train |
| "Is there a bird?" | Yes | Yes |
| "Is there a pizza?" | Yes | Yes |
| "Describe" | tennis racket | tennis racket |

Still fully image-independent. **Four configurations now, all negative:**
- v14 — LM-LoRA r16, captions
- v15_pl — LM+projector LoRA r16, captions
- templated-VQA pilot — LM+projector r16, templated VQA
- Prime-VQA pilot — LM+projector **r64**, Prime-generated VQA

Better data, 4× capacity, and projector training — none shifted the LM off the 23%-attention/text-prior behavior.

## Conclusion

The vision **pipeline is provably perfect** (tower encodes distinct content cos 0.42 → projector 0.37 → injects into the LM's input embeddings at the image positions 0.37). The bottleneck is the **base LM**: `gemma-4-E4B` is a PT *foundation* checkpoint that never learned strong image-token attention, and **LoRA — even high-rank, even with good VQA — cannot bootstrap that capability** on 16 GB.

**Realistic paths forward (both large, separate efforts):**
- **(a) Start from a vision-*instruct* checkpoint that already grounds**, then carefully identity-tune without breaking it — this is exactly issue **`xln`** (re-bake recipe without LM↔tower drift). Validate with bird-vs-pizza.
- **(b) Full fine-tune** (not LoRA) with real compute.

**Recommendation: shelve vision grounding.** Closing `gix` (the LoRA-scope approach is conclusively disproven); the real path is `xln`.

## The methodological lesson (the most valuable takeaway)

**Loss ≠ grounding.** Caption and VQA loss both drop via *answer-prior gaming* with zero image use — it fooled us four times (vision 1.95, then 0.665, etc.). The ONLY trustworthy gate is the **standalone per-image-divergence test**: same prompt, two genuinely different images → outputs *must* differ. Run it on a checkpoint *before* trusting any loss or battery number. This is now the standard vision-eval gate.

## Assets produced this sprint
- `scripts/gen_vqa_prime.py` — local-Prime VQA generator from captions (keyless, GAIA-online).
- Standalone grounding test pattern (`/shared/*grounding*.py`) — the per-image-divergence gate.
- Confirmed: live Prime is a **Qwen3-8B GGUF**, not the Gemma-26B the docs claim (matches stale constants).
- Pilot adapters retained under `/models/lora_adapters/` (core_vqa_pilot, core_vqa_prime_pilot, v15_pl) for reference.

## State at close
- Live Core = **V14** (text upgrade). GAIA → Parked Maintenance.
- `gix` closed (LoRA path exhausted → `xln`). `axa`/`rdn` closed earlier. `22a` mitigated (P3).

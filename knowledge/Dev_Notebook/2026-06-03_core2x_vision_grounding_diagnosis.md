# 2026-06-01 → 06-03 — Core 2.x: the vision-grounding diagnosis marathon

**TL;DR:** Shipped a real win — Core 2.x **v14** is live (text/identity upgrade, battery text 100%), built on the `axa` label-masking fix and the `22a` watchdog mitigation. But the headline lesson is a *negative* result, fully diagnosed: **Core does not ground on images**, and we proved exactly why — not NF4, not the encoder, not the projector, not injection, not the data. The LM is handed correct, distinct image features and has *learned to ignore them* (attends only 23% to image tokens that are 94% of the sequence). Vision grounding is a dedicated VL fine-tuning research sprint, not a recipe tweak. `gix` holds the full map.

---

## How it started

Session opened on memory reconciliation — the auto-memory had drifted ~2 months (still described the old 3-tier Qwen architecture). Rewrote it for **Sovereign Duality** (two-tier Gemma 4, nano deprecated) and the new cognition systems (World Model, Affect, Unified Skills, training infra). Then picked up the live thread: **Core 2.x multimodal training**.

## Act 1 — the vision stall was a label-masking bug, not image diversity (`axa`)

Found the prior run (`v12`) had completed but vision loss was pinned at **~11.6** (≈ ln(vocab) for Gemma-4's 256k vocab) while every text category converged. `6s8` had blamed image *diversity* (421 unique images) and shipped a 5,400-image diverse-COCO pool — but the loss never moved.

Real cause: `MultimodalCollator` built `labels = input_ids.clone()` masking only padding, so the **~256 image soft-token placeholders per vision sample were scored** — unlearnable, dominating the vision mean. Fix: mask everything up to the assistant turn marker (`<|turn>assistant<turn|>` = atomic tokens `[105,111457,106]`); score only the answer. Follow-up: skip samples whose answer truncates past `MAX_SEQ_LENGTH` (else all-`-100` → NaN; killed v13 at the vision phase with `multiturn=NaN`).
Commits: `e52aedd`, `2676837`.

## Act 2 — 22a: surviving the RC watchdog

v14 died at step 6844/11000 with `the launch timed out and was terminated` (bitsandbytes `ops.cu`) — the **NVRM RC watchdog** (the RTX 5080 drives both the desktop *and* compute). Mitigation (`76137bb`): `scripts/train_core_multimodal_resilient.sh` (auto-resume across crashes) + `--save-steps 500` + a **numeric checkpoint-sort bugfix** (lexical sort would resume from `checkpoint-900` over `checkpoint-11000`). Resumed from checkpoint-5500 → completed clean.

## Act 3 — v14 result and the false summit

Full 11000-step v14: vision loss **1.955**, audio 0.218, all text converged. Merged (`merge_and_save_adapter.py`, towers grafted), cut over `/models/core → V14` (V11 retained as rollback). Battery: **text 100%**.

Then the battery's `vis-real-*` tests scored **0%** — the model emits generic COCO captions *identical across different images* ("a man on a skateboard" for bird, pizza, bus). **The vision loss of 1.95 was the model learning the caption *prior*, not grounding.** Answer-token CE can't tell a grounded caption from a plausible generic one. (Lesson #1, learned the hard way: **loss is not grounding.**)

## Act 4 — `rdn`: it's not the engine

Suspected the engine's managed-mode vision path. Instrumented it: image **is** fully encoded (266 image tokens + real `pixel_values`), the vision path engages, features reach `generate()`. backend=`engine`, not cpp. `has_vision=True`, tower not detached, NF4 skips the towers. So `rdn` (engine bug) → **closed, not an engine bug.** It's modality collapse.

## Act 5 — `gix`: chasing the projector, then the data, then the truth

Systematically eliminated each remaining hypothesis with cheap standalone tests (the **bird-vs-pizza per-image-divergence test** — the gate that actually detects grounding):

1. **Prompt order** — reordering image-first: no change. ✗
2. **NF4 corrupting the tower** — disproven; tower output for bird vs pizza is *distinct* (cos 0.42), projector preserves it (0.37). NF4 is fine.
3. **Frozen projector** — trained it. First via `modules_to_save` (added `--modules-to-save` + a dequant-before-PEFT fix, `2f494c1`/`6c426e4`) → crashed at the vision phase (`AuxiliaryTrainingWrapper.forward() missing 'x'`, PEFT incompat). Pivoted to **projector-LoRA** (regex incl. `embed_vision.embedding_projection`) → trained to 4500 → **negative** (bird ≈ pizza). Projector was already producing distinct features; training it was the wrong lever.
4. **Injection** — hooked the LM's `inputs_embeds`: image features land at the 266 image positions verbatim (cos 0.37; text-position control 1.0). Features reach the LM. ✓
5. **Grounding-forcing data** — generated templated VQA (6,274 pairs: object-naming + yes/no presence incl. negatives) and ran a pilot. **Negative**: "Is there a bird?" → "Yes" on *both* bird and pizza; vision loss fell to 0.665 but only by gaming answer priors ("person"/"Yes."). (Lesson #1 again.)

### The verdict — attention diagnostic
`output_attentions` from the answer position: **0.23 attention mass on image tokens**, though image tokens are **94% of the sequence** (uniform would be 0.94). 77% of attention goes to the 16 text tokens. So **not a hard structural mask** (image tokens are attendable), but the LM has **learned to under-attend to the image** and ride the text/caption prior.

## Conclusion

Full chain proven: tower encodes ✅ → projector preserves ✅ → injects into LM ✅ → **LM under-attends (23%) and won't extract content** ⚠️. This is a **learned-behavior / training-capacity** problem, not a bug — and not cracked by caption data, projector training, or cheap templated VQA.

**Real fix = a dedicated VL fine-tuning sprint:** real LLaVA-Instruct (balanced, non-gameable) + higher-capacity training (higher LoRA rank and/or more of the LM, longer runs) to reshape attention onto image tokens, validated *only* by the bird-vs-pizza per-image-divergence test.

## What shipped / state

- **Live:** Core = **V14** (text/identity upgrade, battery text 100%; vision == V11, no regression).
- **Closed:** `axa` (masking), `rdn` (not an engine bug). **Mitigated:** `22a` (resilient wrapper, P3). **Open w/ full map:** `gix` (vision grounding = VL sprint), `ujs`/`6s8`/`48m`/`4d3`.
- Commits pushed: `e52aedd`, `2676837`, `76137bb`, `2f494c1`, `6c426e4`.

## Lessons for next time

1. **Loss ≠ grounding.** A strong caption/answer prior drives loss down with zero image use. The only trustworthy gate is the **standalone per-image-divergence test** (same prompt, two different images → outputs *must* differ). Run it *before* trusting any loss or battery number.
2. **Diagnose in stages with cheap probes** (pixel_values → tower → projector → injection → attention). It turned an opaque "vision broken" into "LM under-attends at 23%."
3. **Operational:** free the GPU for training via **maintenance-mode (doctor `/maintenance/enter`) + deep-sleep**, not the `consciousness/training` pin (reconcile fights it). Use the **resilient wrapper** for any >2h run (`22a`). `docker exec -i` for heredocs (stdin), absolute script paths (container workdir is `/app`). See [[project_training_gpu_offline]].
4. **PEFT `modules_to_save` is incompatible with Gemma4's `embed_vision`** (forward-signature crash); also requires bf16 (dequant 4-bit first). LoRA-on-the-inner-Linear works instead.

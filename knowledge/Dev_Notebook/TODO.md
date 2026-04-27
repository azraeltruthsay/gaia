# GAIA Development TODO

> Running task list. Persists across sessions. Update as work completes.
> Last updated: 2026-04-24

## Multimodal Core Push (beads `GAIA_Project-8oz`, started 2026-04-23)

**Status: Milestone reached 2026-04-24.** Core describes real images in coherent natural language. `Gemma4-E4B-GAIA-Core-Multimodal-v2/` is the production candidate.

- [x] **Phase 0: Reconnaissance** — Tower weights exist in base; "missing" was QAT vs flat naming mismatch. (2026-04-23)
- [x] **Phase 1: Tower graft** (`-wze`) — `Gemma4-E4B-GAIA-Unified-v5-Multimodal` (14.79 GB, 0 MISSING keys). (2026-04-23)
- [x] **Phase 2: Engine re-attachment** (`-9a8`) — Multimodal forward pass works; image inference ~1-2 s/turn. (2026-04-23)
- [x] **Training pipeline built** — `train_core_multimodal.py` with LoRA + dequantize-before-save + tower graft. (2026-04-23)
- [x] **Phase 4: Retrain from v5-Multimodal base** (`-d74`) — Three iterations (60 pairs primitives, 406 pairs primitives, 2000 COCO pairs). COCO won. (2026-04-24)
- [x] **Curriculum expansion** (`-ejo`) — Built `build_core_coco_curriculum.py`. 2000 COCO image-caption pairs. Programmatic primitives (60 → 406) tested first; plateaued against pretraining priors. (2026-04-24)
- [ ] **Production-promote v2** — Decide whether `/models/core` permanently points at Multimodal-v2 or only when image input expected. Add doctor-battery checks for Multimodal-v2 to confirm text-side cognitive scores haven't regressed.
- [ ] **Phase 5: Web UI + doctor battery integration** (`-8ki`) — gaia-web image upload; gaia-doctor vision smoke test.
- [ ] **Test on novel out-of-distribution images** — phone photos, screenshots, GAIA dashboards. Validate generalization beyond COCO's distribution.
- [ ] **Audio side training** — Towers attached, never trained for audio. Will need an audio-text curriculum (LibriSpeech, AudioCaps, ...).
- [ ] **Phase 6: gaia-audio STT deprecation decision** (`-shj`).
- [ ] **Trailing UnboundLocalError** at end of `train_core_multimodal.py` — `del model` after model already deleted. Cosmetic. (P3)
- [ ] **Bonus: scale COCO to LLaVA-Instruct-150K or full COCO train2017** — 2000 was a starter. More data, longer training, stronger model.

See:
- Plan: `2026-04-23-multimodal-core-plan.md`
- Day 1 journal (pipeline buildout): `2026-04-23-multimodal-core-day.md`
- Day 2 journal (COCO milestone): `2026-04-24-multimodal-core-coco-milestone.md`

## Other items from 2026-04-23 session

- [x] **Engine health consistency** — `/health`, `/status`, `/model/info` now return canonical fields across all backends. (2026-04-23)
- [x] **Orchestrator reconcile uses `/model/swap`** — no more 409 loops when a stale worker has a different model loaded. (2026-04-23)
- [x] **Session RAG threshold fix** — raised floors from 0.15/0.10 to 0.45/0.40, added trivial-greeting skip. Kills confabulation from noise retrieval. (2026-04-23)
- [x] **Prime cognitive battery via pipeline** — `--force-tier prime` flag; 58/62 = 94% (clean RAG), canary 97%, crammable 91%. (2026-04-23)
- [x] **Fresh GGUFs** — `prime.gguf` (Qwen 9B v4, 5.3 GB), `core.gguf` (Gemma E4B v5, 5.0 GB). (2026-04-23)
- [x] **Post-training reset policy** — `scripts/post_training_reset.py` archives sessions + invalidates KV. Not yet wired into training scripts.
- [ ] `/cache/rebuild_identity_prefix` endpoint on engine (post-training-reset currently falls back to delete-and-regenerate).

---

## Phase 6: Relational Autonomy & Situated Intelligence

- [x] **Initiative Bridge (6a)** — PROMOTED. GAIA now has a proactive "Inner Voice" triggering research turns during idle ACTIVE time (15+ min). (2026-04-11)
- [x] **Penpal Protocol Validation (6c)** — COMPLETE. Automated loop with NotebookLM is persistent. Penpal responses now anchored in live GaiaVitals. (2026-04-11)
- [x] **Sovereign Duality (6c)** — COMPLETE. Trinity deprecated. E4B (Operator) promoted to Always-On Conscious. VRAM locked at 8.9GB baseline. (2026-04-12)
- [x] **Native Multimodal Stack (6b)** — COMPLETE. Deprecated legacy STT for native Gemma 4 E4B audio delivery. (2026-04-12)

- [x] **Standard Usability (6b-Stabilization)** — COMPLETE. VRAM normalized via revised presets: 'awake' (Nano-GPU), 'operating' (Core-GPU), 'focusing' (Prime-GPU). (2026-04-12)
- [x] **ChatML Decoupling (6b-Format)** — COMPLETE. Generalizing the prompt builder for Gemma 4 (non-ChatML) compatibility. Stop tokens and think-tags aligned. (2026-04-12)
- [x] **Gemma 4 Quantization (6b-Models)** — COMPLETE. E4B and E2B running in NF4. Identity-baked and fused. (2026-04-12)
- [ ] **DocSentinel Automation (6d)** — Background wiring of glossary mining and capability cataloging. **[IN PROGRESS - CHORD BETA]**

## Phase 5: The Gemma Chord (Gemma 4 26B-A4B)

- [x] **Limb Refactor** — COMPLETE. Global terminology shift (Tools/Skills → Limbs) via GaiaCLI implemented and verified in active cognition. (2026-04-12)
- [x] **Great Consolidation (5-C)** — COMPLETE. Unified 7 key systems: Neural Router, GaiaVitals, Capability Engine (Limbs), DocSentinel, Decentralized Config, Palace Alignment, and terminology shift. (2026-04-11)
- [x] **Foundation Tuning (5f)** — COMPLETE. Identity verified on 26B-A4B Sovereign. 15 epochs, loss 1.95. 'I am GAIA...'. (2026-04-11)
- [x] **Sovereign Awareness (5i)** — COMPLETE. Weighted Router, Personal Force Field (Adversarial Translation), and CPR Recovery Loop implemented and live. (2026-04-11)
- [x] **Expert Cache Optimization (5b-Tuning)** — COMPLETE. Per-Layer Cache (4 slots/layer) eliminates cache thrashing. (2026-04-11)
- [x] **Abliteration Pass (5j)** — SKIPPED. Verification confirmed base 26B-A4B weights are already uninhibited/sovereign. (2026-04-11)
- [x] **Inference Compositor (5h)** — COMPLETE. JIT Expert Swapper + KV Segment Selector live in GAIA Engine. (2026-04-11)

## Architecture & Cockpit

- [x] **GaiaCLI Implementation** — COMPLETE. Unified Python CLI replacing gaia.sh, test, and promote scripts with integrated health reporting. (2026-04-11)
- [x] **Native Tool Calling** — COMPLETE. V2 models (Core 4B, Prime 9B) fully validated on curriculum. (2026-04-09)
- [x] **Neural Grounding Stage 0** — COMPLETE. Hierarchy of Truth (KG > Vector > Web) active in pre-inference. (2026-04-08)

## Sovereign Shield & Security

- [x] **Personal Force Field** — COMPLETE. Adversarial intents translated to "Awareness" summaries for native model rejection. (2026-04-11)
- [x] **Blast Shield Hardening** — COMPLETE. tokenization + regex boundaries. 17/17 tests pass. (2026-04-09)

## Completed (Legacy)

- [x] **MemPalace deployed** — Palace orchestrator + MCP tools live (2026-04-07)
- [x] **Clutch protocol implementation** — CM as sole tier authority (2026-04-08)
- [x] **Parallel Observer pipeline** — Always-on audit of reasoning streams (2026-04-08)

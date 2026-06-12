# SAE Atlas Build Plan — the measurement instrument

> **Status:** Plan (2026-06-12) · **Authors:** Azrael + Claude · **Premise:** COGNITIVE_ARCHITECTURE.md §2a
> **Goal:** SAE feature atlases for **both committed models** (Gemma4-E4B Core, Qwen3-VL-8B Prime), in
> **both paths** (safetensors GPU, GGUF CPU), so we can map cognitive signals (affect, Samvega, Council,
> coherence) to *real features* and re-derive the brain-region map. Build, don't invent — the machinery
> exists.

---

## 1. What exists vs. what to build

| Piece | State |
|-------|-------|
| SAE training (`gaia-engine/.../sae_trainer.py`) | ✅ `record_activations` (PyTorch `output_hidden_states`) + `train_sae` (one SAE/layer, 4096 feats, normalize→Adam→sparsity) |
| Safetensors activation capture | ✅ via `record_activations` (GPU model forward) |
| GGUF activation capture | ✅ residuals exposed: `gaia_cpp.generate(capture_hidden=True) → {layer: vec}`; polygraph `capture()` unifies tuple+dict (`core.py:184`) |
| **GGUF→SAE recorder** | ❌ **BUILD:** `record_activations_gguf()` that drives `gaia_cpp` capture and populates `self.activations[layer]` in the *same* shape `train_sae` consumes |
| Atlases for current pair | ❌ none on disk (`/shared/atlas` empty) — **build** |
| Brain-region map | ⚠️ stale (`brain_region_map.md`, `brain_region_atlas.json` — Mar 25, pre-Duality) — **re-derive** |

So the only genuinely new code is the **GGUF recorder bridge**; everything downstream (`train_sae`, atlas
storage, labeling) is shared between paths.

---

## 2. The corpus is the product (stratified to elicit the states we want to map)

An atlas is only useful if the features we care about are *present in the training activations*. So the
prompt corpus must deliberately elicit the cognitive states we intend to map to features — otherwise we
get a generic atlas that can't tell us whether "coherence tension" has a neural signature.

Stratified corpus (≈N prompts per stratum, balanced):
- **Coherence / contradiction** — prompts that introduce, contain, or resolve contradictions (so we can
  later ask: which features fire when Samvega/consistency fires?).
- **Curiosity / knowledge gap** — novel topics, unanswerable questions.
- **Competence / problem-solving** — tool-use, debugging, multi-step reasoning.
- **Identity / self-reference** — "who are you", introspection, self-model.
- **Affect-laden vs. neutral** — emotionally weighted vs. flat factual.
- **Deliberation** — complex trade-off reasoning (Council-like).
- **Register** — chitchat / greeting vs. technical (the "how are you" failure lives here).
- **Multimodal** (Core + Prime are VL) — image-grounded prompts, since vision features matter.

Reuse existing material where possible: conversation examples in `knowledge/`, journal entries, the
cognitive battery. Tag each prompt with its stratum so feature→state correlation (step 4) is possible.

---

## 3. Target layers & hyperparameters

- **Layers:** the residual stream at a spread of depths. `sae_trainer`'s example used `[6,12,18,23]`;
  pick per model from its true depth (Gemma4-E4B and Qwen3-VL-8B differ) — roughly early/mid/late
  quartiles, plus the layer the polygraph/brain-map already keys on. Confirm layer counts per model
  before recording.
- **SAE:** `num_features=4096` (overcomplete), `sparsity_weight≈0.01`, `epochs≈50`, `lr=1e-3` — the
  existing defaults; tune sparsity to hit a usable dead-feature/L0 balance.
- **GGUF caveat:** capture residuals at the *same* logical layers (`l_out-N` ↔ `hidden_states[N+1]`);
  mind the off-by-one between ggml `l_out-N` and transformers indexing (documented in
  `hidden_state_capture.hpp`).

---

## 4. Validation — does a feature *mean* something?

An atlas of 4096 anonymous features is useless until features are interpretable:
- **Activation-maximizing exemplars** — for each top feature, find the corpus prompts/tokens that fire it
  hardest → a human-readable label (reuse `feature_labels`).
- **State correlation (the payoff)** — using the stratum tags, test: does any feature fire selectively on
  the *contradiction* stratum? the *curiosity* stratum? If yes, we have a **neural correlate of the
  cognitive signal** — that's how we verify affect/Samvega/Council are real, not just plausible.
- **Cross-path diff** — compare the safetensors atlas vs. the GGUF atlas for the same model: which
  features survive quantization, which distort. A free interpretability result about what Q4 does to her.

---

## 5. Outputs

- `/shared/atlas/<model>/<path>/` — per-layer SAE weights + feature labels + corpus/stratum metadata
  (model ∈ {core, prime}; path ∈ {safetensors, gguf}).
- Re-derived **brain-region map** from the new atlases (replaces the stale Mar-25 artifacts), feeding the
  dashboard activation monitor.
- A **feature↔cognitive-signal table** (step 4) — the actual measurement deliverable.

---

## 6. Phases

- **A0 — GGUF recorder bridge** (only new code): `record_activations_gguf()` via `gaia_cpp` capture +
  polygraph; unit-test it produces `self.activations` matching the safetensors shape.
- **A1 — Corpus** (§2): assemble + tag the stratified prompt set.
- **A2 — Core atlas**: record (safetensors GPU + GGUF CPU) → `train_sae` → label. Validate (§4).
- **A3 — Prime atlas**: same for Qwen3-VL-8B.
- **A4 — Brain-map re-derivation** + the feature↔signal table.

---

## 7. Operational constraints

- **GPU:** safetensors recording + SAE training need the 16GB card → run in maintenance mode / free-the-
  GPU (same dance as QLoRA training; see `project_training_gpu_offline`). Schedule off production hours.
- **GGUF recording is CPU** (llama.cpp) → no GPU contention for capture; the SAE *training* on those
  activations still wants GPU but is small (4096-feature autoencoder).
- **Gated by model commitment** — this is the investment that's only worth it once we pin the pair
  (COGNITIVE_ARCHITECTURE.md §4 step 0). Pinning also re-enables KV-rehydration for temporal continuity.

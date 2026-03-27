# Dev Journal: 2026-03-27 — gaia_cpp + Self-Supervised Code Skill Loop

**Era**: Sovereign Autonomy
**Duration**: ~8 hours
**Theme**: GAIA learns to write code by grading herself

---

## Part 1: gaia_cpp C++ Inference Engine — Fixed & Working

### GIL Crash Fix (the big one)

The pybind11 wrappers for `generate()` and `generate_stream()` had a fatal
GIL management bug: `_generate_impl()` created `py::array_t` (numpy) objects
while the GIL was released — instant SIGABRT on every inference call.

**Root cause**: `GenerateResult::hidden_states` was `map<int, py::array_t<float>>`.
Creating numpy arrays requires the Python GIL. The pybind11 lambda released the
GIL before calling C++ inference, which internally created numpy arrays = crash.

**Fix**: Changed to `map<int, vector<float>>` (raw C++), added
`_hidden_states_to_numpy()` converter called AFTER GIL re-acquisition. Also
fixed streaming callback to use `py::gil_scoped_acquire` instead of raw
`PyGILState_Ensure/Release`.

### cb_eval Optimization

The hidden state capture callback fired for EVERY tensor in the compute graph
(thousands per forward pass) and always returned `true` for `ask`, causing
unnecessary host memory syncs. Now returns `false` for non-target tensors —
only syncs the ~9 layers we actually capture.

### LoRA Adapter Support

Implemented `load_lora()`, `clear_lora()`, `active_adapter_count()` using
llama.cpp b8250's plural API (`llama_set_adapters_lora`). Multiple adapters
with independent scales. HTTP endpoints: `/adapter/load`, `/adapter/unload`,
`/adapter/list`.

Tested with converted GGUF adapters — scale 0.5 gives clean blending with
base identity.

### Performance

Quantized Prime to Q4_K_M: 8.2 GB → 4.7 GB, 6.5 → 10.8 tok/s on CPU (+66%).
Memory bandwidth is the ceiling (DDR5 ~72 GB/s practical).

---

## Part 2: Self-Supervised Code Skill Loop

### The Loop

```
Challenge → Generate (Prime Q4_K_M + LoRA) → Grade (MCP sandbox)
    → Pass: escalate difficulty
    → Fail: self-correct → samvega artifact → training buffer
        → LoRA training (gaia-study) → GGUF convert → /adapter/load
        → Retest with adapter
```

### Infrastructure Built

- `knowledge/curricula/code_challenges.jsonl` — 20 challenges (L1 + L2)
- `gaia-core/.../code_evaluator.py` — sandbox grading (syntax + runtime + assertions)
- `gaia-core/.../code_skill_loop.py` — loop controller with retry, validation, samvega
- `gaia-study/.../adapter_to_gguf.py` — post-training safetensors→GGUF conversion
- `gaia-engine/config.py` — centralized engine configuration

### Training Progression

| Iteration | Examples | L1 | L2 | Total | Key Fix |
|-----------|----------|-----|-----|-------|---------|
| Baseline | 0 | 60% | 20% | 40% | — |
| v1.0 | 13 | 100% | 20% | 60% | Single-line collapse (functions) |
| v1.1 | 20 | 90% | 50% | 70% | Class formatting + algorithms |
| v1.2 | 27 | 90% | 70% | 80% | Memoize, LCS, sliding_window |
| v1.5 | 37 | ~95% | ~78% | ~80% | Timer, RangeIterator, reinforcement |

### Failure Modes Identified

1. **Off-topic hallucination** (~20% at temp 0.3): "caesar" → "gaussian_blur".
   Fixed by retry + validation + low temperature (0.1).

2. **ChatML token leakage** (~10%): `<|im_start|>` as first token.
   Fixed by stripping + rejection in `_extract_code()`.

3. **Single-line collapse**: Multi-line code emitted on one line with spaces
   instead of newlines. Deterministic for certain prompts. Fixed by LoRA
   training on properly-formatted examples.

4. **Algorithm bugs**: LCS backtracking, memoize decorator confusion.
   Partially fixed by training — 2 solid failures remain.

### Training Logistics

- GPU must be completely free for 8B QLoRA (15.5 GB needed)
- `consciousness/training` mode frees GPU but zombie CUDA contexts can persist
- Must stop gaia-core + gaia-nano containers for reliable training
- transformers 4.51+ required for Qwen3 architecture support
- Training time: ~3 min for 37 examples, 170 steps on RTX 5080
- GGUF conversion: 2 seconds via `convert_lora_to_gguf.py`

### Key Insight

The 8B Q4_K_M model understands algorithms correctly — the failures are
almost entirely output discipline (formatting, token selection, hallucination).
LoRA training on ~37 curated examples was enough to fix most issues. The
model has the knowledge; the adapter teaches it the discipline.

---

## Commits This Session

1. `fix: GIL crash in gaia_cpp` — defer numpy to after GIL re-acquisition
2. `refactor: centralize env config into gaia_engine/config.py`
3. `feat: Q4_K_M default, 16 threads, config.py in Dockerfile`
4. `feat: LoRA adapter support via llama.cpp b8250 plural API`
5. `feat: self-supervised coding skill loop` — curriculum, evaluator, converter
6. `fix: code skill loop` — temp 0.1, retry on off-topic/collapse
7. `feat: code_skill_v1 training data` — L1 100% with LoRA
8. `feat: L2 challenges + retrained adapter` — L2 50%→70%
9. `feat: code_skill_v1.2` — 27 examples, 80% total
10. `milestone: L1 100% L2 ~78%` — 5 training iterations

---

## What's Next

- **L2 cleanup**: Train out memoize decorator + LCS failures (2 solid)
- **L3 challenges**: Recursion, trees, graphs, dynamic programming
- **Autonomous loop**: Wire into SLEEP cycle for overnight practice
- **MCP tool**: `code_skill_drill` for on-demand triggering
- **Nano/Core on gaia_cpp**: Three-stage Dockerfile for all tiers

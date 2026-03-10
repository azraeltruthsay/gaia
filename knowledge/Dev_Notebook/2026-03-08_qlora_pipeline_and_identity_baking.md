# Dev Journal: QLoRA Pipeline Wiring & Identity Baking Architecture

**Date**: 2026-03-08
**Session focus**: Wire QLoRA adapters into live cognitive pipeline (Phase 2), discover adapter/AWQ incompatibility, design the full identity-baking architecture (Phase 3 plan)

---

## Phase 2 Completed: Adapter Pipeline Wiring

### What was built

Wired LoRA adapter support through the entire cognitive pipeline so that when a trained adapter exists, Prime automatically uses it for all generation.

**Files modified** (production + candidate):

1. **`_model_pool_impl.py`** — `forward_to_model()` gained `adapter_name: Optional[str]` parameter. When set, uses `create_chat_completion_with_adapter()` (atomic save/restore). Fallback chain intentionally skips adapters for cloud/CPU models. Added `register_adapter_with_prime()` helper for dynamic vLLM adapter registration via `/v1/load_lora_adapter`.

2. **`agent_core.py`** — Added `_DEFAULT_ADAPTER = "gaia_persona_v1"`, `_ADAPTER_BASE_PATH`, and `_resolve_adapter(model_name)` method. Checks if model is Prime/Thinker AND adapter exists on disk (`adapter_config.json`). Wired into 9 `forward_to_model` call sites (planning, think-tag retry, slim prompt, thinker polish, recitation, fragment gen/assembly, confidence assessment, truncation reflection). ExternalVoice streaming: sets adapter before stream, clears after.

3. **`sleep_task_scheduler.py`** — Fixed broken `_call_prime_with_adapter()`: replaced `getattr(self.model_pool, "_primary_model", None)` with `self.model_pool.models.get("gpu_prime") or self.model_pool.models.get("prime")`.

4. **`vllm_remote_model.py`** — Added LoRA-specific 400 error handling in `_post()`. Detects adapter rejection, logs warning, clears `_active_adapter`, retries with base model. Prevents missing/incompatible adapters from breaking generation.

5. **`main.py`** — Startup auto-registration: checks for adapter on disk, attempts to register with vLLM via API. Graceful fallback if adapter doesn't exist or API unavailable.

### Verification: pipeline wiring works

- Adapter loads in vLLM via `--lora-modules` flag
- `_resolve_adapter()` correctly returns adapter name for Prime, `None` for Lite/Nano/API
- Graceful degradation confirmed: no adapter on disk → bare model, adapter rejected → retry without

---

## Critical Discovery: bf16-Trained Adapters Are Incompatible with AWQ Inference

### The problem

The `gaia_persona_v1` adapter was trained via QLoRA on the bf16 base model (`Qwen3-8B-abliterated`) using BnB NF4 quantization. vLLM serves the AWQ-quantized version (`Qwen3-8B-abliterated-AWQ`).

**A/B test results:**
- **Base model (AWQ, no adapter)**: Clean, coherent identity response
- **With adapter on AWQ**: Degenerate output — garbage tokens, CJK characters, infinite repetition loops

### Root cause

LoRA adapters are weight deltas trained against a specific numerical space. bf16 weights and AWQ-dequantized weights occupy different numerical distributions. The LoRA offsets learned against bf16 activations produce meaningless corrections when applied to AWQ-dequantized activations. This is not a metadata issue — it's a fundamental weight-space mismatch.

### What doesn't work
- Training on AWQ directly: AWQ GEMM kernels don't implement backward pass (no gradient flow). The trainer already correctly rejects this (line 215 of `qlora_trainer.py`).
- Patching `base_model_name_or_path`: vLLM doesn't actually check this field — the shapes match, and the adapter loads. But the outputs are garbage because the numerical spaces differ.

### Additional discovery: mount path mismatch

gaia-prime mounts `/mnt/gaia_warm_pool` → `/models` (read-only). gaia-core and gaia-study mount `/gaia/GAIA_Project/gaia-models` → `/models`. The adapter existed in `gaia-models` but not in `gaia_warm_pool`. Docker doesn't follow host symlinks inside bind mounts. Fixed by copying adapter files to the warm pool. For Phase 3, the merge pipeline should output to both locations (or unify the mount).

---

## Phase 3 Plan: Identity Baking & Dual-Mode Serving

### Architecture overview

Instead of runtime LoRA adapters for identity, **merge the adapter into the base weights** and re-quantize. The persona becomes part of the model itself. Runtime LoRA is reserved for topic-specific swappable adapters via an optional bf16+BnB serving mode.

### The merge+fan-out pipeline

```
Curriculum data (gaia_persona_training.jsonl)
      │
      ├─── Train LoRA on Qwen3-8B bf16 base (QLoRA, BnB NF4, ~5min GPU)
      │         │
      │      LoRA adapter (15MB)
      │         │
      │      merge_and_unload()  ← folds deltas into base weights in RAM
      │         │
      │      Merged bf16 model (~16GB)  ← new training base for future adapters
      │         │
      │    ┌────┴─────────┐
      │    ▼              ▼
      │  AWQ quantize   GGUF quantize
      │  (GPU, ~10min)  (CPU, ~5min)
      │    │              │
      │    ▼              ▼
      │  Prime AWQ      Core/Lite GGUF Q4_K_M
      │  (vLLM, GPU)   (llama_cpp, CPU)
      │
      ├─── Train LoRA on Qwen3-0.6B bf16 base (separate, same curriculum)
      │         │
      │      merge + GGUF quantize
      │         │
      │         ▼
      │      Nano GGUF (triage classifier with identity)
      │
      └─── All tiers speak with same identity knowledge
```

### Dual-mode serving

```
GAIA_PRIME_MODE=awq        → Merged+requantized AWQ, fast, no runtime LoRA
GAIA_PRIME_MODE=bnb_lora   → Merged bf16 + BnB NF4, slower, hot-swappable adapters
```

**Production (AWQ)**: Identity baked into weights. No adapter overhead. Maximum inference speed. Used for normal operation.

**Adapter-dev (BnB)**: Merged bf16 base (identity already baked in) loaded with BnB NF4 quantization. Swappable topic adapters (code-architect, worldbuilding, etc.) can be loaded/unloaded at runtime. ~2-3x slower than AWQ but enables live adapter iteration.

### Key insight: merged bf16 as the new training base

The merged bf16 model (with persona baked in) becomes the base for ALL future adapter training. When training a code-architect or worldbuilding adapter, it's trained on top of the identity-aware base. Every adapter inherits identity knowledge automatically. No mode ever runs without self-knowledge.

### Training data curation principles

The persona adapter curriculum must be **surgically scoped**:
- DO: "In the context of GAIA the AI system, the cognitive pipeline has 20 stages..."
- DO: "GAIA's architecture consists of 11 services: gaia-core, gaia-web, gaia-prime..."
- DON'T: "Gaia is a sovereign AI" (pollutes the mythological concept)
- DON'T: Redefine Python syntax, general coding patterns, or common knowledge

The adapter teaches **contextual self-knowledge** — facts about GAIA's architecture, identity, and capabilities that are true within the GAIA system context. Not concept redefinition.

### Nano model upgrade opportunity

Current Nano: Qwen2.5-0.5B (different architecture family from Prime/Core).

Options for upgrade:
- **Qwen3-0.6B** — same architecture family as Prime, same tokenizer, better quality
- **Qwen3.5-0.8B** — released 2026-03-02, designed for on-device, latest improvements

Either enables same-family adapter training and better triage classification quality.

### Memory/resource plan for merge pipeline

1. **Training** (gaia-study, GPU): Load bf16 base with BnB NF4 (~5GB VRAM). Train LoRA. Save adapter.
2. **Merge** (gaia-study or host, CPU): Load bf16 base in system RAM (~16GB of 64GB available). Merge adapter. Save merged bf16.
3. **AWQ quantize** (gaia-study, GPU): Stop gaia-prime first (free GPU). Load merged bf16, run AutoAWQ calibration. Save new AWQ. ~10 min.
4. **GGUF quantize** (host, CPU): Run llama.cpp convert + quantize on merged bf16. Save GGUF Q4_K_M. ~5 min.
5. **Restart**: Deploy new AWQ to gaia-prime, new GGUF to llama_cpp models dir. Restart services.

Total pipeline: ~20-30 minutes, mostly automated. Can run as a sleep task.

### Model selection rationale

Qwen3-8B remains the right choice for Prime/Core:
- Top of class at 8B tier (trades benchmark leads with Llama 3.1-8B)
- Unique thinking/non-thinking mode switching (critical for GAIA's cascade)
- Abliterated variant removes refusals (necessary for sovereign operation)
- Strong multilingual and reasoning capabilities
- Active development (Qwen3.5 series just released)

No benefit to switching model families. The adapter/training infrastructure we're building is Qwen-native.

---

## Implementation priority

1. **Curriculum curation** — Review and refine `gaia_persona_training.jsonl` for surgical scoping
2. **Merge pipeline** — `merge_and_requantize.py` script in gaia-study
3. **GGUF fan-out** — Add llama.cpp quantization step to pipeline
4. **Dual-mode docker-compose** — `GAIA_PRIME_MODE` env var switching AWQ vs bf16+BnB
5. **Nano upgrade** — Evaluate Qwen3-0.6B / Qwen3.5-0.8B as Nano replacement
6. **Sleep task integration** — Wire merge pipeline into autonomous sleep cycle

---

---

## DECISION: Qwen3.5 Abliterated Trio — Official Model Lineup

### The new cascade

| Tier | Model | Params | Architecture | Serving | Quantization | Vision |
|------|-------|--------|-------------|---------|-------------|--------|
| **Nano** | Qwen3.5-0.8B-Abliterated | 0.8B | Qwen3_5ForConditionalGeneration | llama_cpp (CPU) | GGUF Q8_0 | Yes |
| **Core/Lite** | Qwen3.5-4B-Abliterated | 4B | Qwen3_5ForConditionalGeneration | llama_cpp (CPU) | GGUF Q4_K_M | Yes |
| **Prime** | Qwen3.5-9B-Abliterated | 9B | Qwen3_5ForCausalLM | vLLM (GPU) | AWQ | Yes |

### What changes from current lineup

| Tier | Old | New | Delta |
|------|-----|-----|-------|
| Nano | Qwen2.5-0.5B (text-only, different family) | Qwen3.5-0.8B (multimodal, same family) | +vision, same family, better quality |
| Core | Qwen3-8B-abliterated Q4_K_M (text-only) | Qwen3.5-4B Q4_K_M (multimodal) | +vision, smaller/faster, newer arch |
| Prime | Qwen3-8B-abliterated AWQ (text-only) | Qwen3.5-9B AWQ (multimodal) | +vision, +1B params, newer arch |

### Rationale

- **Same family**: All Qwen3.5, same tokenizer (vocab 248320), same training pipeline
- **All abliterated**: No refusal guardrails to fight against
- **All multimodal**: GAIA gains vision at every tier (early fusion architecture)
- **Identity baking**: One curriculum, three merge+quantize fan-outs
- **4B as Core**: The old Core was 8B Q4_K_M running on CPU — 4B Q4_K_M will be significantly faster on CPU while being architecturally newer. The 9B Prime handles the heavy lifting on GPU.

### Identity baking pipeline (finalized)

```
Same curriculum (gaia_persona_training.jsonl)
    │
    ├── Train LoRA on Qwen3.5-9B bf16    ──→ merge ──→ AWQ (Prime)
    ├── Train LoRA on Qwen3.5-4B bf16    ──→ merge ──→ GGUF Q4_K_M (Core)
    └── Train LoRA on Qwen3.5-0.8B bf16  ──→ merge ──→ GGUF Q8_0 (Nano)
```

### Dual-mode serving (Prime only)

```
GAIA_PRIME_MODE=awq       → Merged 9B AWQ, fast, baked identity, no runtime LoRA
GAIA_PRIME_MODE=bnb_lora  → Merged 9B bf16 + BnB NF4, hot-swappable topic adapters
```

---

## Quantization Toolchain Added to gaia-study

### Dependencies added

**requirements.txt**:
- `autoawq>=0.2.0` — bf16 → AWQ quantization (GPU accelerated)
- `gguf>=0.6.0` — HuggingFace → GGUF format conversion library

**Dockerfile** (new build layers):
- `autoawq` installed with CUDA 12.0 arch (Blackwell/RTX 5080)
- `llama.cpp` cloned and built from source with CUDA support:
  - `llama-quantize` binary → `/usr/local/bin/` (GGUF quantization: F16→Q4_K_M, Q8_0, etc.)
  - `convert_hf_to_gguf.py` remains in `/opt/llama.cpp/` with PYTHONPATH set
  - `cmake` + `build-essential` added as build deps
- Build artifacts cleaned, source kept for convert scripts

### Available tools after rebuild

| Tool | Location | Purpose |
|------|----------|---------|
| `autoawq` | Python import | `model.quantize()` for bf16 → AWQ |
| `llama-quantize` | `/usr/local/bin/` | GGUF quantization (e.g., `llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M`) |
| `convert_hf_to_gguf.py` | `/opt/llama.cpp/` | HuggingFace safetensors → GGUF F16 conversion |

### Usage (merge+requantize pipeline)

```bash
# Inside gaia-study container:

# 1. Merge LoRA into bf16 base (Python, ~16GB RAM)
python -c "from peft import PeftModel; ..."  # merge_and_unload()

# 2. AWQ quantize (GPU, ~10min)
python -c "from awq import AutoAWQForCausalLM; ..."  # model.quantize()

# 3. GGUF convert + quantize (CPU, ~5min)
python /opt/llama.cpp/convert_hf_to_gguf.py /models/merged_bf16/ --outfile /models/model-f16.gguf
llama-quantize /models/model-f16.gguf /models/model-q4_k_m.gguf Q4_K_M
```

---

## Candidate Configs Updated for Qwen3.5 Trio

### Files modified (candidates only — production unchanged until quantized models exist)

1. **`candidates/gaia-common/gaia_common/constants/gaia_constants.json`**:
   - `MODEL_CONFIGS.thinker.path`: `/models/Qwen3-8B-abliterated-AWQ` → `/models/Qwen3.5-9B-Abliterated-AWQ`
   - `MODEL_CONFIGS.core.path`: `/models/Qwen3-8B-abliterated-Q4_K_M.gguf` → `/models/Qwen3.5-4B-Abliterated-Q4_K_M.gguf`
   - `MODEL_CONFIGS.reflex.path`: `/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` → `/models/Qwen3.5-0.8B-Abliterated-Q8_0.gguf`
   - `STUDY_MODE.base_model_path`: `/models/Qwen3-8B-abliterated` → `/models/Qwen3.5-9B-Abliterated`

2. **`candidates/gaia-core/gaia_core/models/_model_pool_impl.py`**:
   - Fallback default path updated to `Qwen3.5-9B-Abliterated-AWQ`

3. **`candidates/gaia-core/gaia_core/models/vllm_remote_model.py`**:
   - Fallback default + docstring updated to `Qwen3.5-9B-Abliterated-AWQ`
   - Comment updated: "Qwen3/3.5 thinking mode"

4. **`candidates/gaia-study/gaia_study/server.py`**:
   - Fallback default base model path updated to `Qwen3.5-9B-Abliterated`

### What still needs updating (after quantized models exist)

- `docker-compose.yml`: `PRIME_MODEL_PATH`, `PRIME_MODEL`, `GAIA_LITE_GGUF`, `BASE_MODEL_PATH` env defaults
- `docker-compose.yml`: gaia-audio Nano mount (`Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` → new Q8_0)
- Warm pool: copy new AWQ model to `/mnt/gaia_warm_pool/`
- Production `gaia_constants.json`: update after candidate validation

### Blocking issue: warm pool capacity

Warm pool is tmpfs (10GB), 3.8GB free. bf16 9B model is 17GB — can't test through vLLM without expanding tmpfs or switching mount. AWQ model (~5GB) will fit. Testing deferred to post-quantization.

---

## Commits this session

- Phase 2 pipeline wiring (5 files, production + candidate)
- docker-compose.yml: `--lora-modules` tested and reverted (adapter incompatible with AWQ)
- Adapter files copied to `/mnt/gaia_warm_pool/lora_adapters/tier1_global/gaia_persona_v1/` for future use
- Qwen3.5 model trio downloaded: 0.8B, 4B, 9B (all abliterated, bf16)
- Quantization toolchain added to gaia-study (AutoAWQ + llama.cpp)
- Candidate configs updated for Qwen3.5 trio (gaia_constants.json, model pool, study server)

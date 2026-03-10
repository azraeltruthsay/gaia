# Dev Journal: All-Qwen Stack — Model Lineup v3

**Date**: 2026-03-09
**Session focus**: Resolve Qwen3.5-9B VRAM impossibility, redesign model lineup around hardware constraints, adopt Qwen audio models, validate vLLM v0.17.0 with FP8

---

## Critical Finding: Qwen3.5-9B Does NOT Fit in 16GB VRAM

### The math (from previous session, captured here for the record)

The Qwen3.5-9B has a 248K vocabulary with **untied embeddings** (`tie_word_embeddings: False`). This means:

| Component | Size |
|-----------|------|
| Model body (quantized GPTQ 4-bit) | ~12.7GB |
| embed_tokens (bf16, unquantizable) | ~1.9GB |
| lm_head (bf16, separate, unquantizable) | ~1.9GB |
| **Total model weight** | **~14.6GB** |
| RTX 5080 total VRAM | 15.46GB |
| **Remaining for KV cache** | **0-1GB** |

Not viable for any real workload. The 248K vocab with untied embeddings eats 3.8GB that no quantization method can touch — 25% of VRAM just for embeddings.

Additionally, the GPTQ quantization of the 9B OOM'd at the 48GB container limit (GPTQModel needs full bf16 model + Hessian buffers in RAM).

### Options considered

1. **Qwen3.5-4B as Prime** — fits, capability downgrade but newer arch ✅ CHOSEN
2. Wait for smaller-vocab variant — unlikely, 248K is a family design choice
3. vLLM CPU offload for embeddings — saves 3.8GB but hacky
4. FP8 quantization — still can't touch embeddings

---

## Decision: Model Lineup v3 — "All-Qwen Stack"

### Cognition (Qwen3.5 Abliterated, same family/tokenizer)

| Tier | Model | Params | Serving | Quantization | Role |
|------|-------|--------|---------|-------------|------|
| **Nano** | Qwen3.5-0.8B-Abliterated | 0.8B | llama_cpp (CPU) | GGUF Q8_0 | Triage classifier |
| **Core/Operator** | Qwen3.5-4B-Abliterated | 4B | llama_cpp (CPU) | GGUF Q4_K_M | Intent detection, tool selection, fast answers |
| **Prime/Thinker** | Qwen3.5-4B-Abliterated | 4B | vLLM (GPU) | FP8 (on-the-fly) | Complex reasoning, streaming, full context |

**Key insight**: Core and Prime are the **same model, different serving tiers**. Escalation from Core→Prime is not "dumber→smarter" but rather CPU→GPU (faster tok/s, larger context). The cascade becomes a pure speed gradient: 0.8B (~instant) → 4B CPU (~fast) → 4B GPU (~thorough).

**Cognitive continuity during speech**: When Prime sleeps for TTS GPU swap, Core (same 4B on CPU) continues processing. GAIA doesn't go deaf while speaking.

### Audio (Qwen3 series, replaces Whisper + Coqui)

| Component | Old | New | Size | Improvement |
|-----------|-----|-----|------|-------------|
| **STT** | Whisper large-v3 (1.5B) | Qwen3-ASR-0.6B | ~1.8GB | 97.9% language ID (vs 94.1%), competitive WER |
| **TTS** | Coqui XTTS v2 (~450M) | Qwen3-TTS-12Hz-1.7B-Base | ~4.3GB | Stable long-form (2 pauses vs 106), voice cloning built-in, beats ElevenLabs |

**Why 1.7B TTS over 0.6B**: Long-form stability is dramatically different. On the same narration test, 0.6B produced 106 pauses >1.5s while 1.7B produced 2. GAIA gives multi-sentence spoken responses — the 0.6B falls apart.

**Why 0.6B ASR is fine**: STT is just transcribing audio → text. The 0.6B can transcribe 2000 seconds of speech in 1 second at concurrency 128. More than sufficient.

### Cloud fallbacks (unchanged)

- Oracle: GPT-4o-mini
- Groq: Llama-3.3-70b-versatile

---

## Architecture Properties

### Qwen3.5-4B characteristics

- **Hybrid Mamba+Attention**: 24 linear attention + 8 full attention layers (same ratio as 9B)
- **Tied word embeddings**: `tie_word_embeddings: True` — no 3.8GB embedding tax!
- **Multimodal**: `Qwen3_5ForConditionalGeneration` — vision capable at every tier (text-only mode via `--language-model-only`)
- **248K vocab**: Same tokenizer across all Qwen3.5 models
- **Hidden size**: 2560, intermediate: 9216, 16 attention heads

### FP8 quantization (no pre-processing needed!)

vLLM v0.17.0 supports on-the-fly FP8 quantization via `--quantization fp8`. All Linear layers quantized to FP8_E4M3 at load time with per-tensor scale. No calibration data, no GPTQ/AWQ toolchain headaches. The hybrid Mamba+Attention architecture that broke GPTQModel is handled natively by vLLM.

**Why not GPTQ/AWQ**: Both failed on Qwen3.5-4B:
- GPTQModel 5.7.0 doesn't support `qwen3_5` model type (multimodal wrapper `Qwen3_5ForConditionalGeneration` has nested config that GPTQModel can't parse)
- Upgrading transformers to 5.3.0 for GPTQ caused `vocab_size` attribute error (nested in `text_config`, not top-level)
- AWQ has same architecture compatibility issues
- FP8 on-the-fly bypasses all of this — vLLM handles the architecture natively

### GPU time-sharing architecture (Prime ↔ TTS swap)

Prime and TTS **cannot coexist** on the 16GB GPU. Instead, they time-share:

```
User speaks → ASR (CPU) → text
  → Prime wakes (if sleeping), reasons at full GPU speed
  → Prime sleeps (VRAM → RAM via vLLM sleep mode)
  → TTS loads on GPU, generates speech with full VRAM
  → TTS unloads
  → Prime wakes (RAM → VRAM, KV cache restored via LMCache)
  → Ready for next turn

Meanwhile: Core (4B on CPU) handles any incoming queries — same intelligence, slightly slower
```

### VRAM budget (validated)

| Config | Model | KV cache | Total | Context |
|--------|-------|----------|-------|---------|
| **0.65 util** | 4.64 GiB | 5.14 GiB (42K tokens) | ~10.6 GiB | 24K max ✓ |
| **0.70 util** | 4.64 GiB | 5.92 GiB (48K tokens) | ~11.4 GiB | 24K max ✓ |
| **0.80 util** | 4.64 GiB | 7.47 GiB (61K tokens) | ~12.5 GiB | 24K max ✗ OOM on inference |

**Selected: 0.65 utilization, 24K context** — conservative, proven stable, leaves headroom.

### QLoRA pipeline preserved

Train on **two** base models (0.8B + 4B) instead of three. One curriculum, fewer fan-outs:

```
Curriculum (gaia_persona_training.jsonl)
    │
    ├── Train LoRA on Qwen3.5-4B bf16  ──→ merge ──→ new bf16 base (identity baked)
    │                                       ├──→ serve with FP8 on-the-fly (Prime GPU)
    │                                       └──→ GGUF Q4_K_M (Core CPU)
    │
    └── Train LoRA on Qwen3.5-0.8B bf16 ──→ merge ──→ GGUF Q8_0 (Nano CPU)
```

### Dynamic adapters still viable

With FP8 serving at 0.65 utilization, ~5GB VRAM remains for KV cache + potential LoRA adapters. Phase 2 adapter pipeline (agent_core, model_pool, vllm_remote_model) works as-is for topic-specific swappable adapters.

### Compressed knowledge via QLoRA adapters (new idea)

Three tiers of knowledge density:
1. **Baked weights** (QLoRA merged) — model *is* the persona. Replaces pages of system prompt.
2. **Compressed semantic codex per adapter** — domain-specific adapters (architect, worldbuilder, coder) that understand compressed references without spelling them out.
3. **Dynamic prompt** — freed from basics, carries only novel context: current task, user intent.

This massively compresses prompt tokens while improving consistency.

---

## Validated: vLLM v0.17.0 with Pre-built Image

### What was tested

Used `vllm/vllm-openai:latest` (v0.17.0, released March 7 2026) as `gaia-prime-candidate` on port 7778. No custom Dockerfile, no patches.

### Test results

| Test | Result |
|------|--------|
| Model load from tmpfs | **1.1 seconds** |
| FP8 quantization | **Working** — CutlassFP8ScaledMMLinearKernel |
| `--language-model-only` | **Working** — skips vision encoder |
| `--quantization fp8` | **Working** — 4.64 GiB model weight |
| Inference (short) | **Working** — coherent, thinking mode active |
| Inference (500 tokens) | **Working** — stable, no OOM |
| Health endpoint | **Working** — `:7778/health` |
| 0.65 utilization + 24K context | **Working** — 42K token KV cache, stable |
| 0.80 utilization + 24K context | **OOM** — Triton out of memory on first inference |

### Why pre-built image works (no custom build needed)

Production gaia-prime was built from source (vLLM v0.15.1 on NGC PyTorch 25.03) with Float8 compatibility patches for Blackwell. The pre-built `vllm/vllm-openai:latest` v0.17.0 includes native Blackwell support, no patches needed. This eliminates the custom Dockerfile entirely.

### Previous vLLM (v0.15.1) couldn't serve Qwen3.5

vLLM 0.15.1 (bundled transformers 4.57.6) doesn't recognize `qwen3_5` model type. Qwen3.5 support was added in vLLM v0.17.0 via PR #34110. Upgrading was mandatory.

---

## Downloads & Warm Pool

### Downloaded this session

| Model | Location | Size |
|-------|----------|------|
| Qwen3-ASR-0.6B | `gaia-models/Qwen3-ASR-0.6B/` | 1.8GB |
| Qwen3-TTS-12Hz-1.7B-Base | `gaia-models/Qwen3-TTS-12Hz-1.7B-Base/` | 4.3GB |

### Warm pool populated (tmpfs, 20GB)

| Model | Size | Role |
|-------|------|------|
| `Qwen3.5-4B-Abliterated/` | 9.2GB | Prime bf16 (FP8 on-the-fly via vLLM) |
| `Qwen3.5-4B-Abliterated-Q4_K_M.gguf` | 2.6GB | Core (CPU, llama_cpp) |
| `Qwen3.5-0.8B-Abliterated-Q8_0.gguf` | 775MB | Nano (CPU, llama_cpp) |
| `Qwen3-ASR-0.6B/` | 1.8GB | STT (replaces Whisper) |
| `Qwen3-TTS-12Hz-1.7B-Base/` | 4.3GB | TTS (replaces Coqui) |
| **Total** | **18.7GB** | 1.3GB headroom |

Cleared the failed 9B GPTQ (14GB) to make room.

---

## Quantization Attempts (for the record)

### GPTQ via GPTQModel — FAILED

1. **transformers 4.57.6**: `KeyError: 'qwen3_5'` — model type not recognized
2. **trust_remote_code=True**: Same error — GPTQModel doesn't pass flag to `AutoConfig.from_pretrained`
3. **Upgraded transformers to 5.3.0**: `AttributeError: 'Qwen3_5Config' object has no attribute 'vocab_size'` — multimodal wrapper has nested config, GPTQModel expects flat
4. **Root cause**: GPTQModel 5.7.0 doesn't support `Qwen3_5ForConditionalGeneration` architecture

### AWQ via AutoAWQ — NOT ATTEMPTED

Same architecture incompatibility expected. Community `cyankiwi/Qwen3.5-4B-AWQ-4bit` exists but uses `compressed-tensors` format (not real AWQ) and is from non-abliterated base.

### FP8 via vLLM — SUCCESS

`--quantization fp8` on vLLM v0.17.0. No pre-processing. Model loads in 1.1s, 4.64 GiB VRAM. This is the path forward.

---

## Fixes This Session

### claude-mem MCP plugin failure

- **Symptom**: `chroma-mcp connection in backoff` errors on all search calls
- **Root cause**: Worker daemon couldn't find `claude` CLI binary. `CLAUDE_CODE_PATH` was empty in `~/.claude-mem/settings.json`, and the bun daemon's PATH didn't include `~/.local/bin/`
- **Fix**: Set `CLAUDE_CODE_PATH` to `/home/azrael/.local/bin/claude`, killed worker daemon (PID from `~/.claude-mem/worker.pid`), respawned automatically

### Production docker-compose.yml partially modified

- Changed `PRIME_MODEL_PATH` default from `Qwen3-8B-abliterated-AWQ` to `Qwen3.5-4B-Abliterated`
- Added `--quantization fp8`, removed `--enable-lora` flags
- **NOTE**: This was for testing only. Revert before production use — production gaia-prime should use the pre-built image approach, not the old custom-built one.

---

## What's Next (Prioritized)

### Immediate (next session)

1. **LMCache integration** — Hierarchical KV cache persistence (GPU → CPU → disk). Install into vLLM container, configure YAML, set disk persistence path. Preserves conversation context across Prime sleep/wake cycles.
2. **Wake comparison logic** — On Prime wake, compare LMCache KV state against Prime.md semantic summary. KV cache = exact computational state, Prime.md = human-readable backup. Detect drift/corruption.
3. **Production gaia-prime image switch** — Replace custom-built vLLM v0.15.1 with `vllm/vllm-openai:latest` (v0.17.0). Update docker-compose.yml to use pre-built image instead of building from source.
4. **Update candidate configs** — gaia_constants.json (model paths), _model_pool_impl.py (fallback defaults), vllm_remote_model.py (model name), docker-compose.yml (env vars)

### Short-term

5. **TTS/Prime GPU swap orchestration** — Wire vLLM `/sleep` and `/wake_up` endpoints into gaia-audio pipeline. Orchestrator manages the handoff.
6. **Integrate Qwen3-ASR** into gaia-audio (replace Whisper STT)
7. **Integrate Qwen3-TTS** into gaia-audio (replace Coqui XTTS v2)
8. **QLoRA identity baking** — Train on 4B + 0.8B bases with persona curriculum, merge, deploy as new bf16 bases

### Medium-term

9. **Per-persona QLoRA adapters** — Train domain-specific adapters (architect, worldbuilder, coder) on identity-baked base. Hot-swap via vLLM LoRA serving.
10. **Qwen3.5 vision enablement** — Remove `--language-model-only` flag, test multimodal input at all tiers
11. **Dynamic adapter testing** — Verify LoRA hot-swap works on FP8 model in vLLM v0.17.0

---

## Key Learnings

1. **Vocabulary size matters more than model size** for VRAM budgeting. The 9B's 248K untied vocab was the killer, not the parameter count.
2. **FP8 on-the-fly beats pre-quantization** for hybrid architectures. No toolchain compatibility issues, competitive quality, trivial deployment.
3. **Same model at two serving tiers** (CPU + GPU) is a powerful pattern. Cognitive continuity during GPU handoff for free.
4. **Pre-built vLLM images work** for Blackwell. The custom build with Float8 patches is no longer necessary.
5. **Warm pool (tmpfs)** enables 1.1s model loads. Critical for the sleep/wake swap pattern.

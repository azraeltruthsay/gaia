# 2026-04-09 — Phase 5: The Gemma Chord — Engine Compatibility Plan

## Context
Architect is pivoting to a pure Gemma 4 ecosystem. Three models downloaded:
- E2B (2.3B effective, 128K context, native audio)
- E4B (4.5B effective, 128K context, native vision)
- 26B-A4B MoE (4B active / 26B total, 256K context, 128 experts Top-8)

## Tier Mapping
| Current (Qwen) | Proposed (Gemma) | Gains |
|-----------------|------------------|-------|
| Nano 0.8B (Reflex) | E2B 2.3B (Reflex + audio) | Native STT/TTS, eliminates gaia-audio |
| Core 4B (Operator) | E4B 4.5B (Operator + vision) | Native vision, GUI understanding |
| Prime 9B (Thinker) | 26B-A4B MoE (Sovereign) | 256K ctx, native tools, only 4B active VRAM |

## GAIA Engine Changes Required

### CRITICAL: Chat Template (est. 30 min)

Replace hardcoded ChatML (`<|im_start|>/<|im_end|>`) with dynamic `tokenizer.apply_chat_template()`.

**Files to modify:**
- `gaia-engine/gaia_engine/core.py` — Lines 318, 1055, 1067-1070, 1127-1137, 1322-1325
- `gaia-engine/gaia_engine/cpp/backend.py` — Lines 340-357 (`_build_prompt()`)
- `gaia-engine/gaia_engine/adapter_surgeon.py` — Line 128

**Approach:** Create a `format_messages()` helper that delegates to `tokenizer.apply_chat_template()` with fallback to manual ChatML for legacy models without templates.

**Gemma 4 template format:** `<|turn>role<turn|>` (from Gemini's research)

### HIGH: Thinking Tokens (est. 15 min)

Make `<think>` token injection conditional on vocab presence.

- Qwen3/3.5: `<think>`/`</think>`
- Gemma 4: `<|think|>` (enabled via system prompt, different syntax)
- Fix: Check tokenizer vocab, inject appropriate format or skip

**Files:** core.py:1068-1069, 1134-1137, 1174-1183; cpp/backend.py:346, 356

### LOW: EOS Token Fallback (est. 5 min)

- `core.py:1329` — hardcoded `151643` (Qwen EOS)
- Fix: Remove fallback, trust `tokenizer.eos_token_id`

## Gemma 4 Architecture Notes (from Gemini's research)

- **MoE:** 128 experts, Top-8 routing per token
- **Context:** 256K via Dual RoPE (standard + proportional for global layers)
- **Attention:** Alternating sliding-window (512-1024) + global full-context
- **Shared KV Cache:** Last N layers reuse KV states from earlier layers
- **Vision:** Learned 2D positions, multidimensional RoPE, configurable token budgets (70-1120)
- **Audio:** USM-style conformer encoder (E2B/E4B only, NOT 26B)
- **Tool Protocol:** 6 special tokens + `<|'|>` string wrappers
- **Thinking:** `<|think|>` in system prompt enables reasoning mode

## What Already Works (No Changes Needed)

- Tokenizer loading (AutoTokenizer)
- Vision detection and processor loading
- KV cache architecture detection (no hybrid for Gemma = standard cache)
- LoRA/PEFT adapter loading
- Quantization (NF4, GPTQ, AWQ)
- ROME MLP path search (Gemma uses same LLaMA-style structure)
- SAE trainer layer selection
- Model size estimation
- Thought composer (safe fallback for non-hybrid caches)

## VRAM Estimates (16GB GPU)

| Model | BF16 | NF4 | Q4_K_M GGUF | Fits 16GB? |
|-------|------|-----|-------------|-----------|
| E2B (2.3B) | ~4.6GB | ~1.5GB | ~1.4GB | YES (GPU) |
| E4B (4.5B) | ~9GB | ~3GB | ~2.7GB | YES (GPU) |
| 26B-A4B (4B active) | ~52GB total | ~15GB? | ~13GB? | NEEDS TESTING |

The 26B MoE is the question mark — even though only 4B params are active per token, the full 26B must be in memory for expert routing. Q4 quantization should bring it to ~13-15GB. Tight but potentially viable.

## Implementation Order

1. Chat template helper function (covers all 3 files)
2. Thinking token conditional injection
3. EOS fallback cleanup
4. Test with E2B first (smallest, fastest validation)
5. Test with E4B (vision capabilities)
6. Test with 26B-A4B (MoE VRAM benchmark)
7. Identity bake feasibility on Gemma architecture

## Open Questions

- Does LoRA on MoE work correctly? (experts vs shared params)
- Can we identity-bake Gemma without fighting Google's alignment?
- Does `apply_chat_template` work correctly for the GGUF/llama-server path?
- Shared KV cache — can we leverage this for our prefix cache optimization?

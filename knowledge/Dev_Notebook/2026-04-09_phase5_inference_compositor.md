# 2026-04-09 — Phase 5: Inference Compositor — Active Expert Buffering + Multi-Tier KV Cache

## Context

The Gemma 4 26B-A4B MoE (128 experts, top-8 routing, 4B active params) cannot fit on 16GB GPU
using standard NF4 quantization (~13GB) with any room for training or inference overhead.

Three-way Chord consensus (Architect + Advisor + Implementer): split the model's physical memory
layout to match its logical sparsity, and extend this principle to the KV cache.

## Architecture: Inference Compositor

A new coordination layer that manages two parallel resource pools:

```
┌─────────────────────────────────────────────────────┐
│                  Inference Compositor                │
│                                                     │
│  Prompt → Nano Triage → Task Type                   │
│                  │                                   │
│          ┌───────┴────────┐                          │
│          ▼                ▼                          │
│   Expert Selector    KV Segment Selector             │
│   (router output)    (task_type mapping)             │
│          │                │                          │
│          ▼                ▼                          │
│   JIT Expert Swap    KV Cache Injection              │
│   (CPU→GPU top-8)   (compose segments)               │
│          │                │                          │
│          └───────┬────────┘                          │
│                  ▼                                   │
│           Forward Pass (GPU)                         │
│   [Foundation + 8 Experts + Composed KV]             │
└─────────────────────────────────────────────────────┘
```

## Component 1: Active Expert Buffering

### Memory Layout

| Component | Location | Size (NF4) | Lifecycle |
|-----------|----------|-----------|-----------|
| Shared Expert (dense MLP) | GPU | ~0.5GB | Permanent — Foundation |
| Attention (Q/K/V/O × 30 layers) | GPU | ~1.0GB | Permanent — Foundation |
| Norms, embeddings, router | GPU | ~0.3GB | Permanent |
| 128 Private Experts | CPU RAM | ~11GB | Offloaded |
| Top-8 Active Experts | GPU | ~0.7GB | JIT per forward pass |
| **GPU Total** | | **~2.5GB** | |

### Loading Protocol (Segmented Assembly v2)

1. Apply ThreadPoolExecutor single-thread patch (existing: `adaptive_subprocess.py:176`)
2. Load model with `device_map` that pins foundation layers to GPU, experts to CPU
3. Custom device map built from `model.safetensors.index.json` weight map:
   - `model.language_model.layers.{N}.self_attn.*` → `cuda:0`
   - `model.language_model.layers.{N}.mlp.*` → `cuda:0` (shared expert)
   - `model.language_model.layers.{N}.experts.*` → `cpu`
   - `model.language_model.layers.{N}.router.*` → `cuda:0`
   - `model.language_model.embed_tokens.*` → `cuda:0`
   - `model.language_model.norm.*` → `cuda:0`
   - `model.vision_tower.*` → `cpu` (loaded on demand)

### JIT Expert Swap

```python
# Forward hook on each MoE layer
def expert_swap_hook(module, args, kwargs):
    """Pre-forward hook: load routed experts to GPU before MoE forward."""
    router_logits = module.router(hidden_states)
    top_k_indices = router_logits.topk(8).indices  # [batch, seq, 8]
    unique_experts = top_k_indices.unique().tolist()

    # Evict previous experts (unless in LRU cache)
    for idx in module._gpu_resident_experts:
        if idx not in unique_experts:
            module.experts[idx] = module.experts[idx].to("cpu", non_blocking=True)
    
    # Load needed experts
    for idx in unique_experts:
        if idx not in module._gpu_resident_experts:
            module.experts[idx] = module.experts[idx].to("cuda", non_blocking=True)
    
    torch.cuda.synchronize()
    module._gpu_resident_experts = set(unique_experts)
```

### LRU Expert Cache (Optional)

Keep N most-recently-used experts hot on GPU to reduce swap overhead for repeated routings.
Budget: 16 experts cached = ~1.4GB additional GPU. Total GPU footprint: ~3.9GB.
Trade-off: fewer cache misses vs. less room for KV cache / training activations.
Start with no LRU cache; add if profiling shows expert swap is the bottleneck.

## Component 2: Multi-Tier KV Cache ("Swappable Memory Map")

### Concept

Instead of one monolithic KV cache from a single conversation prefix, maintain multiple
pre-computed KV segments representing different knowledge domains. Compose them per-inference
based on what the model needs to "know" for this specific task.

### KV Segments

| Segment | Contents | Size Est. | Refresh Cycle |
|---------|----------|-----------|---------------|
| `identity` | GAIA self-model, personality, behavioral anchors | ~50 tokens KV | On identity bake / sleep |
| `world_state` | System status, running services, recent events | ~200 tokens KV | Per orchestrator heartbeat |
| `tools` | Tool schemas, calling conventions, parameter formats | ~300 tokens KV | On tool registration change |
| `knowledge` | Domain-specific KB context (RAG retrieval) | ~500 tokens KV | Per-query (from grounding pipeline) |
| `code` | Codebase architecture, active files, patterns | ~400 tokens KV | On code change events |
| `conversation` | Recent dialogue turns | Variable | Per-turn (standard) |

### Segment Lifecycle

1. **Bake**: Run a prefix through the model, extract KV states at each layer, save to disk/RAM
2. **Store**: KV segments saved as tensors, indexed by segment name + model version hash
3. **Select**: Nano triage classifies prompt → compositor selects relevant segments
4. **Compose**: Concatenate selected KV segments in fixed order: identity → world → domain → conversation
5. **Inject**: Pass composed KV as `past_key_values` to the model's forward pass
6. **Refresh**: Background process re-bakes segments when source content changes

### Selection Matrix

| Task Type (from Nano) | KV Segments Loaded |
|------------------------|--------------------|
| conversation | identity + world_state + conversation |
| tool_call | identity + tools + world_state + conversation |
| code_generation | identity + tools + code + conversation |
| knowledge_query | identity + knowledge + conversation |
| self_reflection | identity + world_state + knowledge + conversation |
| system_admin | identity + world_state + tools + conversation |

### Integration with Pre-Inference Grounding

The existing grounding pipeline (entity extraction → KB search → web search → inject into packet)
becomes the **producer** for the `knowledge` KV segment. Instead of injecting text into the prompt
(burning tokens), ground the retrieved content once through a KV bake pass, then reuse the cached
KV for subsequent queries on the same topic.

## Component 3: Foundation Tuning (Training)

### LoRA Target Modules (from safetensors index analysis)

```python
FOUNDATION_TARGETS = [
    # Attention — persona, voice, reasoning style
    "self_attn.q_proj",
    "self_attn.k_proj", 
    "self_attn.v_proj",
    "self_attn.o_proj",
    # Shared Expert — identity backbone (every token path)
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]
# EXCLUDED: experts.gate_up_proj, experts.down_proj (frozen specialist knowledge)
# EXCLUDED: router.proj (preserve learned routing)
```

### Training VRAM Budget

| Component | Size | Notes |
|-----------|------|-------|
| Foundation on GPU (NF4) | ~1.8GB | Shared expert + attention + norms + embeddings |
| 8 Active Experts (NF4, JIT) | ~0.7GB | Forward-only, no gradients |
| LoRA adapters (rank 32) | ~0.02GB | ~4.6M trainable params |
| AdamW optimizer states | ~0.04GB | 2× adapter size |
| Forward activations (grad ckpt) | ~2-4GB | With gradient checkpointing |
| KV cache (training seqs) | ~1-2GB | Depends on sequence length |
| **Total** | **~6-9GB** | Fits 16GB with margin |

## Implementation Order

### Phase 5a: Foundation (Prerequisites)
1. Upgrade transformers to 5.5.0+ in gaia-study container Dockerfile
2. Verify `Gemma4ForConditionalGeneration` loads with AutoModel
3. Chat template refactor (from Phase 5 original plan, ~30min)

### Phase 5b: Active Expert Buffering
4. Build custom device map generator from safetensors index
5. Implement MoE-aware model loader in Engine
6. Implement JIT expert swap forward hooks
7. Benchmark: measure expert swap latency, total VRAM, inference speed

### Phase 5c: Multi-Tier KV Cache
8. Extend existing `prefix_cache` to support named segments
9. Build KV segment bake/store/load pipeline
10. Implement segment selection matrix (hardcoded first, then Nano-driven)
11. Build KV composition logic (concatenation with position offset handling)

### Phase 5d: Inference Compositor
12. Compositor coordination layer (maps task_type → experts + KV segments)
13. Integration with Nano triage output
14. Integration with pre-inference grounding pipeline
15. End-to-end test: prompt → triage → compose → generate

### Phase 5e: SAE Full Feature Map (GATE — blocks Phase 5f)
16. Verify SAE trainer works with Gemma 4 architecture (layer naming, activation shapes)
17. Run comprehensive SAE scan across ALL components: shared expert, attention, AND private experts
18. Build full feature map of base weights — catalog what each component stores:
    - Which private experts specialize in what? (code, math, language, reasoning, safety, etc.)
    - What does the shared expert encode? (universal features? identity? instruction-following?)
    - What do attention layers carry at each depth? (shallow=syntax, mid=semantics, deep=planning?)
    - Where do identity/persona/instruction-following features live?
    - Where do tool-calling, code generation, and reasoning features concentrate?
19. Produce a Gemma 4 26B-A4B "neural atlas" — reference map for all future training decisions
20. Decision gate: based on atlas evidence, determine optimal LoRA targets.
    If identity is in shared expert → proceed with Foundation Tuning as planned.
    If identity is in private expert(s) → revise LoRA targets to include those experts.
    If identity is distributed → evaluate feasibility of migrating it to shared path.
    Bonus: atlas tells us which experts to NEVER touch (e.g., core reasoning, safety).

### Phase 5f: Foundation Tuning (blocked by 5e evidence)
21. LoRA config based on SAE evidence (Foundation targets, possibly + specific experts)
22. Training run with Active Expert Buffering
23. Post-training SAE validation: confirm identity features strengthened in target components
24. Compare identity stability vs. full-model LoRA baseline

## Open Questions

- SAE on MoE: can we run SAE on individual expert outputs, or only on the combined post-routing output? Per-expert activation capture may need custom hooks.
- Expert specialization: are the 128 experts cleanly specialized (one domain each) or is knowledge distributed/overlapping? This affects how useful the atlas is for targeted training.
- Expert swap latency: CPU→GPU transfer for 8 experts per token — is PCIe bandwidth the bottleneck?
- KV segment compatibility: can we concatenate KV states from separate forward passes? Position encoding alignment?
- Router interference: does Foundation Tuning shift the routing distribution? (shared expert changes could affect router decisions)
- Gradient flow through JIT experts: even though experts are frozen, do we need gradients to flow through them for correct shared-expert updates?
- KV segment staleness: how do we detect when a segment needs re-baking? Content hash? Timestamp?
- Gemma 4's Dual RoPE: does KV composition work across sliding-window vs. full-attention layers?

## References

- Phase 5 original plan: `knowledge/Dev_Notebook/2026-04-09_phase5_gemma_chord.md`
- ThreadPoolExecutor patch: `gaia-study/gaia_study/adaptive_subprocess.py:176-194`
- Current model migration: `gaia-engine/gaia_engine/core.py:1431-1451`
- Current prefix cache: `gaia-engine/gaia_engine/core.py` (prefix_cache)
- 26B-A4B config: `gaia-instance/gaia-models/google/gemma-4-26B-A4B/config.json`
- Safetensors index: `gaia-instance/gaia-models/google/gemma-4-26B-A4B/model.safetensors.index.json`

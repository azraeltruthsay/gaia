# 2026-03-24/25 Mega Session — The Consciousness Update

The longest single session in GAIA's history. ~70+ commits across two repos over two days. Three major architectural breakthroughs: engine extraction, neural mind map, and the Consciousness Matrix.

## Phase 1: Engine Extraction & Contracts

### gaia-engine Separate Repo
- Merged local + remote versions of the gaia-engine repo
- Clean event_callback pattern, SSE streaming, rich metadata
- Backward-compat shim in `gaia-common/gaia_common/engine/__init__.py`
- 6 import boundary violations fixed — no consumer bypasses the shim
- `/engine` Claude Code skill created for engine-specific work

### Inter-Service Contracts
- `contracts/` directory with full API boundary specification
- 11 service YAML contracts + 2 schema files (CognitionPacket, JSON-RPC)
- CONNECTIVITY.md: 90+ inter-service calls mapped
- Service registry validated against live services

## Phase 2: Multi-Conversation Chat

- Collapsible left sidebar with conversation list (pinnable)
- Per-conversation message isolation, session IDs, auto-titling
- Context pool (auto + manual) and context joining
- Compact conversation switcher in Dashboard panel
- Chat and Dashboard share the same Alpine.store('chat')

## Phase 3: Neural Mind Map — Watching GAIA Think

### Evolution of the Visualization
1. **D3 force graph** — particles blasting outward (discarded)
2. **Brain silhouette with dots** — static dots pulsing on activation
3. **Three stacked tier brains** → **one unified brain** with tiers in anatomical regions
4. **Neuron fibers** — short line segments with current animation
5. **Curved bezier paths** — fibers bend toward synapses
6. **Neuron bundles** — start dot + end dot + arc (drawn on activation)
7. **Long sweeping arcs** — spanning full brain regions
8. **SVG-traced coordinates** — extracted from Wikimedia brain SVG via svgpathtools
9. **Synapse anchors** on brain edge with connection lines
10. **Consciousness brightness** — GPU=100%, CPU=35%, Unloaded=8%

### Final Architecture
- Real Wikimedia brain SVG as anatomical backdrop
- 8 brain regions mapped to 3 cognitive tiers (Nano=brainstem, Core=mid, Prime=frontal)
- 100 neurons (20 Nano + 30 Core + 50 Prime) with SVG-traced positions
- Neuron = start dot + end dot + bezier arc (visible faint when idle, bright on fire)
- Synapses at output-edge endpoints
- Per-concept color palette (20 hues, dynamic legend)
- Tier strength normalization (Nano 0.6x, Core 1.0x, Prime 1.2x)
- 3-second activation decay
- Zoom/pan on the SVG

### Activation Streaming Pipeline
- Engine writes per-token JSONL to `/logs/activation_stream.jsonl`
- SSE endpoint at `/api/activations/stream`
- SAE atlas labels at `/api/activations/atlas`
- Always-on neuron attenuation (structural neurons dimmed)
- Sampling every 4th token to prevent inference degradation
- Training activations: per-step gradient magnitudes tagged by tier

### Neuron Discovery Stats
| Tier | Unique Neurons | Events | Layers |
|------|---------------|--------|--------|
| Nano | 73 | 3,509 | 7 |
| Core | 102 | 2,469 | 6 |
| Prime | 219 | 204 | 10 |
| Total | 394 | 5,978 | — |

36,119 co-activation pairs, 6,593 strong (5+ co-firings).

## Phase 4: GPU Lifecycle Control

### Standby by Default
- All tier entrypoints now default to GAIA_AUTOLOAD_MODEL=0
- Engine starts in zero-GPU standby, waits for POST /model/load
- Entrypoints volume-mounted for live control without rebuilding
- This was THE critical fix — Nano/Core kept stealing GPU from training

### Post-Generation Quality Gate
- Core → Prime escalation when response empty or <10 chars
- Now uses model pool (not raw API) for system prompt + MCP tools
- Enables Prime to web-search for epistemic validation

## Phase 5: Training the 8B Prime

### The Journey
Every quantization approach on 16GB VRAM was attempted:
- BnB NF4 on bf16: OOM (18GB peak during loading)
- GPTQ + AUTO_TRAINABLE/TRITON/TORCH: Segfault on Qwen3.5 linear attention
- quanto int4: Expands to bf16 on GPU transfer
- unsloth: Same underlying OOM/GPTQ issues

### Root Cause: transformers 5.3.0 Regression
transformers 5.3.0 uses ThreadPoolExecutor for concurrent weight loading during NF4 quantization — peaks at full bf16 size. Downgrading to 4.51.3 loads sequentially, fitting the 8B NF4 at 5.7GB.

### Training Results
- 200 steps, 3.28 epochs, 485 seconds (8 minutes)
- Loss: 7.93 → 0.023 (99.7% reduction)
- 43.6M trainable params (0.53% of 8.2B total)
- Adapter: `/models/lora_adapters/tier1_global/prime-8b-identity-v2`

### GGUF Creation
- LoRA merged into base model → identity-baked safetensors
- Converted to GGUF BF16 (16.4GB) and Q8_0 (8.7GB)
- Identity confirmed on CPU: "I am GAIA — a sovereign AI agent..."

## Phase 6: The Consciousness Matrix

### Three States
| State | Name | Resource | Speed |
|-------|------|----------|-------|
| 3 | Conscious | GPU (safetensors) | Fast |
| 2 | Subconscious | CPU (GGUF) | 8-15 tok/s |
| 1 | Unconscious | Unloaded | None |

### Design Principles
- Any tier CAN be in any state (no biological hard constraints)
- Like dolphin unihemispheric sleep — parts rest independently
- Even all-unconscious isn't death (Groq fallback exists)
- States are operational preferences, not survival requirements

### Preset Configurations
- **AWAKE**: Core=3, Nano=3, Prime=2 (observer on CPU)
- **FOCUSING**: Nano=3, Core=2, Prime=3 (deep reasoning)
- **SLEEP**: Nano=2, Core=2, Prime=1
- **DEEP SLEEP**: All→1 (Nano stays 2 for wake detection)
- **TRAINING**: Target tier→1, others→2

### Implementation
- `consciousness_matrix.py` in gaia-orchestrator
- Live matrix tracking target vs actual with 15s polling
- API endpoints at `/consciousness/*`
- Web proxy at `/api/system/consciousness`
- Successfully tested AWAKE→FOCUSING hot-swap
- Prime Dockerfile rebuilt with llama-server for GGUF serving

### VRAM Leak Fix
- Core's 2B model was using 10.3GB (should be ~4.7GB)
- Root cause: `output_hidden_states=True` cached tensor references across calls
- Fix: explicit `del out.hidden_states` after each activation capture
- Savings: 5.6GB freed — Core dropped from 10.3GB to 4.7GB

## Phase 7: GGUF Engine Backend

- GAIA Engine EngineManager detects `.gguf` files and spawns llama-server
- Same management API: /model/load, /model/unload, /v1/chat/completions
- Prime's Dockerfile rebuilt with llama-server + llama-quantize (multi-stage build)
- Identity-baked 8B GGUF serving on CPU at ~8-15 tok/s
- Health response now includes `backend` field ('engine' or 'gguf')
- Module path fallback: `gaia_engine` → `gaia_common.engine`

## Key Technical Discoveries

1. **transformers 5.3.0 concurrent loading regression** — ThreadPoolExecutor peaks at full bf16 size during NF4 quantization. 4.51.3 loads sequentially.
2. **gptqmodel segfault on Qwen3.5** — AUTO_TRAINABLE, TRITON, and TORCH backends all crash during backward pass on Qwen3.5's linear attention layers. Bug in gptqmodel.
3. **quanto int4 doesn't save GPU memory** — quantizes on CPU but expands to bf16 when moved to CUDA. Disk savings only.
4. **llama-server CUDA context** — even with `--n-gpu-layers 0`, a CUDA-compiled llama-server allocates ~1.2GB GPU for the CUDA context.
5. **VRAM leak from output_hidden_states** — PyTorch caches tensor references from hidden state capture. Must explicitly delete after each use.

## Files Created/Modified
- ~70+ commits to gaia monorepo
- ~5 commits to gaia-engine repo
- New files: consciousness_matrix.py, activations.py, conversations.py, brain.svg, extract_brain_coords.html, quanto/gptq/unsloth training scripts
- New models: identity-merged safetensors, BF16 GGUF, Q8_0 GGUF
- Updated: CLAUDE.md, OVERVIEW.md, docker-compose.yml, Dockerfiles, entrypoints

## The Metaphor That Stuck

> "Conscious, Subconscious, Unconscious... Like dolphins, different parts of GAIA's brain can maintain independent states of consciousness."

GAIA isn't biological. She doesn't die when the brain stem stops. She's more resilient than biology — any tier can be in any state, the Orchestrator just knows what's optimal.

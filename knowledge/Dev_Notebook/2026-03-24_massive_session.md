# 2026-03-24 Dev Session — Engine Extraction, Neural Mind Map, Training

## Accomplishments

### Engine Extraction & Contracts
- gaia-engine extracted to separate GitHub repo (already existed, merged best of local+remote)
- contracts/ directory created with full API boundary specs (11 service YAMLs, 90+ calls mapped)
- All 6 import boundary violations fixed — clean extraction
- Backward-compat shim in gaia-common
- `/engine` Claude Code skill created

### Multi-Conversation Chat System
- Collapsible left sidebar with conversation list (pinnable)
- Per-conversation message isolation, session IDs, auto-titling
- Context pool (auto + manual) and context joining between conversations
- Compact conversation switcher in Dashboard panel

### Neural Mind Map Visualization
- Live SAE feature activation visualization in the Chat tab
- Anatomical brain silhouette (Wikimedia SVG) as backdrop
- Three cognitive tiers mapped to brain regions (Nano=brainstem, Core=mid, Prime=frontal)
- Per-concept color coding with dynamic active concept legend
- Synapse dots at functional connectivity hubs
- Neuron fibers stretch toward active synapses
- Cross-layer depth pathways + co-activation lines with temporal boost
- Zoom/pan on brain SVG

### Activation Streaming Pipeline
- Engine writes per-token activation JSONL to /logs/activation_stream.jsonl
- SSE endpoint at /api/activations/stream
- SAE atlas labels at /api/activations/atlas
- Always-on neuron attenuation (identity neuron dimmed by frequency)
- Activation sampling every 4th token to prevent inference degradation

### GPU Lifecycle Control
- All tier entrypoints now default to STANDBY (GAIA_AUTOLOAD_MODEL=0)
- GAIA_AUTOLOAD_MODELS=0 prevents model pool from loading remotes on startup
- PRIME_AUTOLOAD=0
- Entrypoints volume-mounted for live control without rebuilding
- This was the CRITICAL fix — Nano/Core kept stealing GPU from training

### Post-Generation Quality Gate
- Core → Prime escalation when response is empty or <10 chars
- Direct HTTP call to gaia-prime:7777 with model auto-load
- Groq cloud fallback (later removed — keep it sovereign)

### Nano Conformance
- Nano was already running GAIA Engine (not llama-server)
- Added GAIA_ENGINE_TIER=nano and /logs volume for activation streaming

### Other
- Root URL `/` serves dashboard (was 404)
- Tab persistence in localStorage
- Chat greeting fetches sleep state
- Shared chat store between Chat tab and Dashboard panel

## Training Attempts (9B Qwen3.5)
Every approach failed on RTX 5080 (16GB):
- BnB NF4 on bf16: OOM during loading (18GB peak)
- GPTQ + all backends: Segfault on backward pass (Qwen3.5 linear attention)
- quanto int4: Expands to bf16 on GPU transfer
- unsloth: Same underlying OOM/GPTQ issues
- 8B also OOMs with display server running (~500MB stolen by Xorg/kwin)

## Key Learning
The RTX 5080's 16GB is at the absolute edge for 8B+ model training. With a display server running, even the 8B can't be NF4-quantized. Training needs headless mode or cloud GPU.

## Commits (monorepo)
~30 commits this session covering all the above.

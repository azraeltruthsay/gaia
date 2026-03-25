# 2026-03-25 Session Continuation

## Accomplishments

### Consciousness Matrix — Live and Working
- Three-state model (Conscious/Subconscious/Unconscious) fully operational
- Orchestrator tracks target vs actual state with continuous polling
- Preset configurations: awake(), focusing(), sleep(), deep_sleep(), training()
- Successfully tested AWAKE→FOCUSING hot-swap (Core GPU↔Prime GPU)
- API endpoints at /consciousness/* on orchestrator port 6410
- Web proxy at /api/system/consciousness

### VRAM Leak Fix
- Core's 2B model was using 10.3GB (should be ~4.7GB)
- Root cause: output_hidden_states=True cached tensor references
- Fix: explicit `del out.hidden_states` + `monitor._last_snapshot = None` after each capture
- Savings: 5.6GB freed

### Brain Visualization Improvements
- Long sweeping neuron arcs spanning full brain regions
- SVG-traced coordinates for all 8 regions (correct path mapping)
- Consciousness-aware brightness (GPU=100%, CPU=35%, Unloaded=8%)
- Tier strength normalization (Nano 0.6x, Prime 1.2x)
- 3-second decay for visible activations
- Bigger dots (r=1.8 idle, r=3 active), thicker arcs (up to 2.5px)
- All regions now have visible idle dots

### Prime Escalation with Tool Access
- Escalation now uses model pool (not raw API) for system prompt + MCP tools
- Enables Prime to web_search when it doesn't know something
- Falls back to direct API if model pool unavailable

### Engine Fixes
- Module path fallback: gaia_engine → gaia_common.engine
- Health response always includes manager state (model_loaded, backend, mode)
- Volume-mount orchestrator source for live code changes
- GAIA_AUTOLOAD_MODELS=1 restored for model pool HTTP clients

## Key Issue: Core Empty Responses
Core (2B) frequently generates empty responses for non-trivial questions
(Jabberwocky, detailed architecture, creative tasks). The post-generation
quality gate catches this and escalates to Prime. Long-term fix: better
Core training curriculum or route complex queries directly to Prime.

## Next Steps
- Web search integration for epistemic validation
- Multi-neuron synapses (hub connections)
- Cross-tier pathway arcs
- Training visualization in the brain
- Q4_K_M quantization of identity GGUF (need llama-quantize)

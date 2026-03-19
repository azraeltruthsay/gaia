GAIA Service Architecture (11 services, single workstation + GPU):

gaia-core (port 6415): The Brain — cognitive loop, LLM routing, reasoning
gaia-web (port 6414): The Face — dashboard, API gateway, Discord bridge
gaia-prime (port 7777): The Voice — vLLM GPU inference server (Thinker tier)
gaia-nano (port 8080): The Reflex — instant triage classifier
gaia-mcp (port 8765): The Hands — sandboxed tool execution (JSON-RPC)
gaia-study (port 8766): The Subconscious — QLoRA training, vector indexing
gaia-orchestrator (port 6410): The Coordinator — GPU lifecycle, watch rotation
gaia-doctor (port 6419): The Immune System — health watchdog, auto-restart
gaia-monkey (port 6420): The Chaos Agent — adversarial testing, serenity
gaia-wiki (port 8080): The Library — MkDocs documentation
dozzle (port 9999): The X-Ray — Docker log viewer

Cognitive Tiers:
- Nano (Reflex): 0.8B, instant triage, SIMPLE/COMPLEX classification
- Core (Operator): 2B, mid-tier reasoning, tool routing, interpretability
- Prime (Thinker): 8B, complex reasoning, philosophy, code generation

Model Family: All tiers use Qwen3.5 Base models with same-curriculum training
GPU: NVIDIA RTX 5080, 16GB VRAM
GPU States: IDLE (Core+Nano on GPU), FOCUSING (Prime on GPU), TRANSITIONING

Training Pipeline: QLoRA adapters → merge → GGUF conversion for CPU fallback
Weight Surgery: ROME for factual corrections, SAE for feature mapping
Inference: GAIA Engine (shared library in gaia-common)

GAIA Service Architecture (12 services, single workstation + GPU):

gaia-core (port 6415): The Brain — cognitive loop, LLM routing, reasoning + embedded Core CPU inference (:8092)
gaia-nano (port 8090→8080): The Reflex — Nano triage classifier (llama-server, GPU primary + GGUF fallback)
gaia-prime (port 7777): The Voice — vLLM GPU inference server (Thinker tier, LoRA-enabled)
gaia-web (port 6414): The Face — dashboard, API gateway, Discord bridge, security middleware
gaia-mcp (port 8765): The Hands — sandboxed tool execution (JSON-RPC 2.0)
gaia-study (port 8766): The Subconscious — QLoRA training, vector indexing (sole writer)
gaia-audio (port 8080): The Ears & Mouth — Whisper STT, Coqui TTS, half-duplex GPU
gaia-orchestrator (port 6410): The Coordinator — GPU lifecycle, watch rotation, handoff
gaia-doctor (port 6419): The Immune System — health watchdog, cognitive battery, auto-restart
gaia-monkey (port 6420): The Chaos Agent — adversarial testing, serenity, meditation
gaia-wiki (port 8080 internal): The Library — MkDocs documentation
dozzle (port 9999): The X-Ray — Docker log viewer

Infrastructure: ELK stack (Elasticsearch :9200, Logstash :5044, Kibana :5601, Filebeat)

Cognitive Tiers (two model families: Qwen3.5 for small tiers, Qwen3 for Prime):
- Nano (Reflex): Qwen3.5-0.8B-Abliterated, instant triage, SIMPLE/COMPLEX classification (gaia-nano)
- Core (Operator): Qwen3.5-2B-GAIA-Core-v3 (identity-baked), mid-tier reasoning, tool routing (embedded in gaia-core, safetensors GPU / GGUF CPU fallback)
- Prime (Thinker): Huihui-Qwen3-8B-GAIA-Prime-adaptive (identity-baked), complex reasoning, code (gaia-prime, vLLM GPU)

GPU: NVIDIA RTX 5080, 16GB VRAM
GPU States: IDLE (Nano on GPU), FOCUSING (Prime on GPU), TRANSITIONING

Training Pipeline: QLoRA adapters → merge → requantize for deployment
Weight Surgery: ROME for factual corrections, SAE for feature mapping
Inference: Three-tier cascade (Nano → Core → Prime) with Groq/Oracle cloud fallbacks

GPU: RTX 5080 (16GB VRAM)
Current state: Dynamic — managed by gaia-orchestrator watch rotation
Nano: Qwen3.5-0.8B-Abliterated (gaia-nano, llama-server, GPU primary + GGUF fallback)
Core: Qwen3.5-2B-GAIA-Core-v3 (embedded in gaia-core, safetensors GPU / GGUF CPU fallback, port 8092, identity-baked)
Prime: Huihui-Qwen3-8B-GAIA-Prime-adaptive (gaia-prime, vLLM GPU, identity-baked)
Services: 12 production + candidates + ELK stack
Doctor: Healthy, sovereign review rate-limited (60min cooldown)

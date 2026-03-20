GPU: RTX 5080 (16GB VRAM)
Current state: Dynamic — managed by gaia-orchestrator watch rotation
Nano: Qwen3.5-0.8B-Abliterated (gaia-nano, llama-server, GPU primary + GGUF fallback)
Core: Qwen3-8B-abliterated Q4_K_M GGUF (embedded llama-server in gaia-core, CPU, port 8092)
Prime: Huihui-Qwen3-8B-GAIA-Prime-adaptive (gaia-prime, vLLM GPU, identity-baked)
Services: 12 production + candidates + ELK stack
Doctor: Healthy, sovereign review rate-limited (60min cooldown)

# GAIA Architectural Overview

> Distilled reference for specialist agents. Not a full design doc — a high-signal summary of what exists and how it connects.

## Service Topology

| Service | Role | Port | Responsibility |
|---------|------|------|----------------|
| **gaia-core** | The Brain | 6415 | Cognitive loop, reasoning, session state, packet dispatch, embedded Core inference (Qwen3.5-2B, :8092) |
| **gaia-nano** | The Reflex | 8090 | Nano triage classifier (Qwen3.5-0.8B, llama-server, GPU primary) |
| **gaia-prime** | The Voice | 7777 | GAIA Engine Thinker inference (GPU), OpenAI-compatible API, LoRA + polygraph + thought snapshots |
| **gaia-web** | The Face | 6414 | HTTP API gateway, Discord bot, Discord voice, web console, security middleware |
| **gaia-study** | The Subconscious | 8766 | Vector indexing (sole writer), QLoRA training, sleep-cycle reflection |
| **gaia-mcp** | The Hands | 8765 | Tool execution sandbox, JSON-RPC 2.0 interface, ~45+ tools |
| **gaia-orchestrator** | The Conductor | 6410 | GPU handoff state machine, container lifecycle |
| **gaia-audio** | The Ears & Mouth | 8080 | STT (Whisper), TTS (Coqui), half-duplex audio pipeline |
| **gaia-doctor** | The Immune System | 6419 | HA watchdog, cognitive test battery, auto-restart (stdlib only) |
| **gaia-monkey** | The Adversary | 6420 | Chaos drills, serenity/meditation management, PromptFoo red-teaming |
| **gaia-wiki** | The Library | 8080* | MkDocs internal developer documentation |

## Communication Pattern

```
User (Discord / Web Console / CLI)
  → gaia-web (gateway, sole external boundary)
    → gaia-core (CognitionPacket dispatch)
      → intent detection → knowledge enhancement → semantic probe
      → [if tool needed] tool routing → gaia-mcp (JSON-RPC)
      → [generation] gaia-prime (vLLM inference)
    ← CognitionPacket (enriched with response)
  ← output router → format for destination (Discord/Web/CLI/Audio)
```

All inter-service communication is HTTP on internal Docker network (`gaia-network`, 172.28.0.0/16). No public-facing APIs except gaia-web.

## Candidate/Live Architecture

Every service has a candidate counterpart (`-candidate` suffix, +1 port offset). Candidates mount from `./candidates/` instead of production paths.

- **Parallel mode**: candidates talk to candidates (isolated testing)
- **Injection mode**: live calls candidate endpoints (A/B testing)
- **HA fallback**: `CORE_FALLBACK_ENDPOINT` and `MCP_FALLBACK_ENDPOINT` route to candidates when live fails
- **Maintenance flag**: `/shared/ha_maintenance` suppresses failover during deploys

## Gateway Principle

gaia-web is the **sole external communications boundary**. All user-facing I/O (Discord, web console, API) enters through gaia-web. No other service accepts external traffic.

## Key Architectural Invariants

1. **CognitionPacket is the universal message format** — every cognitive operation flows through it
2. **gaia-study is the sole vector store writer** — all other services read only
3. **gaia-mcp tools require approval for sensitive operations** — challenge-response with TTL
4. **Blueprints are epistemic** — CANDIDATE (prescriptive/unvalidated) vs LIVE (descriptive/validated)
5. **Loop detection is built-in** — LoopRecoveryManager caps reinjection at 3x
6. **Observer pattern** — StreamObserver monitors for repetition, spam, loops in real-time

# Architecture Overview

GAIA is a service-oriented AI system where each container plays a distinct cognitive role. The architecture follows a strict separation of concerns: inference is decoupled from reasoning, reasoning from interface, and all mutable state is governed by a single writer (`gaia-study`).

## Service Topology

```
                          ┌──────────────────────┐
                          │   gaia-prime          │
                          │   The Voice           │
                          │   GAIA Engine         │
                          │   :7777               │
                          └──────────┬────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
              ┌─────┴──────┐  ┌─────┴──────┐  ┌──────┴──────┐
              │ gaia-core  │  │ gaia-nano  │  │ gaia-study  │
              │ The Brain  │  │ (deprec.)  │  │ The Sub-    │
              │ :6415      │  │ :8090      │  │ conscious   │
              └─────┬──────┘  └────────────┘  │ :8766       │
                    │                         └─────────────┘
              ┌─────┴──────┐
              │ gaia-mcp   │       ┌──────────────┐
              │ The Hands  │       │ gaia-orch    │
              │ :8765      │       │ The Coord    │
              └────────────┘       │ :6410        │
                    │              └──────────────┘
              ┌─────┴──────┐
              │ gaia-web   │       ┌──────────────┐
              │ The Face   │       │ gaia-audio   │
              │ :6414      │       │ The Ears/Mouth│
              └────────────┘       │ :8080        │
                                   └──────────────┘
```

## Data Flow: The Cascade

1. **gaia-web** receives input (Discord message or web UI POST).
2. Constructs a `CognitionPacket` and POSTs to **gaia-core** `/process_packet`.
3. **gaia-core** triages the request itself (Sovereign Duality — the Nano tier is
   deprecated; **gaia-nano** is now a socat passthrough that forwards `:8080` to
   Core's embedded engine at `gaia-core:8092`, preserving the DNS name).
4. **gaia-core** runs the cognitive loop:
    - Loads session history and context.
    - If COMPLEX: Offloads to **gaia-prime** (The Voice) or the Groq cloud fallback.
    - If SIMPLE: Handles locally via the embedded Core engine.
    - Optionally calls **gaia-mcp** for tool execution.
5. Response flows back through the packet to **gaia-web**.
6. **gaia-web** routes the response to the destination (Discord, Web, or Audio).

## Volume Access Matrix

| Volume | gaia-core | gaia-study | gaia-mcp | gaia-web | gaia-orchestrator |
|--------|-----------|------------|----------|----------|-------------------|
| knowledge/ | RW | RW (SOLE WRITER for vectors) | RW | RO | RO |
| vector_store/ | RO | RW (SOLE WRITER) | - | - | - |
| gaia-models/ | RO | RW (LoRA adapters) | RO | - | RO |
| gaia-shared | RW | RW | RW | RW | RW |
| warm_pool/ | RO | RO | - | - | RW |

## Dual Stack: Live + Candidate

Core services have candidate counterparts in `candidates/` (defined in
`docker-compose.candidate.yml`). The candidate stack:
- Shares the `gaia-net` network with live services.
- Uses separate Docker volumes (`gaia-candidate-shared`).
- Maps to different external ports (+1 offset, with jumps for collisions).

See [Candidate Pipeline](../operations/candidate-pipeline.md) for the full promotion workflow.

## GAIA Inference Engine

Both inference tiers (Prime, Core) run the standalone **GAIA Engine** (separate repository, `github.com/azraeltruthsay/gaia-engine`). It provides:
- Optimized generation (~22 tok/s).
- Hidden State Polygraph (real-time monitoring).
- KV Cache Snapshots.
- Dynamic LoRA adapter management.

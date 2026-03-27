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
              │ The Brain  │  │ The Reflex │  │ The Sub-    │
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
3. **gaia-core** sends a triage request to **gaia-nano** (The Reflex):
    - Nano classifies as SIMPLE (handled by Core/Groq) or COMPLEX (requires Prime).
    - Nano also cleans up voice transcripts.
4. **gaia-core** runs the cognitive loop:
    - Loads session history and context.
    - If COMPLEX: Offloads to **gaia-prime** (The Thinker) or cloud fallback.
    - If SIMPLE: Handles locally or via fast Groq/Oracle fallback.
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

Every service has a candidate counterpart in `candidates/`. The candidate stack:
- Shares the `gaia-net` network with live services.
- Uses separate Docker volumes (`gaia-candidate-shared`).
- Maps to different external ports (+1 offset, with jumps for collisions).

See [Candidate Pipeline](../operations/candidate-pipeline.md) for the full promotion workflow.

## GAIA Inference Engine

All inference tiers (Prime, Core, Nano) run the standalone **GAIA Engine** (separate repository). It provides:
- Optimized generation (~22 tok/s).
- Hidden State Polygraph (real-time monitoring).
- KV Cache Snapshots.
- Dynamic LoRA adapter management.

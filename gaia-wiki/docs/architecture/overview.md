# Architecture Overview

GAIA is a service-oriented AI system where each container plays a distinct cognitive role. The architecture follows a strict separation of concerns: inference is decoupled from reasoning, reasoning from interface, and all mutable state is governed by a single writer.

## Service Topology

```
                          ┌──────────────────────┐
                          │   gaia-prime (GPU)    │
                          │   The Voice           │
                          │   vLLM inference      │
                          │   :7777               │
                          └──────────┬────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
              ┌─────┴──────┐  ┌─────┴──────┐  cloud fallback
              │ gaia-core  │  │ gaia-study │  (groq/openai/
              │ The Brain  │  │ The Sub-   │   gemini)
              │ :6415      │  │ conscious  │
              └─────┬──────┘  │ :8766      │
                    │         └────────────┘
              ┌─────┴──────┐
              │ gaia-mcp   │
              │ The Hands  │
              │ :8765      │
              └────────────┘
                    │
              ┌─────┴──────┐       ┌──────────────┐
              │ gaia-web   │       │ gaia-orch    │
              │ The Face   │       │ The Coord    │
              │ :6414      │       │ :6410        │
              └────────────┘       └──────────────┘
```

## Data Flow: User Message

1. **gaia-web** receives input (Discord message or web UI POST)
2. Constructs a `CognitionPacket` with routing, context, and metadata
3. POSTs packet to **gaia-core** `/process_packet`
4. **gaia-core** runs the cognitive loop:
    - Loads session history
    - Builds prompt (system prompt + context + prime.md checkpoint)
    - Sends inference request to **gaia-prime** (or cloud fallback)
    - Runs Observer for side-effect verification
    - Optionally calls **gaia-mcp** for tool execution
5. Response flows back through the packet to **gaia-web**
6. **gaia-web** routes the response to the original destination (Discord channel, web UI, audio)

## Volume Access Matrix

| Volume | gaia-core | gaia-study | gaia-web | gaia-mcp |
|--------|----------|-----------|---------|---------|
| knowledge/ | RW | RW (SOLE WRITER for vectors) | RO | RW |
| vector_store/ | RO | RW (SOLE WRITER) | - | - |
| gaia-models/ | RO | RW (LoRA adapters) | - | RO |
| gaia-shared | RW | RW | - | - |

## Dual Stack: Live + Candidate

Every service has a candidate counterpart in `candidates/`. The candidate stack:

- Shares the `gaia-net` network with live services
- Uses separate Docker volumes (`gaia-candidate-shared`)
- Maps to different external ports (e.g., candidate-core: `6416:6415`)
- Can be run in HA mode as a hot standby (see [HA Failover](../operations/network-layout.md))

See [Candidate Pipeline](../operations/candidate-pipeline.md) for the full promotion workflow.

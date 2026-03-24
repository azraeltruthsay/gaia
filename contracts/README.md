# GAIA Inter-Service Contract Specification

This directory documents every API boundary between GAIA services. It is the
authoritative reference for how services communicate, what they expect, and
what they promise.

## Structure

```
contracts/
  README.md              -- This file
  CONNECTIVITY.md        -- Master connectivity matrix (at-a-glance view)
  services/
    gaia-engine.yaml     -- The Spine: shared inference engine (SEPARATE REPO)
    gaia-core.yaml       -- The Brain: cognitive loop, LLM routing, reasoning
    gaia-web.yaml        -- The Face: dashboard, API gateway, Discord bridge
    gaia-prime.yaml      -- The Voice: GAIA Engine inference (GPU)
    gaia-nano.yaml       -- The Reflex: Nano triage classifier
    gaia-mcp.yaml        -- The Hands: sandboxed tool execution (JSON-RPC)
    gaia-study.yaml      -- The Subconscious: training, vector indexing
    gaia-orchestrator.yaml -- The Coordinator: GPU lifecycle, HA overlay
    gaia-doctor.yaml     -- The Immune System: HA watchdog
    gaia-monkey.yaml     -- The Chaos Agent: adversarial testing
    gaia-audio.yaml      -- The Ears & Mouth: STT/TTS
  schemas/
    cognition_packet.yaml -- CognitionPacket structure (web <-> core)
    json_rpc.yaml         -- JSON-RPC 2.0 format (core <-> mcp)
```

## Separate Repositories

**gaia-engine** (github.com/azraeltruthsay/gaia-engine) is the only component
extracted to its own repo. It is a Python library (not a service) consumed by
gaia-prime, gaia-nano, gaia-core, gaia-orchestrator, and gaia-study. Its contract
is defined in `services/gaia-engine.yaml` and covers both the Python API and the
HTTP API exposed when running in managed mode.

## How to Use

- **Adding a new endpoint?** Update the provider service YAML and add consumers.
- **Adding a new inter-service call?** Update both the consumer's `consumes`
  section and the provider's `provides` section.
- **Reviewing architecture?** Start with `CONNECTIVITY.md` for the graph view,
  then drill into individual service YAMLs for schemas and details.

## Validation Approach

1. **Static**: Service YAMLs are the declared contracts. PRs that add or change
   inter-service calls must update the relevant YAML files.
2. **Live**: `gaia-doctor` compiles a service registry
   (`/shared/registry/service_registry.json`) and validates edges against
   running services' OpenAPI schemas via `GET /api/system/registry/paths`.
3. **Smoke**: The cognitive test battery includes architecture tests that
   verify GAIA knows her own service topology.

## Conventions

- Ports listed are the **internal container port** (what services use on the
  Docker network). Host-mapped ports are noted separately where relevant.
- All HTTP communication uses the `gaia-net` Docker bridge network.
- gaia-doctor uses stdlib-only HTTP (`http.server`), not FastAPI. Its endpoints
  are plain `do_GET`/`do_POST` handlers.
- gaia-prime and gaia-nano run the GAIA Engine (custom inference server), not
  FastAPI. Their APIs are OpenAI-compatible with GAIA extensions.

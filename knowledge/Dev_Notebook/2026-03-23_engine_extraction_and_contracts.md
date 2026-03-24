# Dev Journal — 2026-03-23: Engine Extraction & Service Contracts

## Summary

Major architectural milestone: extracted the GAIA Inference Engine into its own GitHub repository and established formal inter-service contract boundaries for the entire SOA.

## gaia-engine Extraction

- **New repo**: `github.com/azraeltruthsay/gaia-engine` (Apache-2.0 license)
- The inference engine (`GAIAEngine`, KV cache, polygraph, LoRA adapter management, GPU/CPU migration, vision support, ROME/SAE companions) is now an independent library, not buried inside `gaia-common`
- **Merged local + remote versions**: Combined SSE streaming support with the clean `event_callback` interface into a single coherent codebase
- Engine is consumed by 5 services: gaia-prime, gaia-nano, gaia-core, gaia-orchestrator, gaia-study

## Backward-Compat Shim

- `gaia-common/gaia_common/engine/` is now a **shim layer** that delegates all imports to the `gaia_engine` package
- Existing service code using `from gaia_common.engine import X` continues to work unchanged
- New code and scripts should import directly: `from gaia_engine import X`

## Import Boundary Cleanup

- Found and fixed **6 import violations** — direct imports into `gaia_common.engine.core`, `.manager`, `.sae_trainer`, `.rome` etc.
- All imports now go through either the shim (`gaia_common.engine`) or `gaia_engine` directly
- Boundary is clean: no service reaches past the public API

## contracts/ Directory

- Created `contracts/` with full API boundary specifications for all GAIA services
- `contracts/services/` — per-service YAML files defining endpoints, request/response schemas, dependencies
- `contracts/schemas/` — shared schema definitions (CognitionPacket, JSON-RPC)
- `CONNECTIVITY.md` — master matrix documenting all 90+ inter-service calls
- `gaia-engine.yaml` — contract spec for the extracted engine library
- gaia-doctor validates the registry against live services

## Sync Status

- All changes synced to `candidates/`
- gaia-engine repo pushed to GitHub with full history

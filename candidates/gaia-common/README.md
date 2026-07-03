# gaia-common

Shared library for GAIA SOA services — protocols, utilities, constants, and lifecycle
definitions used by every containerized service.

## Key Contents

- `config.py` - Centralized configuration (Config singleton merging constants + env + defaults)
- `protocols/` - Inter-service message formats (e.g. `cognition_packet.py`)
- `lifecycle/states.py` - GPU gearbox state machine (P/1/1+/2/S/0/T)
- `constants/gaia_constants.json` - Master config (token budgets, endpoints, model configs)
- `utils/safe_execution.py` - Safe shell execution primitives
- `utils/world_state.py` - World state utilities
- `engine/` - Backward-compat shim delegating to the `gaia_engine` package (engine code
  lives in the separate `gaia-engine` repo, NOT here)

gaia-common is volume-mounted read-only into all containers; source changes require a
container restart. Services must not depend on gaia-core — shared code lives here to keep
the SOA decoupled.

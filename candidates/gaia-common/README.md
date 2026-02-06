# gaia-common (Candidate)

Shared library for GAIA SOA services - Candidate version for testing SOA-decoupled architecture.

## Changes from Active

This candidate version includes:
- `config.py` - Centralized configuration (moved from gaia-core)
- `utils/safe_execution.py` - Safe shell execution primitives
- `utils/world_state.py` - World state utilities
- `utils/gaia_rescue_helper.py` - Updated rescue helper

These changes support the SOA-decoupled architecture where gaia-mcp no longer depends on gaia-core.

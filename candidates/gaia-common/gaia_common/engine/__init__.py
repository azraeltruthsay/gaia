"""
GAIA Inference Engine — shared cognitive inference library.

Used by all tier containers (gaia-core, gaia-nano, and Prime when loaded).
Each container imports this library and serves one model through it.

    from gaia_common.engine import GAIAEngine, serve

    engine = GAIAEngine("/models/Qwen3.5-2B-GAIA-Core-v3", device="cuda")
    result = engine.generate(messages=[...])

    # Or run as a standalone server:
    serve("/models/Qwen3.5-2B-GAIA-Core-v3", port=8092)

When the standalone gaia-engine package is installed, this module
re-exports from it. Otherwise, uses the local implementation.
"""

try:
    # Prefer standalone gaia-engine package when installed
    from gaia_engine import GAIAEngine, serve, EngineManager, serve_managed
    from gaia_engine import compose_thoughts, estimate_composed_size
    __all__ = ["GAIAEngine", "serve", "EngineManager", "serve_managed",
               "compose_thoughts", "estimate_composed_size"]
except ImportError:
    # Fall back to local implementation
    from gaia_common.engine.core import GAIAEngine, serve
    from gaia_common.engine.thought_composer import compose_thoughts, estimate_composed_size
    __all__ = ["GAIAEngine", "serve", "compose_thoughts", "estimate_composed_size"]

"""Backward-compatibility shim — delegates to standalone gaia_engine package."""
try:
    from gaia_engine import (
        GAIAEngine, serve, EngineManager, serve_managed,
        compose_thoughts, estimate_composed_size,
    )
except ImportError:
    # Fallback to local if gaia_engine not installed
    from gaia_common.engine.core import GAIAEngine, serve
    from gaia_common.engine.manager import EngineManager, serve_managed
    from gaia_common.engine.thought_composer import compose_thoughts, estimate_composed_size

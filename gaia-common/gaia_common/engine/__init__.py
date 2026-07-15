"""Backward-compatibility shim — delegates to standalone gaia_engine package.

Imports each sub-module dynamically so that a partial gaia_engine install
(e.g. pip-installed base missing gaia_engine.core) still resolves serve_managed
and EngineManager from the installed manager.py, while avoiding eager imports
of heavy libraries like PyTorch and vLLM in managed/standby mode.
"""

def __getattr__(name):
    if name in ("GAIAEngine", "serve"):
        try:
            from gaia_engine.core import GAIAEngine, serve
        except ImportError:
            from gaia_common.engine.core import GAIAEngine, serve
        if name == "GAIAEngine":
            return GAIAEngine
        return serve

    if name in ("compose_thoughts", "estimate_composed_size"):
        try:
            from gaia_engine.thought_composer import compose_thoughts, estimate_composed_size
        except ImportError:
            from gaia_common.engine.thought_composer import compose_thoughts, estimate_composed_size
        if name == "compose_thoughts":
            return compose_thoughts
        return estimate_composed_size

    if name in ("EngineManager", "serve_managed"):
        try:
            from gaia_engine.manager import EngineManager, serve_managed
        except ImportError:
            from gaia_common.engine.manager import EngineManager, serve_managed
        if name == "EngineManager":
            return EngineManager
        return serve_managed

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

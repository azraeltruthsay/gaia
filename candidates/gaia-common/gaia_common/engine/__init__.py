"""Backward-compatibility shim — delegates to standalone gaia_engine package.

Imports each sub-module independently so that a partial gaia_engine install
(e.g. pip-installed base missing gaia_engine.core) still resolves serve_managed
and EngineManager from the installed manager.py.
"""
try:
    from gaia_engine.manager import EngineManager, serve_managed
except ImportError:
    from gaia_common.engine.manager import EngineManager, serve_managed

try:
    from gaia_engine.core import GAIAEngine, serve
except ImportError:
    from gaia_common.engine.core import GAIAEngine, serve

try:
    from gaia_engine.thought_composer import compose_thoughts, estimate_composed_size
except ImportError:
    from gaia_common.engine.thought_composer import compose_thoughts, estimate_composed_size

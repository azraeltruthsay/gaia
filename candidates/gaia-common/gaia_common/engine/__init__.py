"""
GAIA Inference Engine — shared cognitive inference library.

Used by all tier containers (gaia-core, gaia-nano, and Prime when loaded).
Each container imports this library and serves one model through it.

    from gaia_common.engine import GAIAEngine, serve

    engine = GAIAEngine("/models/Qwen3.5-2B-GAIA-Core-v3", device="cuda")
    result = engine.generate(messages=[...])

    # Or run as a standalone server:
    serve("/models/Qwen3.5-2B-GAIA-Core-v3", port=8092)
"""

from gaia_common.engine.core import GAIAEngine, serve
from gaia_common.engine.thought_composer import compose_thoughts, estimate_composed_size

__all__ = ["GAIAEngine", "serve", "compose_thoughts", "estimate_composed_size"]

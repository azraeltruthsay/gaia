"""
gaia-core: The Brain - Cognitive loop and reasoning engine.

This service is the heart of GAIA, responsible for:
- Agent cognitive loop (reason-act-reflect)
- Model pool orchestration (Prime/Lite/Embedding)
- Prompt assembly and response generation
- Intent detection and planning
- Tool selection and MCP client communication

Dependencies:
- gaia-common: Shared protocols and utilities
- gaia-mcp: Tool execution (via HTTP/JSON-RPC)
- gaia-study: Background processing (via HTTP)
"""

import multiprocessing

# Ensure the multiprocessing start method is set to 'spawn' as early as
# possible. This prevents CUDA re-initialization errors when worker
# subprocesses are spawned (vLLM / torch require 'spawn' on CUDA systems).
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set - this is fine
    pass

__version__ = "0.1.0"
__service__ = "gaia-core"

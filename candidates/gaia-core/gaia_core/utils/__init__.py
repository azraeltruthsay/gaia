"""
gaia_core.utils - Utility modules for the GAIA cognitive engine.

This package provides:
- prompt_builder: Construct prompts from CognitionPackets
- packet_builder: Build and manipulate CognitionPackets
- packet_templates: Template rendering for packets
- world_state: Dynamic world state snapshots
- mcp_client: MCP server communication client
- gaia_rescue_helper: Recovery and diagnostic utilities
- dev_matrix_utils: Development matrix utilities
"""

# Note: Explicit imports deferred until app.* dependencies are fully migrated.
# Once migration is complete, add convenience imports here.

__all__ = [
    "prompt_builder",
    "packet_builder",
    "packet_templates",
    "world_state",
    "mcp_client",
    "gaia_rescue_helper",
    "dev_matrix_utils",
    "stream_observer",
    "output_router",
]

# NOTE: Avoid eager imports of modules that have complex import chains.
# stream_observer -> external_voice -> prompt_builder creates circular issues.
# These modules should be imported directly when needed:
#   from gaia_core.utils.stream_observer import StreamObserver
#   from gaia_core.utils.output_router import route_output


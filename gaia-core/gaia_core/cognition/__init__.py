"""
gaia_core.cognition - Cognitive processing and reasoning modules.

This package provides:
- agent_core: Main cognitive loop (AgentCore class)
- cognitive_dispatcher: Route and dispatch cognitive tasks
- knowledge_enhancer: Enhance packets with retrieved knowledge
- self_reflection: Self-reflection and refinement
- tool_selector: Tool selection and routing logic
- topic_manager: Topic tracking and management
- packet_utils: Packet manipulation utilities
- packet_upgrade: Packet version migration
"""

# Note: Explicit imports deferred until app.* dependencies are fully migrated.
# Once migration is complete, add convenience imports here like:
# from .agent_core import AgentCore

__all__ = [
    "agent_core",
    "cognitive_dispatcher",
    "goal_detector",
    "knowledge_enhancer",
    "self_reflection",
    "tool_selector",
    "topic_manager",
    "packet_utils",
    "packet_upgrade",
    "external_voice",
    "thought_seed",
]

from . import agent_core
from . import cognitive_dispatcher
from . import goal_detector
from . import knowledge_enhancer
from . import self_reflection
from . import tool_selector
from . import topic_manager
from . import packet_utils
from . import packet_upgrade
from . import external_voice
from . import thought_seed


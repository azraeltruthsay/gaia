"""
GAIA Orchestrator - GPU and Container Lifecycle Coordination Service.

Provides centralized control for:
- GPU ownership and handoff between services
- Container lifecycle management (live/candidate stacks)
- Prime↔Study handoff protocol for training sessions
- Oracle fallback notifications
"""

__version__ = "0.1.0"

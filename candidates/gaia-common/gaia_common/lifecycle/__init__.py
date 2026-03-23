"""
GAIA Lifecycle — unified GPU lifecycle state machine types.

Shared by orchestrator (authority) and gaia-core (consumer).

When the standalone gaia-engine package is installed, this module
re-exports from it. Otherwise, uses the local implementation.
"""

try:
    from gaia_engine.lifecycle.states import (
        LifecycleState, TransitionTrigger, TRANSITIONS, TIER_EXPECTATIONS,
        TierExpectation, available_transitions, validate_transition,
    )
    from gaia_engine.lifecycle.snapshot import (
        LifecycleSnapshot, TierLiveStatus, TransitionRecord, TransitionResult,
    )
except ImportError:
    from gaia_common.lifecycle.states import (
        LifecycleState, TransitionTrigger, TRANSITIONS, TIER_EXPECTATIONS,
        TierExpectation, available_transitions, validate_transition,
    )
    from gaia_common.lifecycle.snapshot import (
        LifecycleSnapshot, TierLiveStatus, TransitionRecord, TransitionResult,
    )

__all__ = [
    "LifecycleState",
    "TransitionTrigger",
    "TRANSITIONS",
    "TIER_EXPECTATIONS",
    "TierExpectation",
    "available_transitions",
    "validate_transition",
    "LifecycleSnapshot",
    "TierLiveStatus",
    "TransitionRecord",
    "TransitionResult",
]

"""
Lifecycle state definitions — the single source of truth for GAIA's GPU lifecycle.

Sovereign Duality Gearbox — Pure Gemma 4 Edition
=================================================

States map to "gears" in the transmission metaphor:

  Gear 0  = OFF          All containers stopped. 0 GPU. (~415 MB host baseline)
  Gear P  = PARKED       Core on CPU (GGUF). GPU empty. Sentinel standby.
  Gear 1  = AWAKE        Core E4B on GPU (NF4, ~8.8 GB). Prime on CPU (GGUF).
  Gear 2  = FOCUSING     Prime 26B-A4B on GPU (Expert Buffered, ~4.6 GB). Core on CPU.
  ---
  SLEEP                  Core on CPU. Prime unloaded. Low-power standby.
  DEEP_SLEEP             Everything unloaded. GPU empty. Groq fallback only.
  MEDITATION             Study owns GPU for training. All cognitive tiers off.
  LISTENING              AWAKE + audio STT active.

The "clutch" is not a state — it's the transition protocol itself:
  1. Capture context (Neural Handoff)
  2. Unload active GPU tier
  3. Load target tier
  4. Replay context into new backend

VRAM budget (measured on RTX 5080 16GB):
  Core E4B NF4 safetensors:    ~8.8 GB
  Prime 26B-A4B Expert Buffered: ~4.6 GB
  CUDA context overhead:        ~2.9 GB (manager process, unavoidable while running)
  Both on GPU simultaneously:   NOT FEASIBLE on 16 GB (13.4 GB + KV cache = OOM)
"""

from enum import Enum
from typing import Dict, List, Optional, Set


class LifecycleState(str, Enum):
    """Primary lifecycle states for GAIA's GPU allocation.

    Sovereign Duality: two-tier architecture (Core + Prime), Nano deprecated.
    """
    AWAKE = "awake"              # Gear 1: Core E4B on GPU (~8.8 GB). Prime on CPU.
    LISTENING = "listening"      # Gear 1+: AWAKE with audio STT active.
    FOCUSING = "focusing"        # Gear 2: Prime on GPU (~4.6 GB). Core on CPU.
    MEDITATION = "meditation"    # Study owns GPU for training. All cognitive tiers off.
    SLEEP = "sleep"              # Core on CPU. Prime unloaded. GPU empty.
    DEEP_SLEEP = "deep_sleep"    # Everything unloaded. GPU empty. Groq fallback only.
    PARKED = "parked"            # Gear P: Core on CPU (GGUF). GPU empty. Pre-warmed sentinel.
    TRANSITIONING = "transitioning"  # Clutch engaged — handoff in progress.


class TransitionTrigger(str, Enum):
    """Events that trigger lifecycle state transitions."""
    IDLE_TIMEOUT = "idle_timeout"            # No user activity for threshold minutes
    WAKE_SIGNAL = "wake_signal"              # Message received during sleep/park
    VOICE_JOIN = "voice_join"                # User joined Discord voice channel
    VOICE_LEAVE = "voice_leave"              # User left Discord voice channel
    ESCALATION_NEEDED = "escalation_needed"  # Complex query requires Prime (Gear 1→2)
    TASK_COMPLETE = "task_complete"           # Prime finished, downshift (Gear 2→1)
    TRAINING_SCHEDULED = "training_scheduled" # Study needs GPU for QLoRA/merge
    TRAINING_COMPLETE = "training_complete"   # Study finished training
    USER_REQUEST = "user_request"            # Manual transition from dashboard/API
    EXTENDED_IDLE = "extended_idle"           # Long idle → deeper sleep
    PREEMPT = "preempt"                      # Wake signal during MEDITATION
    ENGAGE_CLUTCH = "engage_clutch"          # Shift from PARKED → AWAKE (Gear P→1)
    BOOT_TO_PARK = "boot_to_park"            # Initial system boot into PARKED state


class TierExpectation:
    """Expected state of a tier in a given lifecycle state."""
    __slots__ = ("device", "required")

    def __init__(self, device: str, required: bool = True):
        self.device = device      # "gpu", "cpu", "unloaded"
        self.required = required  # Must this tier be in this state?

    def __repr__(self):
        return f"TierExpectation({self.device}, required={self.required})"


# ── Transition Table ──────────────────────────────────────────────────────────
# Dict[source_state] → Dict[trigger] → target_state
# USER_REQUEST can target multiple states, handled specially in validate_transition.
#
# Sovereign Duality transition map:
#
#   OFF ──boot──→ PARKED ──clutch──→ AWAKE ←──task_complete── FOCUSING
#                    ↑                 │                          ↑
#                    │              escalation                    │
#                    │                 └────────────────────────→─┘
#                    │              idle_timeout
#                    ├──────────────────┘
#                    │
#                    └──extended_idle──→ DEEP_SLEEP

TRANSITIONS: Dict[LifecycleState, Dict[TransitionTrigger, LifecycleState]] = {
    LifecycleState.AWAKE: {
        TransitionTrigger.VOICE_JOIN: LifecycleState.LISTENING,
        TransitionTrigger.ESCALATION_NEEDED: LifecycleState.FOCUSING,
        TransitionTrigger.IDLE_TIMEOUT: LifecycleState.PARKED,
        TransitionTrigger.TRAINING_SCHEDULED: LifecycleState.MEDITATION,
        # USER_REQUEST handled in validate_transition
    },
    LifecycleState.LISTENING: {
        TransitionTrigger.VOICE_LEAVE: LifecycleState.AWAKE,
        TransitionTrigger.ESCALATION_NEEDED: LifecycleState.FOCUSING,
    },
    LifecycleState.FOCUSING: {
        TransitionTrigger.TASK_COMPLETE: LifecycleState.AWAKE,
        TransitionTrigger.VOICE_JOIN: LifecycleState.LISTENING,
        TransitionTrigger.TRAINING_SCHEDULED: LifecycleState.MEDITATION,
        TransitionTrigger.IDLE_TIMEOUT: LifecycleState.PARKED,
    },
    LifecycleState.MEDITATION: {
        TransitionTrigger.TRAINING_COMPLETE: LifecycleState.AWAKE,
        TransitionTrigger.PREEMPT: LifecycleState.AWAKE,
        TransitionTrigger.WAKE_SIGNAL: LifecycleState.AWAKE,
    },
    LifecycleState.SLEEP: {
        TransitionTrigger.WAKE_SIGNAL: LifecycleState.AWAKE,
        TransitionTrigger.EXTENDED_IDLE: LifecycleState.DEEP_SLEEP,
        TransitionTrigger.TRAINING_SCHEDULED: LifecycleState.MEDITATION,
        TransitionTrigger.USER_REQUEST: LifecycleState.PARKED,
    },
    LifecycleState.DEEP_SLEEP: {
        TransitionTrigger.WAKE_SIGNAL: LifecycleState.AWAKE,
        TransitionTrigger.USER_REQUEST: LifecycleState.PARKED,
    },
    LifecycleState.PARKED: {
        TransitionTrigger.WAKE_SIGNAL: LifecycleState.AWAKE,
        TransitionTrigger.ENGAGE_CLUTCH: LifecycleState.AWAKE,
        TransitionTrigger.TRAINING_SCHEDULED: LifecycleState.MEDITATION,
        TransitionTrigger.EXTENDED_IDLE: LifecycleState.DEEP_SLEEP,
    },
    LifecycleState.TRANSITIONING: {
        # No triggers — transitions out are handled by the machine itself
    },
}

# States reachable via USER_REQUEST from each state
USER_REQUEST_TARGETS: Dict[LifecycleState, Set[LifecycleState]] = {
    LifecycleState.AWAKE: {
        LifecycleState.FOCUSING, LifecycleState.SLEEP,
        LifecycleState.DEEP_SLEEP, LifecycleState.MEDITATION,
        LifecycleState.PARKED,
    },
    LifecycleState.LISTENING: {
        LifecycleState.AWAKE, LifecycleState.FOCUSING,
    },
    LifecycleState.FOCUSING: {
        LifecycleState.AWAKE, LifecycleState.SLEEP, LifecycleState.DEEP_SLEEP,
        LifecycleState.PARKED,
    },
    LifecycleState.MEDITATION: {
        LifecycleState.AWAKE,
    },
    LifecycleState.SLEEP: {
        LifecycleState.AWAKE, LifecycleState.DEEP_SLEEP, LifecycleState.PARKED,
    },
    LifecycleState.DEEP_SLEEP: {
        LifecycleState.AWAKE, LifecycleState.SLEEP, LifecycleState.PARKED,
    },
    LifecycleState.PARKED: {
        LifecycleState.AWAKE, LifecycleState.FOCUSING,
        LifecycleState.SLEEP, LifecycleState.DEEP_SLEEP,
        LifecycleState.MEDITATION,
    },
    LifecycleState.TRANSITIONING: set(),
}


# ── Tier Expectations Per State ───────────────────────────────────────────────
# Defines what each tier should look like in each lifecycle state.
# The lifecycle machine uses this to determine which load/unload actions to take.
#
# Sovereign Duality: Nano is deprecated. Only Core and Prime are active tiers.
# Nano entries kept as "unloaded" for backward compatibility with probes.

TIER_EXPECTATIONS: Dict[LifecycleState, Dict[str, TierExpectation]] = {
    # Gear 1: Core on GPU (~8.8 GB), Prime on CPU for fallback
    LifecycleState.AWAKE: {
        "core":  TierExpectation("gpu", required=True),
        "nano":  TierExpectation("unloaded", required=False),
        "prime": TierExpectation("cpu", required=False),
        "study": TierExpectation("unloaded", required=False),
    },
    # Gear 1+: Same as AWAKE with audio active
    LifecycleState.LISTENING: {
        "core":  TierExpectation("gpu", required=True),
        "nano":  TierExpectation("unloaded", required=False),
        "prime": TierExpectation("cpu", required=False),
        "study": TierExpectation("unloaded", required=False),
    },
    # Gear 2: Prime on GPU (~4.6 GB), Core drops to CPU
    LifecycleState.FOCUSING: {
        "core":  TierExpectation("cpu", required=True),
        "nano":  TierExpectation("unloaded", required=False),
        "prime": TierExpectation("gpu", required=True),
        "study": TierExpectation("unloaded", required=False),
    },
    # Training: Study owns GPU, everything else off
    LifecycleState.MEDITATION: {
        "core":  TierExpectation("unloaded", required=False),
        "nano":  TierExpectation("unloaded", required=False),
        "prime": TierExpectation("unloaded", required=False),
        "study": TierExpectation("gpu", required=True),
    },
    # Low-power: Core on CPU, Prime unloaded
    LifecycleState.SLEEP: {
        "core":  TierExpectation("cpu", required=True),
        "nano":  TierExpectation("unloaded", required=False),
        "prime": TierExpectation("unloaded", required=False),
        "study": TierExpectation("unloaded", required=False),
    },
    # Everything off — Groq/API fallback only
    LifecycleState.DEEP_SLEEP: {
        "core":  TierExpectation("unloaded", required=False),
        "nano":  TierExpectation("unloaded", required=False),
        "prime": TierExpectation("unloaded", required=False),
        "study": TierExpectation("unloaded", required=False),
    },
    # Gear P: Core on CPU (sentinel), GPU empty for fast clutch-in
    LifecycleState.PARKED: {
        "core":  TierExpectation("cpu", required=True),
        "nano":  TierExpectation("unloaded", required=False),
        "prime": TierExpectation("unloaded", required=False),
        "study": TierExpectation("unloaded", required=False),
    },
    LifecycleState.TRANSITIONING: {
        # No expectations — clutch engaged, handoff in progress
    },
}


# ── Gear Metadata ────────────────────────────────────────────────────────────
# Human-readable gear descriptions for dashboards and logging.

GEAR_INFO: Dict[LifecycleState, dict] = {
    LifecycleState.PARKED: {
        "gear": "P", "name": "Parked",
        "description": "Sentinel standby. Core on CPU (GGUF). GPU empty.",
        "vram_estimate_mb": 0,
    },
    LifecycleState.AWAKE: {
        "gear": "1", "name": "Operator",
        "description": "Core E4B on GPU (NF4). Rapid chat, vision, audio, tools.",
        "vram_estimate_mb": 8800,
    },
    LifecycleState.LISTENING: {
        "gear": "1+", "name": "Operator + Audio",
        "description": "AWAKE with active audio STT pipeline.",
        "vram_estimate_mb": 8800,
    },
    LifecycleState.FOCUSING: {
        "gear": "2", "name": "Sovereign",
        "description": "Prime 26B-A4B on GPU (Expert Buffered). Deep reasoning.",
        "vram_estimate_mb": 4600,
    },
    LifecycleState.MEDITATION: {
        "gear": "T", "name": "Training",
        "description": "Study owns GPU for QLoRA/merge. All tiers off.",
        "vram_estimate_mb": 0,  # study manages its own VRAM
    },
    LifecycleState.SLEEP: {
        "gear": "S", "name": "Sleep",
        "description": "Core on CPU. Prime unloaded. Low-power standby.",
        "vram_estimate_mb": 0,
    },
    LifecycleState.DEEP_SLEEP: {
        "gear": "0", "name": "Deep Sleep",
        "description": "Everything unloaded. Groq/API fallback only.",
        "vram_estimate_mb": 0,
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def validate_transition(
    current: LifecycleState,
    trigger: TransitionTrigger,
    target: Optional[LifecycleState] = None,
) -> Optional[LifecycleState]:
    """Validate and resolve a transition. Returns target state or None if invalid.

    For USER_REQUEST triggers, `target` must be specified.
    For all other triggers, the target is determined by the transition table.
    """
    if current == LifecycleState.TRANSITIONING:
        return None  # Can't transition while already transitioning

    if trigger == TransitionTrigger.USER_REQUEST:
        if target is None:
            return None
        valid_targets = USER_REQUEST_TARGETS.get(current, set())
        return target if target in valid_targets else None

    table = TRANSITIONS.get(current, {})
    return table.get(trigger)


def available_transitions(current: LifecycleState) -> List[dict]:
    """Return list of available transitions from the current state.

    Each item: {"trigger": str, "target": str} or
               {"trigger": "user_request", "targets": [str, ...]}
    """
    if current == LifecycleState.TRANSITIONING:
        return []

    result = []
    table = TRANSITIONS.get(current, {})
    for trigger, target in table.items():
        result.append({
            "trigger": trigger.value,
            "target": target.value,
        })

    # Add user_request targets
    user_targets = USER_REQUEST_TARGETS.get(current, set())
    if user_targets:
        result.append({
            "trigger": TransitionTrigger.USER_REQUEST.value,
            "targets": sorted(t.value for t in user_targets),
        })

    return result


def get_gear_info(state: LifecycleState) -> dict:
    """Return gear metadata for a lifecycle state."""
    return GEAR_INFO.get(state, {
        "gear": "?", "name": state.value,
        "description": "Unknown state",
        "vram_estimate_mb": 0,
    })

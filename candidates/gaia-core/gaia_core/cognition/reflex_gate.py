"""Reflex skip-gate (GAIA_Project-19i).

The speculative reflex pre-flight in `main.py` serves simple prompts
via Nano in ~100-300ms before agent_core's full model-selection runs.
That's a big latency win in the common case but silently downgrades
two scenarios where the user has explicitly chosen a heavier tier:

  1. Explicit tier prefix in the prompt:
     - `prime:`, `thinker:`, `oracle:`  (and `[prime]`, `::prime`, ...
        variants matching agent_core's recognizer)
     The user typed these to override default routing.

  2. Packet carries `knowledge_base_name`:
     Non-trivial retrieval-grounded work; the reflex can't honor that
     context — it would generate from Nano's prior, ungrounded.

  3. Orchestrator lifecycle state is FOCUSING:
     Prime is loaded on GPU. The gear-shift cost has been paid; the
     user / orchestrator is intentionally on Prime. Reflex would
     downgrade.

`should_skip_reflex(user_input, packet, lifecycle_client)` returns a
short reason string for logging if any of the above triggers, or None
to let the reflex run.
"""

from __future__ import annotations

from typing import Optional


# Tier-override tags. Must mirror the recognizer in
# agent_core's model selector at line ~1736 — keep these lists in sync.
_TIER_OVERRIDE_TAGS: tuple[str, ...] = (
    "prime:",   "[prime]",   "::prime",
    "thinker:", "[thinker]", "::thinker",
    "oracle:",  "[oracle]",  "::oracle",
)


def _has_tier_prefix(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(tag in t for tag in _TIER_OVERRIDE_TAGS)


def _has_kb_field(packet) -> bool:
    """True iff packet.content.data_fields has a non-empty knowledge_base_name."""
    if packet is None:
        return False
    try:
        fields = packet.content.data_fields or []
    except Exception:
        return False
    for f in fields:
        try:
            if getattr(f, "key", None) == "knowledge_base_name" and getattr(f, "value", None):
                return True
        except Exception:
            continue
    return False


def _is_focusing(lifecycle_client) -> bool:
    """True iff the cached lifecycle state is FOCUSING.

    Uses the client's cached `current_state` property — no network call
    on the hot path. If the cache is stale by a single tick the
    consequence is just one reflex run that should've been skipped, or
    one reflex skip that wasn't required; not a safety issue.
    """
    if lifecycle_client is None:
        return False
    try:
        state = lifecycle_client.current_state
    except Exception:
        return False
    # LifecycleState enum has .name; str() falls back to "LifecycleState.FOCUSING"
    name = getattr(state, "name", None)
    if name == "FOCUSING":
        return True
    return str(state).endswith("FOCUSING")


def should_skip_reflex(user_input: str, packet, lifecycle_client) -> Optional[str]:
    """Return a reason string if reflex should be skipped, else None.

    The reason string is intended for logging; it tells you WHY the
    reflex was bypassed. None means "reflex is allowed to run".
    """
    if _has_tier_prefix(user_input):
        return "explicit tier prefix in user input"
    if _has_kb_field(packet):
        return "knowledge_base_name set"
    if _is_focusing(lifecycle_client):
        return "lifecycle state FOCUSING"
    return None

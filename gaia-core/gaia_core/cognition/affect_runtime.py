"""Runtime affect surface (GAIA_Project-usv Phase 2).

Reads the live affect vector from the World Model KG and renders it
into:

  - `affect_state_lines()` — short prompt fragment for inclusion in
    the system prompt's identity block (additive over static persona
    traits).
  - `affect_inference_params()` — modulation hints for the inference
    dispatcher (temperature, max_tokens, escalate-to-Prime). Phase 2
    surfaces these as a dict; wiring into the actual tier-selector /
    sampler is downstream of this commit but uses the same shape.

Designed to be safe to call from anywhere on the hot path: any failure
(KG missing, db locked, malformed triples) is swallowed with a debug
log and the call returns a no-op snapshot, so prompt building can
never fail because of affect.

Phase 1 (data layer) lives in `gaia_common.utils.affect_kg`. This
module is the Phase-2 consumer.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

logger = logging.getLogger("GAIA.AffectRuntime")


# Single shared AffectKG instance. Lazy — first call builds it.
_affect_kg = None
_lock = threading.Lock()


def _get_affect_kg():
    """Lazy-init the shared AffectKG. Returns None if init fails."""
    global _affect_kg
    if _affect_kg is not None:
        return _affect_kg
    with _lock:
        if _affect_kg is not None:
            return _affect_kg
        try:
            from gaia_common.utils.knowledge_graph import KnowledgeGraph
            from gaia_common.utils.affect_kg import AffectKG
            _affect_kg = AffectKG(KnowledgeGraph())
        except Exception:
            logger.debug("AffectKG init failed; affect runtime disabled", exc_info=True)
            _affect_kg = None
    return _affect_kg


def reset_for_tests(affect_kg=None) -> None:
    """Replace or clear the cached AffectKG. Tests only."""
    global _affect_kg
    with _lock:
        _affect_kg = affect_kg


# ── Snapshot ────────────────────────────────────────────────────────

_EMPTY_SNAPSHOT = {
    "traits": {}, "feels": {}, "drives": {},
    "curious_about": {}, "tired_of": {},
    "active_context": None, "as_of": None,
}


def current_affect_snapshot(active_context: Optional[str] = None) -> dict:
    """Read the current affect vector. Returns _EMPTY_SNAPSHOT on any failure."""
    af = _get_affect_kg()
    if af is None:
        return dict(_EMPTY_SNAPSHOT)
    try:
        return af.flatten_current_affect(active_context=active_context)
    except Exception:
        logger.debug("flatten_current_affect failed", exc_info=True)
        return dict(_EMPTY_SNAPSHOT)


# ── Prompt rendering ────────────────────────────────────────────────

# Only surface axes with at least this much energy — avoids polluting
# the prompt with near-zero noise from heavily decayed triples.
_PROMPT_THRESHOLD = 0.15

# Hard cap on items per axis. Prevents a runaway accumulation of
# `curious_about_*` triples from blowing up the system prompt.
_MAX_ITEMS_PER_AXIS = 4


def _fmt_kv(name: str, value: float) -> str:
    return f"{name}={value:.2f}"


def _top_items(axis_dict: dict, k: int = _MAX_ITEMS_PER_AXIS) -> list[tuple[str, float]]:
    items = [(n, v) for n, v in axis_dict.items() if v >= _PROMPT_THRESHOLD]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:k]


def affect_state_lines(snapshot: Optional[dict] = None) -> list[str]:
    """Render the affect snapshot as a short list of system-prompt lines.

    Returns an empty list when the snapshot is effectively empty. The
    output is meant to slot into the existing identity_lines block in
    `prompt_builder.build_from_packet` right after static persona traits.

    Format (mirrors the existing 'Traits: a: 0.9, b: 0.7' line style):

      Current Affect (feels): irritation=0.62, curious=0.55
      Current Affect (drives): hunger_for_novelty=0.71
      Current Affect (focus): consistency_detector=0.84
      Current Affect (aversion): dnd_session=0.42

    No-op axes are omitted entirely. Traits are NOT re-rendered here —
    the static persona JSON already populates them on the prior line.
    """
    if snapshot is None:
        snapshot = current_affect_snapshot()

    lines: list[str] = []

    feels = _top_items(snapshot.get("feels", {}))
    if feels:
        lines.append(
            "Current Affect (feels): "
            + ", ".join(_fmt_kv(n, v) for n, v in feels)
        )

    drives = _top_items(snapshot.get("drives", {}))
    if drives:
        lines.append(
            "Current Affect (drives): "
            + ", ".join(_fmt_kv(n, v) for n, v in drives)
        )

    curious = _top_items(snapshot.get("curious_about", {}))
    if curious:
        lines.append(
            "Current Affect (focus): "
            + ", ".join(_fmt_kv(n, v) for n, v in curious)
        )

    tired = _top_items(snapshot.get("tired_of", {}))
    if tired:
        lines.append(
            "Current Affect (aversion): "
            + ", ".join(_fmt_kv(n, v) for n, v in tired)
        )

    return lines


# ── Inference modulation ────────────────────────────────────────────

# Public modulation shape:
#   {
#     "temperature_delta": float,      # added to base temperature; clamped at the call site
#     "max_tokens_multiplier": float,  # multiplicative scale
#     "escalate_to_prime": bool,       # hint to the tier selector
#     "style_hint": str | None,        # human-readable mood label for downstream prompts
#     "reasons": list[str],            # why each adjustment was made (for audit)
#   }

_DEFAULT_MOD = {
    "temperature_delta": 0.0,
    "max_tokens_multiplier": 1.0,
    "escalate_to_prime": False,
    "style_hint": None,
    "reasons": [],
}


def affect_inference_params(snapshot: Optional[dict] = None) -> dict:
    """Derive inference modulation hints from the affect snapshot.

    Heuristics (intentionally simple, easy to tune):

      - High `caution` trait (≥0.7) AND high `logic_priority` trait
        (≥0.7) → escalate_to_prime. The pairing says "this is a
        careful, reasoning-heavy moment" — exactly Prime's job.
      - High `feels=irritation` (≥0.6) → cap temperature (delta -0.2),
        style_hint='measured'. Don't let a bad mood improvise.
      - High `feels=curiosity` OR `feels=excited` (≥0.6) → expand
        max_tokens (×1.3), style_hint='exploratory'.
      - High `curious_about=<anything>` (≥0.7) → small max_tokens
        bump (×1.15). She wants to chase a topic — give her room.
      - High `feels=fatigue` (≥0.6) → contract max_tokens (×0.8),
        style_hint='terse'.

    Returns a dict copy of _DEFAULT_MOD with adjustments applied. Caller
    is expected to interpret the hints; this module does not directly
    touch the sampler.
    """
    if snapshot is None:
        snapshot = current_affect_snapshot()

    mod = dict(_DEFAULT_MOD)
    mod["reasons"] = []  # fresh list

    traits = snapshot.get("traits", {})
    feels = snapshot.get("feels", {})
    curious = snapshot.get("curious_about", {})

    caution = traits.get("caution", 0.0)
    logic = traits.get("logic_priority", 0.0)
    if caution >= 0.7 and logic >= 0.7:
        mod["escalate_to_prime"] = True
        mod["reasons"].append(
            f"caution={caution:.2f} & logic_priority={logic:.2f} → escalate"
        )

    irritation = feels.get("irritation", 0.0)
    if irritation >= 0.6:
        mod["temperature_delta"] -= 0.2
        mod["style_hint"] = "measured"
        mod["reasons"].append(f"feels.irritation={irritation:.2f} → cap temperature")

    excitement = max(feels.get("curiosity", 0.0), feels.get("excited", 0.0))
    if excitement >= 0.6 and not mod["style_hint"]:
        mod["max_tokens_multiplier"] *= 1.3
        mod["style_hint"] = "exploratory"
        mod["reasons"].append(
            f"feels.curiosity/excited={excitement:.2f} → exploratory"
        )

    top_curious = max(curious.values(), default=0.0)
    if top_curious >= 0.7:
        mod["max_tokens_multiplier"] *= 1.15
        mod["reasons"].append(
            f"curious_about.max={top_curious:.2f} → +15% max_tokens"
        )

    fatigue = feels.get("fatigue", 0.0)
    if fatigue >= 0.6:
        mod["max_tokens_multiplier"] *= 0.8
        # Fatigue overrides earlier style hint — terseness wins.
        mod["style_hint"] = "terse"
        mod["reasons"].append(f"feels.fatigue={fatigue:.2f} → terse")

    return mod


# ── Hot-path-safe wrapper for prompt builder ────────────────────────

def render_into_identity_lines(identity_lines: list[str],
                                active_context: Optional[str] = None) -> None:
    """Append affect lines to an identity_lines list in-place.

    Designed for the exact pattern used in
    `prompt_builder.build_from_packet`, where identity_lines is built up
    iteratively. Never raises — failures are logged at debug level.
    """
    try:
        snapshot = current_affect_snapshot(active_context=active_context)
        for line in affect_state_lines(snapshot):
            identity_lines.append(line)
    except Exception:
        logger.debug("affect render failed; skipping", exc_info=True)

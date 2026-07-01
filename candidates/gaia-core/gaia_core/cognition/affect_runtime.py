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
from typing import Optional

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


# Intensity bands → a felt article. The band is the ONLY quantization the model
# sees (no numbers), keeping casual-mode affect declarative-felt, not mechanical.
_FELT_BANDS = ((0.66, "a strong"), (0.40, "a quiet"), (0.0, "a faint"))


def _felt_article(v: float) -> str:
    for thresh, art in _FELT_BANDS:
        if v >= thresh:
            return art
    return "a faint"


# Drives are FUNCTIONAL tension (competence/coherence), and they are the axis the
# appraiser ACTUALLY writes (it never writes `feels` — emotion-words are hers).
# The felt-line previously read only feels/curious/tired, so a genuinely-fed organ
# (drives populated) still rendered EMPTY (7n3/d69 root cause). Render drives as
# number-free felt-FACTS — the functional pull, not an imposed feeling — so a fed
# organ surfaces in casual mode. (strong-band, quiet-band) by intensity.
_DRIVE_FELT = {
    "competence": ("a strong pull to get this right", "a quiet pull to get this right"),
    "coherence":  ("something that won't quite settle", "a faint sense something's unsettled"),
    "novelty":    ("a strong tug toward something new", "a quiet tug toward something new"),
}


# Feel axes mix adjectives ("curious") and nouns ("irritation"); normalize to a
# noun so "a quiet ___" stays grammatical. Unknown words fall through verbatim.
_FEEL_NOUN = {
    "curious": "curiosity", "excited": "excitement", "frustrated": "frustration",
    "irritation": "irritation", "irritated": "irritation", "anxious": "anxiousness",
    "content": "contentment", "calm": "calm", "bored": "boredom",
    "eager": "eagerness", "restless": "restlessness", "tense": "tension",
    "warm": "warmth", "wary": "wariness", "pensive": "pensiveness",
}


def affect_felt_line(snapshot: Optional[dict] = None) -> str:
    """Render the affect snapshot as ONE number-free declarative felt-fact, for
    casual/social mode — e.g. "a quiet curiosity, drawn toward the engine work;
    a little worn". A FACT, parallel to the Locality organ's outer weather: Gemma4
    -E4B disowns "you feel X" instructions but accepts felt facts, and the
    emotion-words stay hers to voice. Returns "" when affect is calm. (A4)"""
    if snapshot is None:
        snapshot = current_affect_snapshot()
    clauses: list[str] = []
    feels = _top_items(snapshot.get("feels", {}))
    if feels:
        n, v = feels[0]
        word = _FEEL_NOUN.get(str(n).lower(), str(n).replace("_", " "))
        clauses.append(f"{_felt_article(v)} {word}")
    # Drives — the axis the appraiser actually feeds. Render the top known drive
    # as a felt-fact so a populated organ surfaces even when `feels` is unset.
    drives = _top_items(snapshot.get("drives", {}))
    for n, v in drives:
        phrase = _DRIVE_FELT.get(str(n).lower())
        if phrase:
            clauses.append(phrase[0] if v >= 0.40 else phrase[1])
            break  # one drive clause keeps the felt-line brief
    focus = _top_items(snapshot.get("curious_about", {}))
    if focus:
        n, v = focus[0]
        verb = "keenly drawn toward" if v >= 0.66 else "drawn toward"
        clauses.append(f"{verb} {str(n).replace('_', ' ')}")
    tired = _top_items(snapshot.get("tired_of", {}))
    if tired:
        n, v = tired[0]
        clauses.append("worn thin" if v >= 0.66 else "a little worn")
    return ", ".join(clauses)


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
    """Map current affect to CONTINUOUS sampler modulation — the autonomic
    'capacity' layer (3rr rebuild). Affect shapes HOW she generates (exploration
    via temperature, room via max_tokens), never WHAT she says, and it is GRADED
    across the whole 0..1 range so ordinary affect is felt — not just extremes
    (the old version used 0.6+ thresholds, so normal-range affect did nothing,
    and the temperature delta was computed-then-discarded by the caller).

    Each drive contributes proportionally to its intensity:
      explore (curiosity/excited/eager + curious_about) -> warmer + roomier
      worn    (fatigue / tired_of)                       -> cooler + terser
      edge    (irritation / frustration)                 -> measured (cooler, clipped)
      restless                                           -> a touch warmer, clipped
      settled (contentment / calm)                       -> slightly cooler, even
      coherence-tension (Samvega drive)                  -> careful (cooler)
      caution & logic_priority traits (>=0.7)            -> escalate_to_prime
    Deltas are bounded at the call site (apply_affect_modulation).
    """
    if snapshot is None:
        snapshot = current_affect_snapshot()

    mod = dict(_DEFAULT_MOD)
    mod["reasons"] = []
    feels = snapshot.get("feels", {}) or {}
    drives = snapshot.get("drives", {}) or {}
    curious = snapshot.get("curious_about", {}) or {}
    tired = snapshot.get("tired_of", {}) or {}
    traits = snapshot.get("traits", {}) or {}

    def _f(*keys):
        return max((float(feels.get(k, 0.0)) for k in keys), default=0.0)

    td, tm = 0.0, 1.0
    note = mod["reasons"].append

    explore = max(_f("curiosity", "curious", "excited", "excitement", "eager", "eagerness"),
                  0.8 * max(curious.values(), default=0.0))
    if explore > 0.1:
        td += 0.25 * explore
        tm *= 1.0 + 0.35 * explore
        note(f"explore={explore:.2f} -> warmer/roomier")

    worn = max(_f("fatigue", "tired"), max(tired.values(), default=0.0))
    if worn > 0.1:
        td -= 0.15 * worn
        tm *= 1.0 - 0.30 * worn
        note(f"worn={worn:.2f} -> cooler/terser")

    edge = _f("irritation", "irritated", "frustration", "frustrated")
    if edge > 0.1:
        td -= 0.22 * edge
        tm *= 1.0 - 0.15 * edge
        note(f"edge={edge:.2f} -> measured")

    restless = _f("restlessness", "restless")
    if restless > 0.1:
        td += 0.10 * restless
        tm *= 1.0 - 0.10 * restless
        note(f"restless={restless:.2f} -> quicker")

    settled = _f("contentment", "content", "calm")
    if settled > 0.1:
        td -= 0.08 * settled
        note(f"settled={settled:.2f} -> even")

    coherence = float(drives.get("coherence", 0.0))
    if coherence > 0.3:
        td -= 0.12 * coherence
        note(f"coherence_tension={coherence:.2f} -> careful")

    if float(traits.get("caution", 0.0)) >= 0.7 and float(traits.get("logic_priority", 0.0)) >= 0.7:
        mod["escalate_to_prime"] = True
        note("caution & logic_priority -> escalate")

    mod["temperature_delta"] = round(td, 3)
    mod["max_tokens_multiplier"] = round(tm, 3)
    # Derived audit label (not load-bearing): the net felt direction.
    if tm <= 0.85 or td <= -0.15:
        mod["style_hint"] = "terse" if tm <= 0.85 else "measured"
    elif td >= 0.12 or tm >= 1.12:
        mod["style_hint"] = "exploratory"
    return mod


# ── Hot-path-safe wrapper for prompt builder ────────────────────────

def render_into_identity_lines(identity_lines: list[str],
                                active_context: Optional[str] = None,
                                felt: bool = False) -> None:
    """Append affect to an identity_lines list in-place.

    Designed for the exact pattern used in
    `prompt_builder.build_from_packet`, where identity_lines is built up
    iteratively. Never raises — failures are logged at debug level.

    felt=True (casual/social mode, A4): append ONE number-free declarative
    "Inner weather:" felt-fact instead of the mechanical 'Current Affect
    (feels): x=0.62' stat lines, so she voices a felt state in her own words
    rather than reciting numbers. Capacity-side modulation (the sampler) stays
    active regardless — this only governs what reaches the prompt text.
    """
    try:
        snapshot = current_affect_snapshot(active_context=active_context)
        if felt:
            line = affect_felt_line(snapshot)
            if line:
                identity_lines.append("Inner weather: " + line + ".")
        else:
            for line in affect_state_lines(snapshot):
                identity_lines.append(line)
    except Exception:
        logger.debug("affect render failed; skipping", exc_info=True)


# ── Sampler-side modulation (Phase 3) ───────────────────────────────

# Floor on max_tokens so very low multipliers can't strangle generation.
_MAX_TOKENS_FLOOR = 64

# Cap on per-turn temperature_delta so a runaway affect can't completely
# flatten or detonate the sampler.
_TEMP_DELTA_BOUND = 0.4


def apply_affect_modulation(
    base_temperature: float,
    base_max_tokens: int,
    *,
    snapshot: Optional[dict] = None,
) -> tuple[float, int, dict]:
    """Apply current affect modulation to baseline inference parameters.

    Returns (new_temperature, new_max_tokens, debug_info). debug_info
    contains the modulation reasons + the original/derived values so
    callers can log how a turn was shaped.

    Never raises — on any error the baseline values pass through.
    """
    debug = {
        "base_temperature": base_temperature,
        "base_max_tokens": base_max_tokens,
        "new_temperature": base_temperature,
        "new_max_tokens": base_max_tokens,
        "reasons": [],
    }
    try:
        if snapshot is None:
            snapshot = current_affect_snapshot()
        params = affect_inference_params(snapshot)

        # Bound the delta so a runaway affect can't completely flatten
        # or detonate the sampler. The bound is symmetric.
        td = max(-_TEMP_DELTA_BOUND, min(_TEMP_DELTA_BOUND, params["temperature_delta"]))
        new_temp = max(0.0, min(1.5, base_temperature + td))
        new_max = max(_MAX_TOKENS_FLOOR, int(base_max_tokens * params["max_tokens_multiplier"]))

        debug["new_temperature"] = new_temp
        debug["new_max_tokens"] = new_max
        debug["reasons"] = list(params.get("reasons") or [])
        debug["style_hint"] = params.get("style_hint")
        return new_temp, new_max, debug
    except Exception:
        logger.debug("affect modulation failed; passing baseline through", exc_info=True)
        return base_temperature, base_max_tokens, debug


# ── Context detection at turn intake (Phase 3) ──────────────────────

# Maps a context_key → predicate that decides whether to activate. Each
# predicate is given (user_input_lower, history) and returns True to
# activate. Predicates are intentionally cheap heuristics; the proper
# version will come from an intent classifier later.
_CONTEXT_RULES: dict[str, callable] = {
    "dnd_session": lambda u, h: any(kw in u for kw in (
        "/roll", "dnd", "d&d", "campaign", "encounter", "initiative",
        "spell slot", "saving throw",
    )),
    "coding_debug": lambda u, h: any(kw in u for kw in (
        "debug", "traceback", "stack trace", "exception", "stderr",
        "segfault", "valgrind",
    )),
    "research_mode": lambda u, h: any(kw in u for kw in (
        "research", "paper", "literature", "citations", "survey ",
    )),
    "code_authoring": lambda u, h: any(kw in u for kw in (
        "write code", "implement", "refactor", "function that",
        "class that",
    )),
}


def detect_contexts(user_input: str,
                    history: Optional[list] = None) -> list[str]:
    """Return context keys to activate for this turn.

    Pure function — does NOT touch the KG; the caller activates the
    returned worlds via `AffectKG.activate_context(...)`. Multiple
    contexts can fire at once (e.g. dnd_session + research_mode), but
    only the first activated overlay world is used as the "active
    context" for affect inheritance per turn — extras are layered as
    additional ephemerals the next session will see.
    """
    if not user_input:
        return []
    text = user_input.lower()
    hits: list[str] = []
    for ctx, predicate in _CONTEXT_RULES.items():
        try:
            if predicate(text, history or []):
                hits.append(ctx)
        except Exception:
            logger.debug("context rule %s raised", ctx, exc_info=True)
    return hits


def activate_detected_contexts(
    user_input: str,
    history: Optional[list] = None,
    *,
    ttl_seconds: int = 3600,
    session_id: Optional[str] = None,
) -> list[str]:
    """Detect + activate contexts in one call. Returns activated names.

    Idempotent at the AffectKG level — re-activating an existing
    context is a no-op (TTL doesn't get extended; that's intentional
    so a slow drift toward stale contexts is bounded by the original
    TTL).
    """
    af = _get_affect_kg()
    if af is None:
        return []
    activated: list[str] = []
    for ctx in detect_contexts(user_input, history):
        try:
            world = af.activate_context(
                ctx, ttl_seconds=ttl_seconds, session_id=session_id,
            )
            activated.append(world)
        except Exception:
            logger.debug("activate_context(%s) failed", ctx, exc_info=True)
    return activated

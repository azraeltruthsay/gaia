"""Predicate vocabulary for the affect model (GAIA_Project-usv).

Affect state lives in the existing KnowledgeGraph as triples with a
**prefixed-predicate + sentinel-object** scheme:

  (self, feels_irritation, affect_state, conf=0.3, valid_from=...)
  (self, feels_calm,       affect_state, conf=0.5, valid_from=...)
  (self, trait_curiosity,  persona_state, conf=0.9, ...)
  (self, drive_hunger_for_novelty, drive_state, conf=0.7, ...)
  (self, curious_about_consistency_detector, attention_state, conf=0.85, ...)

This dodges two pre-existing constraints in `KnowledgeGraph.add_triple`:

  1. **Same-predicate / different-object contradictions**: GAIA must hold
     multiple simultaneous feelings (irritation + curiosity at once). If
     we used a flat `feels` predicate with different emotion objects,
     each new emotion would conflict with the prior one and trigger the
     "update" path (newer supersedes older), wiping the coexisting
     feelings. Prefixing the predicate puts each affect axis in its own
     name → no contradictions between affects.

  2. **Dedup of unchanged open triples**: re-recording an existing
     (subject, predicate, object) with valid_to NULL returns the
     existing id and ignores the new confidence. Intensity updates must
     therefore close (valid_to = now) the prior open triple first, then
     insert a new one. `AffectKG._update_affect` does this.

See: knowledge/blueprints/affect_model.md
"""

from __future__ import annotations

import re

# ── Canonical subject ───────────────────────────────────────────────
SELF = "self"


# ── Sentinel objects per affect class ───────────────────────────────
# Object is intentionally a fixed sentinel per affect class — the
# semantic axis lives in the predicate name. This keeps the KG's
# entity table small and makes querying by axis trivial.
OBJ_AFFECT_STATE = "affect_state"      # feels_*
OBJ_PERSONA_STATE = "persona_state"    # trait_*
OBJ_DRIVE_STATE = "drive_state"        # drive_*
OBJ_ATTENTION_STATE = "attention_state"  # curious_about_*, tired_of_*


# ── Predicate prefixes ──────────────────────────────────────────────
PREFIX_FEELS = "feels_"
PREFIX_TRAIT = "trait_"
PREFIX_DRIVE = "drive_"
PREFIX_CURIOUS_ABOUT = "curious_about_"
PREFIX_TIRED_OF = "tired_of_"
PREFIX_BELIEVES_ABOUT = "believes_about_"  # theory-of-mind


# Reverse map for the flattener: prefix → (bucket_name_in_result, sentinel)
AFFECT_PREFIX_TABLE: list[tuple[str, str, str]] = [
    # (prefix, result_bucket, expected_sentinel)
    (PREFIX_FEELS,         "feels",          OBJ_AFFECT_STATE),
    (PREFIX_TRAIT,         "traits",         OBJ_PERSONA_STATE),
    (PREFIX_DRIVE,         "drives",         OBJ_DRIVE_STATE),
    (PREFIX_CURIOUS_ABOUT, "curious_about",  OBJ_ATTENTION_STATE),
    (PREFIX_TIRED_OF,      "tired_of",       OBJ_ATTENTION_STATE),
]


# ── Decay half-lives (seconds) ──────────────────────────────────────
# Used by the placeholder decay until Stage 7 (lw4) ships the proper
# kernel. Keyed by predicate **prefix** since prefixes carry the
# semantic class.
HALFLIFE_BY_PREFIX: dict[str, float] = {
    PREFIX_FEELS:         12 * 3600,        # feelings fade through a workday
    PREFIX_DRIVE:         24 * 3600,        # drives renew daily
    PREFIX_CURIOUS_ABOUT: 18 * 3600,        # curiosity survives a sleep
    PREFIX_TIRED_OF:      18 * 3600,        # aversion same scale
    PREFIX_TRAIT:    7 * 24 * 3600,         # base personality drifts slowly
    PREFIX_BELIEVES_ABOUT: 3 * 24 * 3600,   # theory-of-mind beliefs refresh
}


def halflife_for(predicate: str) -> float | None:
    """Return decay halflife (seconds) for a given affect predicate."""
    for prefix, hl in HALFLIFE_BY_PREFIX.items():
        if predicate.startswith(prefix):
            return hl
    return None


# ── Predicate constructors ──────────────────────────────────────────
# Canonicalizing helpers — keeps the prefix-suffix joinery in one place
# so callers can use plain emotion/trait names.

_SAFE_RE = re.compile(r"[^a-z0-9_]+")


def _canon_suffix(name: str) -> str:
    """Lowercase, snake-case, strip non-alphanumeric. Raises on empty."""
    s = _SAFE_RE.sub("_", name.strip().lower()).strip("_")
    if not s:
        raise ValueError(f"Empty/invalid affect suffix: {name!r}")
    return s


def pred_feels(emotion: str) -> str:
    return PREFIX_FEELS + _canon_suffix(emotion)


def pred_trait(trait: str) -> str:
    return PREFIX_TRAIT + _canon_suffix(trait)


def pred_drive(drive: str) -> str:
    return PREFIX_DRIVE + _canon_suffix(drive)


def pred_curious_about(topic: str) -> str:
    return PREFIX_CURIOUS_ABOUT + _canon_suffix(topic)


def pred_tired_of(topic: str) -> str:
    return PREFIX_TIRED_OF + _canon_suffix(topic)


def pred_believes_about(attribute: str) -> str:
    return PREFIX_BELIEVES_ABOUT + _canon_suffix(attribute)


# ── World naming conventions ────────────────────────────────────────
CONTEXT_WORLD_PREFIX = "ctx_"
BELIEF_OF_WORLD_PREFIX = "belief_of_"


def context_world_name(context_key: str) -> str:
    """Canonicalize a context name into its overlay world id."""
    key = _canon_suffix(context_key)
    if key.startswith(CONTEXT_WORLD_PREFIX):
        return key
    return CONTEXT_WORLD_PREFIX + key


def belief_of_world_name(person: str) -> str:
    """Canonicalize a person name into their theory-of-mind world id."""
    key = _canon_suffix(person)
    if key.startswith(BELIEF_OF_WORLD_PREFIX):
        return key
    return BELIEF_OF_WORLD_PREFIX + key


# ── Modality value for context overlays ─────────────────────────────
MODALITY_CONTEXT = "context"

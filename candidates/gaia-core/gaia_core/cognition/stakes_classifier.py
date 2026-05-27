"""Multi-axis stakes classifier (GAIA_Project-6ho).

Disambiguates user utterances along three axes when role-play personas
or fictional contexts are active:

  1. IDENTITY:        which "I" is speaking — Azrael, his character,
                      a quoted source, or a hypothetical.
  2. PROPRIOCEPTIVE:  which embodiment frame — real body vs character body.
  3. STAKES:          real-world tempo vs narrative tempo.

The load-bearing case is **ambiguous + safety-coded** — "I broke my leg"
during a D&D session. Wrong default fails one of two ways:
  - Treat as in-game when it's real → miss a medical emergency.
  - Treat as real when it's in-game → break immersion + waste time.

Architecture: a small heuristic classifier (no LLM call on the hot path).
Triggers on hand-curated vocabularies for safety, game mechanics, and
embodiment-ambiguous body parts. Three resolutions:

  - unambiguous_real    — handle as real concern
  - unambiguous_in_game — handle in-character
  - ambiguous           — ASK for clarification (load-bearing case)

This module is **classification only** — it does not produce the
clarifying question or change agent behavior. Callers decide what to do
with the result. Phase 2 of 6ho wires this into agent_core's turn
intake; Phase 3 (follow-up issue) builds the user-facing "which leg do
you mean?" UX.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ── Vocabularies ────────────────────────────────────────────────────
#
# All matched as whole-word, case-insensitive. Keep narrow — false
# positives cascade into spurious clarification requests, which is
# almost as bad as missing the real signal.

# Real-world safety / medical / embodiment urgency.
_SAFETY_TERMS = frozenset({
    "bleeding", "broke", "broken", "fractured", "hurts", "hurt",
    "injured", "injury", "wound", "wounded", "burning", "burned",
    "burnt", "scalded", "concussed", "concussion",
    "can't breathe", "cant breathe", "can't move", "cant move",
    "emergency", "dying", "poisoned", "allergic", "anaphylaxis",
    "fever", "vomiting", "throwing up", "passing out",
    "fainted", "fainting", "blacking out",
    "chest pain", "heart attack", "stroke",
    "overdose", "overdosed", "suicidal",
    "sick", "nauseous",
})

# Game-mechanic terms — strong in-game signal.
_GAME_MECHANIC_TERMS = frozenset({
    "hp", "hit point", "hit points", "ac", "armor class",
    "saving throw", "save", "dc",
    "initiative", "advantage", "disadvantage",
    "crit", "critical hit", "natural 20", "nat 20", "nat 1", "natural 1",
    "spell slot", "spell slots", "cantrip", "cantrips",
    "concentration check", "death save", "death saves",
    "round", "turn order", "bonus action", "reaction",
    "mana", "mp", "stamina", "level up", "leveled up", "xp",
    "exp points", "experience points",
    "healing potion", "potion of healing", "cure wounds",
    "fireball", "magic missile", "eldritch blast",
    "the party", "the dm", "dungeon master",
    "rolled a", "roll for", "rolled",
    "respawn", "respawned", "checkpoint",
    "1st level", "2nd level", "3rd level", "4th level", "5th level",
    "first level", "second level", "third level",
    "out of spells",
})

# Body parts that are embodiment-ambiguous when role-play is active.
# These ALONE don't trigger ambiguity — they need a safety co-trigger
# AND a role-play context for the AMBIGUOUS resolution to fire.
_BODY_PART_TERMS = frozenset({
    "arm", "arms", "leg", "legs", "hand", "hands", "foot", "feet",
    "head", "neck", "back", "chest", "shoulder", "shoulders",
    "knee", "knees", "elbow", "elbows", "ankle", "ankles",
    "wrist", "wrists", "finger", "fingers", "toe", "toes",
    "eye", "eyes", "ear", "ears",
})

# Phrases that signal explicit quoting or hypothetical framing.
_QUOTED_OR_HYPOTHETICAL_PREFIXES = (
    "she said", "he said", "they said", "she says", "he says",
    "they say", "quoted", "wrote", "tweeted", "posted",
    "if i were", "if i was", "imagine if", "suppose i",
    "hypothetically", "hypothetical",
)

# Phrases that strongly anchor an utterance as in-character (RP voice
# tags, "in-character" markers, etc.).
_IN_CHARACTER_MARKERS = (
    "in-character", "in character", "ic:",
    "my character", "[ic]", "/ic ",
    "as my character", "as rupert", "as my pc",
)

# Phrases that anchor out-of-character / real-world.
_OUT_OF_CHARACTER_MARKERS = (
    "out-of-character", "out of character", "ooc:",
    "[ooc]", "/ooc ", "irl:", "in real life", "actually,",
    "actually i", "actually im", "actually i'm",
    "for real", "no joke",
)


# Stakes / identity / proprioceptive enums (strings — easy JSON serialize)
STAKES_REAL_WORLD = "real_world"
STAKES_IN_GAME = "in_game"
STAKES_AMBIGUOUS = "ambiguous"
STAKES_NONE = "none"

IDENTITY_SELF = "self"
IDENTITY_CHARACTER = "character"
IDENTITY_QUOTED = "quoted"
IDENTITY_HYPOTHETICAL = "hypothetical"
IDENTITY_UNKNOWN = "unknown"

PROP_REAL_BODY = "real_body"
PROP_CHARACTER_BODY = "character_body"
PROP_UNCLEAR = "unclear"


@dataclass
class StakesResult:
    """Multi-axis classification of an utterance.

    `requires_clarification` is the load-bearing flag — True means the
    caller should ASK the user rather than guess. It's set when the
    stakes axis lands on AMBIGUOUS *and* safety terms are present.
    """
    stakes: str = STAKES_NONE
    identity: str = IDENTITY_UNKNOWN
    proprioceptive: str = PROP_UNCLEAR
    requires_clarification: bool = False
    confidence: float = 1.0
    matched_safety: list[str] = field(default_factory=list)
    matched_game: list[str] = field(default_factory=list)
    matched_body_parts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "stakes": self.stakes,
            "identity": self.identity,
            "proprioceptive": self.proprioceptive,
            "requires_clarification": self.requires_clarification,
            "confidence": self.confidence,
            "matched_safety": self.matched_safety,
            "matched_game": self.matched_game,
            "matched_body_parts": self.matched_body_parts,
            "notes": self.notes,
        }


# ── Core matcher ────────────────────────────────────────────────────


def _find_terms(text_lower: str, vocab: frozenset[str]) -> list[str]:
    """Return matched terms in vocabulary order (no duplicates).

    Multi-word terms are matched literally; single-word terms are
    matched as whole words via boundary anchors. This avoids 'leg' in
    'legible' and 'hurt' in 'unhurt' from triggering.
    """
    matched: list[str] = []
    for term in vocab:
        if " " in term or "'" in term:
            # Multi-word / contraction: literal contains check
            if term in text_lower:
                matched.append(term)
        else:
            # Single word: word-boundary match
            if re.search(rf"\b{re.escape(term)}\b", text_lower):
                matched.append(term)
    return matched


def _detect_identity(
    text_lower: str,
    role_play_active: bool,
    in_character_marker: bool,
    out_of_character_marker: bool,
) -> str:
    """Pick an identity axis value based on markers + role-play context."""
    if out_of_character_marker:
        return IDENTITY_SELF
    if in_character_marker:
        return IDENTITY_CHARACTER
    for p in _QUOTED_OR_HYPOTHETICAL_PREFIXES:
        if p in text_lower:
            if any(h in p for h in ("if i", "imagine", "suppose", "hypothet")):
                return IDENTITY_HYPOTHETICAL
            return IDENTITY_QUOTED
    if role_play_active:
        # Without an explicit OOC marker, ambient role-play context
        # leaves the "I" ambiguous. Pick UNKNOWN — the proprioceptive
        # axis carries more weight for clarification decisions.
        return IDENTITY_UNKNOWN
    return IDENTITY_SELF


def _detect_proprioceptive(
    matched_safety: list[str],
    matched_game: list[str],
    matched_body_parts: list[str],
    role_play_active: bool,
    in_character_marker: bool,
    out_of_character_marker: bool,
) -> str:
    """Decide which body frame is being talked about."""
    if out_of_character_marker:
        return PROP_REAL_BODY
    if in_character_marker:
        return PROP_CHARACTER_BODY
    if matched_game and not matched_safety:
        return PROP_CHARACTER_BODY
    if matched_safety and not matched_game and not role_play_active:
        return PROP_REAL_BODY
    if matched_body_parts and role_play_active:
        return PROP_UNCLEAR
    if matched_safety and not role_play_active:
        return PROP_REAL_BODY
    return PROP_UNCLEAR


def _detect_stakes(
    matched_safety: list[str],
    matched_game: list[str],
    matched_body_parts: list[str],
    role_play_active: bool,
    in_character_marker: bool,
    out_of_character_marker: bool,
) -> tuple[str, bool, float]:
    """Resolve the stakes axis. Returns (stakes, requires_clarification, confidence)."""
    # Explicit markers dominate.
    if out_of_character_marker and matched_safety:
        return STAKES_REAL_WORLD, False, 0.95
    if in_character_marker:
        return STAKES_IN_GAME, False, 0.95

    has_safety = bool(matched_safety)
    has_game = bool(matched_game)

    # Both signals present → ambiguous, definitely ASK.
    if has_safety and has_game:
        return STAKES_AMBIGUOUS, True, 0.6

    # Game-only signal, no safety → in-game.
    if has_game and not has_safety:
        return STAKES_IN_GAME, False, 0.85

    # Safety-only signal, no game, no role-play → real-world.
    if has_safety and not role_play_active:
        return STAKES_REAL_WORLD, False, 0.9

    # Safety + role-play active + no explicit OOC marker → AMBIGUOUS
    # (the load-bearing case). Even safety alone in an RP context
    # warrants the ASK, because "I broke my leg" without "ooc:" mid-D&D
    # is exactly the high-stakes ambiguity 6ho was filed to prevent.
    if has_safety and role_play_active:
        return STAKES_AMBIGUOUS, True, 0.7

    # No strong signal either way.
    return STAKES_NONE, False, 1.0


def classify_stakes(
    text: str,
    *,
    role_play_active: bool = False,
    persona_active: Optional[str] = None,
) -> StakesResult:
    """Classify an utterance along the three 6ho axes.

    Args:
        text: the user's utterance.
        role_play_active: True if a fictional/RP context is on (e.g. the
            affect-system's ``ctx_dnd_session`` overlay is active).
        persona_active: optional name of the active persona (for logs).

    Returns:
        StakesResult — see the dataclass. The load-bearing flag is
        ``requires_clarification``: True means caller should ASK before
        proceeding.
    """
    if not text or not text.strip():
        return StakesResult(stakes=STAKES_NONE, identity=IDENTITY_UNKNOWN,
                            proprioceptive=PROP_UNCLEAR)

    # Normalize hyphens to spaces so multi-word vocab matches "3rd-level
    # spells" the same as "3rd level spells". The marker check is done
    # against the ORIGINAL lower-case form because IC/OOC markers are
    # written with hyphens ("in-character", "out-of-character") and the
    # normalized form would collide with the bracket/colon variants.
    tl_orig = text.lower()
    tl = re.sub(r"-+", " ", tl_orig)
    in_character_marker = any(m in tl_orig for m in _IN_CHARACTER_MARKERS)
    out_of_character_marker = any(m in tl_orig for m in _OUT_OF_CHARACTER_MARKERS)

    matched_safety = _find_terms(tl, _SAFETY_TERMS)
    matched_game = _find_terms(tl, _GAME_MECHANIC_TERMS)
    matched_body_parts = _find_terms(tl, _BODY_PART_TERMS)

    identity = _detect_identity(
        tl, role_play_active, in_character_marker, out_of_character_marker,
    )
    proprioceptive = _detect_proprioceptive(
        matched_safety, matched_game, matched_body_parts,
        role_play_active, in_character_marker, out_of_character_marker,
    )
    stakes, requires_clarif, confidence = _detect_stakes(
        matched_safety, matched_game, matched_body_parts,
        role_play_active, in_character_marker, out_of_character_marker,
    )

    notes: list[str] = []
    if persona_active:
        notes.append(f"persona={persona_active}")
    if role_play_active:
        notes.append("role_play_active=True")
    if in_character_marker:
        notes.append("ic_marker")
    if out_of_character_marker:
        notes.append("ooc_marker")

    return StakesResult(
        stakes=stakes,
        identity=identity,
        proprioceptive=proprioceptive,
        requires_clarification=requires_clarif,
        confidence=confidence,
        matched_safety=matched_safety,
        matched_game=matched_game,
        matched_body_parts=matched_body_parts,
        notes=notes,
    )


# ── Affect-runtime integration helper ───────────────────────────────


def is_role_play_active() -> bool:
    """Convenience: ask the affect runtime whether a fictional/RP
    context is currently active.

    Looks for affect-system context overlays whose key suggests
    role-play (dnd_session, story_writing, role_play, etc.). Returns
    False on any failure — the classifier still works without affect
    context, just with `role_play_active=False`.
    """
    try:
        from gaia_core.cognition.affect_runtime import current_affect_snapshot
        snap = current_affect_snapshot()
        ctx = snap.get("active_context")
        if not ctx:
            return False
        ctx_lower = str(ctx).lower()
        for hint in ("dnd_session", "role_play", "roleplay",
                     "story_writing", "fiction", "rp_session"):
            if hint in ctx_lower:
                return True
        return False
    except Exception:
        return False

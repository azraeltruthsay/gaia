"""AffectKG — affect/persona-trait state on top of the World Model.

A thin facade over `KnowledgeGraph` that uses the predicate vocabulary
in `affect_predicates.py`. Provides:

  - record_feeling / record_trait / record_drive / record_curious_about
    / record_tired_of — write GAIA's own affect (subject=SELF) in the
    active world (actuality by default).
  - record_belief_about — theory-of-mind triples for a modeled person,
    scoped to a `belief_of_<person>` world.
  - activate_context / deactivate_context — manage ephemeral overlay
    worlds that carry per-situation trait deltas.
  - flatten_current_affect — read the current affect vector, walking
    the active overlay's inheritance chain and applying placeholder
    decay (until Stage 7 lw4 ships the proper kernel).

**Phase 1** of GAIA_Project-usv. Inference modulation (prompt_builder,
agent_core routing) lives in Phase 2 and is intentionally NOT in this
module — keep this layer pure data so it stays testable in isolation.

See: knowledge/blueprints/affect_model.md
"""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from gaia_common.utils.knowledge_graph import KnowledgeGraph
from gaia_common.utils.affect_predicates import (
    SELF,
    OBJ_AFFECT_STATE,
    OBJ_PERSONA_STATE,
    OBJ_DRIVE_STATE,
    OBJ_ATTENTION_STATE,
    AFFECT_PREFIX_TABLE,
    PREFIX_FEELS,
    PREFIX_TRAIT,
    PREFIX_DRIVE,
    PREFIX_CURIOUS_ABOUT,
    PREFIX_TIRED_OF,
    PREFIX_BELIEVES_ABOUT,
    MODALITY_CONTEXT,
    pred_feels,
    pred_trait,
    pred_drive,
    pred_curious_about,
    pred_tired_of,
    pred_believes_about,
    context_world_name,
    belief_of_world_name,
)
from gaia_common.utils import fact_types
from gaia_common.utils.recency import decayed_relevance


# ── Predicate-prefix → fact_type mapping ────────────────────────────
# Stage 7 (lw4) routes affect decay through the unified recency kernel.
# Each affect predicate prefix maps to a dedicated fact_type so the
# half-lives in fact_types.HALFLIFE drive decay rather than the prefix
# table in affect_predicates. The numeric half-lives are identical, so
# the swap is mechanical — tests that pinned the placeholder math still
# pass.

_PREFIX_TO_FACT_TYPE: list[tuple[str, str]] = [
    (PREFIX_FEELS,           fact_types.AFFECT_FEELS),
    (PREFIX_TRAIT,           fact_types.AFFECT_TRAIT),
    (PREFIX_DRIVE,           fact_types.AFFECT_DRIVE),
    (PREFIX_CURIOUS_ABOUT,   fact_types.AFFECT_CURIOUS_ABOUT),
    (PREFIX_TIRED_OF,        fact_types.AFFECT_TIRED_OF),
    (PREFIX_BELIEVES_ABOUT,  fact_types.AFFECT_BELIEVES_ABOUT),
]


def _fact_type_for_predicate(predicate: str) -> Optional[str]:
    """Return the affect-class fact_type for a predicate, or None."""
    for prefix, ft in _PREFIX_TO_FACT_TYPE:
        if predicate.startswith(prefix):
            return ft
    return None


def _decayed_confidence(
    stored_conf: float,
    valid_from: Optional[str],
    predicate: str,
    *,
    now: Optional[datetime] = None,
) -> float:
    """Apply half-life decay to a stored confidence value (Stage 7).

    Delegates to the unified `recency.decayed_relevance` kernel after
    mapping the predicate prefix to its fact_type. Predicates without an
    affect prefix (defensive — shouldn't happen via this module) return
    the stored confidence unmodified.
    """
    ft = _fact_type_for_predicate(predicate)
    if ft is None:
        # Unknown predicate prefix → don't decay (preserve prior placeholder
        # behavior, which returned conf for unknown prefixes).
        return stored_conf
    return decayed_relevance(stored_conf, valid_from, ft, now=now)


def _clamp01(x: float) -> float:
    try:
        if math.isnan(x):
            return 0.0
    except TypeError:
        pass
    try:
        val = float(x)
        if math.isnan(val):
            return 0.0
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.0


# ── Facade ──────────────────────────────────────────────────────────

class AffectKG:
    """Affect state on the World Model. Wraps a KnowledgeGraph instance."""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    # ─── Writes (GAIA's own state) ──────────────────────────────────

    def record_feeling(self, emotion: str, intensity: float, *,
                       world: str = "actuality",
                       source: Optional[str] = None) -> str:
        return self._update_affect(
            subject=SELF, predicate=pred_feels(emotion),
            obj=OBJ_AFFECT_STATE, confidence=intensity,
            world=world, source=source,
        )

    def record_trait(self, trait: str, value: float, *,
                     world: str = "actuality",
                     source: Optional[str] = None) -> str:
        return self._update_affect(
            subject=SELF, predicate=pred_trait(trait),
            obj=OBJ_PERSONA_STATE, confidence=value,
            world=world, source=source,
        )

    def record_drive(self, drive: str, level: float, *,
                     world: str = "actuality",
                     source: Optional[str] = None) -> str:
        return self._update_affect(
            subject=SELF, predicate=pred_drive(drive),
            obj=OBJ_DRIVE_STATE, confidence=level,
            world=world, source=source,
        )

    def record_curious_about(self, topic: str, weight: float, *,
                             world: str = "actuality",
                             source: Optional[str] = None) -> str:
        return self._update_affect(
            subject=SELF, predicate=pred_curious_about(topic),
            obj=OBJ_ATTENTION_STATE, confidence=weight,
            world=world, source=source,
        )

    def record_tired_of(self, topic: str, aversion: float, *,
                        world: str = "actuality",
                        source: Optional[str] = None) -> str:
        return self._update_affect(
            subject=SELF, predicate=pred_tired_of(topic),
            obj=OBJ_ATTENTION_STATE, confidence=aversion,
            world=world, source=source,
        )

    def _update_affect(
        self, *,
        subject: str, predicate: str, obj: str,
        confidence: float, world: str, source: Optional[str],
    ) -> str:
        """Close any open triple with the same (subject, predicate, object,
        world) by setting valid_to=NOW, then insert a fresh open triple.

        Resolves world name → world ID first so triples land under the
        same identifier the inheritance walker uses (existing KG
        convention: callers pass world IDs to add_triple).

        This is the canonical update path for affect state. It produces
        a historical record (old triple closed, new triple open) that
        decay/audit logic can walk over time, while avoiding the KG's
        dedup-suppression of unchanged-tuple inserts.
        """
        confidence = _clamp01(confidence)
        sub_id = self.kg._entity_id(subject)
        obj_id = self.kg._entity_id(obj)
        pred = predicate.lower().replace(" ", "_")
        now_iso = datetime.now(timezone.utc).isoformat()

        # Resolve world name → world ID. Triples are stored under the ID;
        # the inheritance walker resolves names to IDs at query time, so
        # writing under a name leaves the triple orphaned to inheritance.
        meta = self.kg.get_world(world)
        world_id = meta["id"] if meta else world

        # Close any existing open triple for this affect slot in this world.
        conn = sqlite3.connect(self.kg.db_path)
        try:
            conn.execute(
                "UPDATE triples SET valid_to = ? "
                "WHERE subject = ? AND predicate = ? AND object = ? "
                "AND world = ? AND valid_to IS NULL",
                (now_iso, sub_id, pred, obj_id, world_id),
            )
            conn.commit()
        finally:
            conn.close()

        # Insert fresh open triple with the new confidence.
        # Stage 7 (lw4): pass the predicate's fact_type so retrieval
        # decay routes through the unified recency kernel.
        return self.kg.add_triple(
            subject=subject, predicate=predicate, obj=obj,
            valid_from=now_iso,
            confidence=confidence,
            source=source or "affect_kg",
            world=world_id,
            fact_type=_fact_type_for_predicate(pred),
        )

    # ─── Writes (theory of mind) ────────────────────────────────────

    def record_belief_about(
        self,
        person: str,
        attribute: str,
        value: str,
        confidence: float = 0.7,
        *,
        source: Optional[str] = None,
        create_world_if_missing: bool = True,
    ) -> str:
        """Record GAIA's belief about a person's state.

        Lives in `belief_of_<person>` world (created on first use,
        modality=belief_of, parent=actuality, durable). Each attribute
        gets its own predicate (`believes_about_<attribute>`), so a
        person can simultaneously be "focused" on one topic and
        "frustrated" with another.
        """
        world_name = belief_of_world_name(person)
        if create_world_if_missing and self.kg.get_world(world_name) is None:
            self.kg.create_world(
                name=world_name,
                modality="belief_of",
                parent="actuality",
                description=f"GAIA's theory-of-mind about {person}.",
                lifecycle="durable",
            )
        return self._update_affect(
            subject=person,
            predicate=pred_believes_about(attribute),
            obj=value,  # value IS the object for beliefs (e.g. "focused")
            confidence=confidence,
            world=world_name,
            source=source,
        )

    # ─── Context overlay management ─────────────────────────────────

    def activate_context(
        self,
        context_key: str,
        *,
        ttl_seconds: int = 3600,
        session_id: Optional[str] = None,
        description: Optional[str] = None,
    ) -> str:
        """Activate an ephemeral context overlay world.

        Idempotent: if the world already exists, this is a no-op (it
        returns the world name). To extend TTL, deactivate + reactivate.
        Returns the world name (e.g. "ctx_dnd_session").
        """
        world_name = context_world_name(context_key)
        if self.kg.get_world(world_name) is None:
            self.kg.create_world(
                name=world_name,
                modality=MODALITY_CONTEXT,
                parent="actuality",
                description=description or f"Context overlay: {context_key}",
                lifecycle="ephemeral",
                session_id=session_id,
                ttl_seconds=ttl_seconds,
            )
        return world_name

    def deactivate_context(self, context_key: str) -> bool:
        """Deactivate (delete) a context overlay world.

        Uses KnowledgeGraph.delete_world(force=True) — the existing
        World Model API already handles triple + edge cleanup. Returns
        True if the world existed and was removed.
        """
        world_name = context_world_name(context_key)
        if self.kg.get_world(world_name) is None:
            return False
        return self.kg.delete_world(world_name, force=True)

    # ─── Reads ──────────────────────────────────────────────────────

    def flatten_current_affect(
        self,
        *,
        active_context: Optional[str] = None,
        now: Optional[datetime] = None,
        include_closed: bool = False,
    ) -> dict:
        """Compute GAIA's current affect vector.

        Walks the inheritance chain from the active context (or actuality
        if none active) for SELF's triples, classifies by predicate
        prefix, applies placeholder decay, and returns:

          {
            "traits":  {trait_name: effective_value, ...},
            "feels":   {emotion:    effective_intensity, ...},
            "drives":  {drive_name: effective_level, ...},
            "curious_about": {topic: effective_weight, ...},
            "tired_of":      {topic: effective_aversion, ...},
            "active_context": "ctx_dnd_session" | None,
            "as_of":   ISO timestamp,
          }

        By default only open triples (valid_to IS NULL) participate —
        the historical record produced by _update_affect stays out of
        the active vector. Pass include_closed=True for audit walks.
        """
        if now is None:
            now = datetime.now(timezone.utc)
        world = active_context or "actuality"
        if active_context and self.kg.get_world(active_context) is None:
            world = "actuality"
            active_context = None

        facts = self.kg.query_entity_inherited(
            SELF, world=world, direction="outgoing",
        )
        result = {
            "traits": {},
            "feels": {},
            "drives": {},
            "curious_about": {},
            "tired_of": {},
            "active_context": active_context,
            "as_of": now.isoformat(),
        }

        for f in facts:
            if not include_closed and f.get("valid_to"):
                continue
            pred = f.get("predicate", "")
            bucket = None
            suffix = None
            for prefix, bucket_name, _expected_sentinel in AFFECT_PREFIX_TABLE:
                if pred.startswith(prefix):
                    bucket = bucket_name
                    suffix = pred[len(prefix):]
                    break
            if not bucket:
                continue
            decayed = _decayed_confidence(
                stored_conf=float(f.get("confidence", 1.0)),
                valid_from=f.get("valid_from"),
                predicate=pred,
                now=now,
            )
            prior = result[bucket].get(suffix, 0.0)
            if decayed > prior:
                result[bucket][suffix] = decayed
        return result

    def belief_about(self, person: str) -> dict:
        """Read GAIA's current theory-of-mind for a person.

        Returns a dict mapping attribute → {value, confidence}.
        Only open belief triples participate.
        """
        world_name = belief_of_world_name(person)
        meta = self.kg.get_world(world_name)
        if meta is None:
            return {}
        # Resolve name → ID since triples are stored under the world ID.
        facts = self.kg.query_entity(
            person, world=meta["id"], direction="outgoing",
        )
        out: dict[str, dict] = {}
        for f in facts:
            pred = f.get("predicate", "")
            if not pred.startswith(PREFIX_BELIEVES_ABOUT):
                continue
            if f.get("valid_to"):
                continue
            attribute = pred[len(PREFIX_BELIEVES_ABOUT):]
            value = f.get("object", "")
            decayed = _decayed_confidence(
                stored_conf=float(f.get("confidence", 1.0)),
                valid_from=f.get("valid_from"),
                predicate=pred,
            )
            prior = out.get(attribute)
            if not prior or decayed > prior["confidence"]:
                out[attribute] = {"value": value, "confidence": decayed}
        return out

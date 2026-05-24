# Affect Model — Mood / Drive / Persona Traits as a World Model Layer

> Implements **GAIA_Project-usv**. Builds on `world_model.md` (Stages 0-6 shipped 2026-05-22). Targets Stage 7 (lw4) once recency-decay scoring lands.

## Why this exists

Static persona JSON gives GAIA fixed numbers (`curiosity: 0.9, caution: 0.4`) that don't decay, don't shift by context, and don't track what she thinks anyone else is feeling. The World Model already has every primitive an affect system needs: temporal validity, ephemeral context overlays, counterfactual modality, belief_of modality, modality firewall, consistency audit. So affect is **not a new engine** — it's a vocabulary + a few thin query helpers on top of the KG.

A Creatures-style modular biochemistry is *not* the goal. The goal is a small, persistent state vector that evolves with events, biases inference at turn-build time, and supports theory-of-mind about Azrael and other interlocutors.

## Conceptual model

GAIA's runtime affect is the **flattened projection** of three layered sources, walked through the World Model's inheritance:

  base_persona_traits  (durable triples in `actuality`)
    ↓ inherits-from
  active_context_overlays  (ephemeral worlds with TTL, e.g. `ctx_dnd_session`)
    ↓ overlays
  recent_affect_events  (time-bounded triples with `valid_from` / `valid_to`)
    ↓ projected through
  decay(now - valid_from)  (Stage 7 recency math)
    ↓ produces
  current_affect_vector  (read at turn-build time)

Theory-of-mind state for *other* people lives in `belief_of` worlds parented to actuality — same machinery, modality firewall keeps GAIA from confusing "I think Azrael is frustrated" with "I am frustrated."

## Predicate vocabulary (initial)

Stored as constants in `gaia_common.affect.predicates`. All affect triples use **`self`** as the canonical subject for GAIA's own state; user/agent names for theory-of-mind.

| Predicate | Object type | Confidence semantics | Decays? |
|---|---|---|---|
| `trait` | `<name>` (e.g. `"curiosity"`, `"warmth"`) with numeric in `confidence` (0..1) | Numeric trait level, treated as the value | Slowly (weeks) — base personality drift |
| `feels` | `<emotion>` (e.g. `"irritation"`, `"curiosity"`, `"calm"`) | Strength 0..1 | Fast (minutes-hours) |
| `drive_level` | `<drive>` (e.g. `"hunger_for_novelty"`, `"social_engagement"`) | Drive intensity 0..1 | Medium (hours) |
| `curious_about` | `<topic>` (e.g. `"consistency_detector"`, `"audio_pipeline"`) | Attention weight 0..1 | Medium (hours-day) |
| `tired_of` | `<topic>` | Aversion strength 0..1 | Medium |
| `believes_about` | `<person>:<attribute>` (e.g. `"azrael:current_mood"` with object `"focused"`) | Belief confidence | Per-source decay |

**Notable choices:**

- Trait *values* (the 0..1 number) are stored in the triple's `confidence` field rather than as part of the object string. This means `query_entity_inherited(self)` already returns the right number with the World Model's existing precedence rules — no schema change.
- Single canonical subject `self` keeps queries simple. To distinguish "GAIA right now" from "GAIA in DnD mode," use the world axis: same subject, different inheriting world.
- Predicates are lowercase snake_case strings; objects are plain strings. No JSON-in-object — keeps the KG normalizable.

## Context overlays

Ephemeral worlds with naming convention `ctx_<context_key>` and lifecycle `ephemeral`:

  ctx_dnd_session         parent=actuality  modality=context  ttl=session
  ctx_coding_debug        parent=actuality  modality=context  ttl=session
  ctx_azrael_present      parent=actuality  modality=context  ttl=long  (auto-renews while azrael interacts)
  ctx_morning_routine     parent=actuality  modality=context  ttl=2h

Each context world holds **trait deltas** — e.g. `(self, trait, "playfulness", confidence=+0.3)`. When `flatten_current_affect(active_context_worlds=[...])` queries, the active contexts' deltas combine multiplicatively or additively with the base trait values from actuality. (Mechanism: query through `query_entity_inherited` with the deepest active overlay as the world, then post-process to blend.)

**Activation** is driven from agent_core's turn intake — recognizing a DnD command activates `ctx_dnd_session`. Activation is itself an event the consistency detector can audit ("why is GAIA suddenly more playful — what activated this?").

**Modality `context`** is a new modality value, added to the existing set (`actuality`, `fiction`, `counterfactual`, `hypothetical`, `projection`, `belief_of`). Behaves like `actuality` for inheritance but is conventionally ephemeral.

## Counterfactual mood simulation

When GAIA needs to project a decision's affective consequence ("what would I feel if I refuse this request?"), spawn a hypothetical world:

  cf_refuse_dnd_request_2026_05_23  parent=current_active_world  modality=hypothetical  ttl=10min

Run a small Prime call inside that world (the cognitive scope sees actuality + active context + the counterfactual deltas), capture the projected affect, then either:

  - commit the projection back to actuality if the path is taken, OR
  - discard the world (the modality firewall guarantees no leakage).

This is the cleanest case where the World Model's machinery directly enables a capability we didn't have before.

## Theory of mind

For each modeled person, GAIA maintains a `belief_of_<name>` world:

  belief_of_azrael   parent=actuality  modality=belief_of
    triples: (azrael, feels, "focused", conf=0.6, valid_from=...)
             (azrael, curious_about, "audio_pipeline", conf=0.7, ...)

Updates flow from:

  - explicit user statements in conversation history,
  - samvega artifacts that note someone's tone or frustration,
  - cross-tier audit findings about how a previous response landed.

The modality firewall means a query against `actuality` never accidentally returns these — GAIA doesn't conflate her models of others with her own state. To check "what does GAIA think Azrael is feeling?" the caller explicitly queries `belief_of_azrael`.

## Decay (Stage 7 dependency)

Affect triples carry `valid_from`. Recency-decay scoring (Stage 7, lw4) computes an effective confidence as:

  effective_conf(triple) = stored_conf × decay_kernel(now - valid_from, predicate_type)

Different predicates have different half-lives — `feels` decays in minutes, `trait` in weeks, `believes_about` per-source. The kernel and half-lives are tuned via constants in `gaia_common.affect.predicates`.

Until Stage 7 lands, `flatten_current_affect` will use a placeholder linear decay so the affect vector still evolves between turns; tests will assert decay behavior so swapping in the Stage-7 kernel is mechanical.

## Inference modulation (Phase 2)

Once the data layer is solid, `prompt_builder` reads the flattened affect at turn-build time and:

  1. Prepends a small "current state" header to the system prompt:
     `<state>curiosity=0.92 warmth=0.7 fatigue=0.3 focus=0.85</state>`
  2. Modulates inference parameters:
     - high `caution` + high `logic_priority` → escalate to Prime
     - high `feels=irritation` → cap temperature, force `style=measured`
     - high `curious_about=<current_topic>` → expand `max_tokens`
  3. Routes through the appropriate persona JSON's base configuration (penpal, default, council, etc.).

agent_core's existing routing logic gets a small read-through to query the affect vector before tier selection.

## Out of scope

  - Hebbian learning rules / Creatures-style local update
  - Biochemistry pool simulation (digestion, hormones as separate entities)
  - Reverse-engineering Creatures `.exp` genome encoding
  - Replacing existing PersonaManager — affect *augments* persona, doesn't replace it

## Phasing

  **Phase 1** (this session) — Data layer:
    - This blueprint (anchors the rest)
    - Predicate constants module
    - `AffectKG` facade on `KnowledgeGraph`: record_feeling(), record_trait(), record_belief_about(), activate_context(), deactivate_context(), flatten_current_affect()
    - Tests covering predicates, context activation, theory-of-mind firewall, decay-placeholder

  **Phase 2** — Inference modulation:
    - prompt_builder integration (state header + parameter modulation)
    - agent_core routing read-through
    - Context detection hooks at turn intake

  **Phase 3** — Lived-experience demo:
    - Session that exercises a context shift, an affect event, and a counterfactual
    - Stage 7 (lw4) decay integration when it lands
    - Refine the predicate vocabulary based on what actually shows up in real sessions

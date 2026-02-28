# Saṃvega — Semantic Discernment Artifacts

## Etymology

From the Pali: being moved with force by recognition of error. In GAIA's cognition,
saṃvega is the mechanism that gives negative outcomes — wrong answers, misaligned
responses, user corrections — a richer, more durable cognitive signature than
ordinary reflections.

## Design Philosophy

Saṃvega is **discernment, not punishment**. Artifacts encode not just *what* went
wrong but *why it missed the mark*, and the highest-weight lessons feed into the
training pipeline (Tier 5 knowledge).

**Anti-avoidance principle (design constraint, not optional):** Corrected
understanding must describe what a better approach *would look like*, never
"don't talk about this topic." The goal is refinement of capability, not
narrowing of scope.

## Artifact Schema

```
SamvegaArtifact:
  artifact_type: "samvega"           # Always "samvega"
  timestamp: ISO-8601                # UTC creation time
  session_id: str                    # Originating session
  packet_id: str                     # Originating CognitionPacket
  trigger: str                       # user_correction | confidence_mismatch | pattern_detection
  original_output_summary: str       # First 500 chars of the output that triggered this
  what_went_wrong: str               # LLM-generated single sentence
  root_cause: str                    # Underlying reason (not just "wrong answer")
  values_misaligned: [str]           # 1-3 values from the taxonomy
  corrected_understanding: str       # What a better response would look like
  weight: float                      # 0.0-1.0, computed from trigger + multipliers
  promoted_to_tier5: bool            # True if weight >= threshold (0.7)
  reviewed: bool                     # Set by sleep introspection
  reviewed_at: ISO-8601 | null       # When reviewed
```

## Trigger Conditions

### Trigger A — User Correction
- **When:** `plan.intent == "correction"` during cognition pipeline
- **Base weight:** 0.6
- **Context:** User's corrective message is passed as `user_correction_text`

### Trigger B — Confidence Mismatch
- **When:** Pre-reflection confidence minus post-reflection confidence >= threshold (0.3)
- **Base weight:** The gap itself (e.g., 0.8 → 0.4 = base 0.4)
- **Context:** Detected automatically from packet confidence values

### Trigger C — Observer Pattern Detection
- **When:** Post-stream observer returns `CAUTION` or `BLOCK` level
- **Base weight:** 0.4
- **Context:** Observer severity is passed as a weight multiplier

## Weight Formula

```
base = trigger-dependent (see above)
if observer_severity == BLOCK:  base *= 1.5
if observer_severity == CAUTION: base *= 1.2
if repeated_domain:             base *= 1.3
weight = clamp(base, 0.0, 1.0)
```

## Tier 5 Promotion

Artifacts with `weight >= 0.7` are flagged `promoted_to_tier5 = True`. These are
candidates for transformation into QLoRA training pairs (deferred — future work).

## Sleep Introspection

During sleep cycles, the `samvega_introspection` task (priority 2) runs:

1. Loads all unreviewed artifacts
2. Groups by `root_cause` (string matching)
3. Boosts weight × 1.3 for clusters with 2+ artifacts (repeated blind spots)
4. Marks all as reviewed with timestamp
5. Flags artifacts above threshold as promoted to tier 5
6. Logs summary

## Values Taxonomy

The LLM analysis selects 1-3 values that were violated:

- **accuracy** — factual correctness
- **relevance** — topical appropriateness
- **user_understanding** — grasping what the user actually meant
- **epistemic_humility** — appropriate uncertainty signaling
- **safety** — harm avoidance
- **creativity** — originality and engagement
- **contextual_sensitivity** — reading the room / tone

## Storage

- Active artifacts: `/knowledge/samvega/samvega_*.json`
- Archived artifacts: `/knowledge/samvega/archive/`
- Config: `gaia_constants.json` → `SAMVEGA` block

## Mindscape Location

Mirror Spire (deferred — sleep prompt integration is future work).

## What Is Deferred

- Discord reaction handler (`on_reaction_add` for thumbs down)
- Full Tier 5 training pipeline (artifacts → QLoRA pairs)
- Embedding-based pattern consolidation (sleep task uses string matching)
- Mindscape prompt integration (Mirror Spire framing)

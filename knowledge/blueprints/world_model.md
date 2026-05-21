# GAIA World Model Blueprint

## Role and overview

The World Model is GAIA's mechanism for **scoping assertions to a context** so that fiction, hypotheticals, counterfactuals, and other agents' beliefs can be reasoned about without contaminating consensus reality.

It is built as a named-graph extension on top of the existing Knowledge Graph (`gaia-common/gaia_common/utils/knowledge_graph.py`, persisted at `/shared/knowledge_graph/gaia_kg.sqlite3`). Every triple gets a `world` dimension; queries default to `actuality` (consensus reality) and never see other-world triples unless explicitly requested.

The World Model is **observe-first**: it tracks scoping, catches contradictions, and surfaces frame shifts as Saṃvega artifacts. It does **not** attempt to make GAIA understand new worlds the way a child does — induction is still LLM-bounded. What this architecture gives is a place to write down, scope, check, and revise the model's fallible understanding.

This blueprint is the durable design reference. The narrative origin and the staged rollout decisions live in the gitignored dev journal at `knowledge/Dev_Notebook/2026-05-21_world_model_design.md`.

## Why this exists

Before the World Model:

- The KG was a flat namespace. "Hermione cast Lumos" and "Azrael lives in PST" were both triples in the same store with no distinction between fiction and consensus reality.
- Auto-extraction was noisy enough that the KG had 25 entities, of which 20 were token-fragment garbage (`'sup'`, `'bow'`, `'new'`) from "Super Bowl" mentions sliced by AAAK's compression codes.
- Web grounding searches returned, e.g., `Amin Howeidi` (an Egyptian general) as a top hit for the fragment "AGI. How", and the model dutifully wove him into a philosophical answer about AGI.
- The deliberation pipeline had no structural way to catch a fresh claim that contradicted something the system had previously established.

Each of those failures is a symptom of the same root: GAIA had no scoping layer and no structural consistency check.

## Data model

### The quad

Each assertion is a quad rather than a triple:

```
(world, subject, predicate, object, valid_from, valid_to, confidence, source)
```

This is the RDF named-graph pattern, tested at scale in Virtuoso / Apache Jena / GraphDB. It's the standard way to scope an assertion to a context without committing globally.

### Storage layout

The `triples` table holds quads. After Stage 1 the schema is:

```sql
CREATE TABLE triples (
    id          TEXT PRIMARY KEY,
    subject     TEXT NOT NULL,
    predicate   TEXT NOT NULL,
    object      TEXT NOT NULL,
    valid_from  TEXT,
    valid_to    TEXT,
    confidence  REAL DEFAULT 1.0,
    source      TEXT,
    extracted_at TEXT DEFAULT CURRENT_TIMESTAMP,
    world       TEXT NOT NULL DEFAULT 'actuality',
    FOREIGN KEY (subject) REFERENCES entities(id),
    FOREIGN KEY (object)  REFERENCES entities(id)
);

CREATE INDEX idx_triples_world          ON triples(world);
CREATE INDEX idx_triples_world_subject  ON triples(world, subject);
```

Stage 3 adds:

```sql
CREATE TABLE worlds (
    id          TEXT PRIMARY KEY,        -- opaque atom: w_3f9a
    name        TEXT NOT NULL,
    modality    TEXT NOT NULL,           -- actuality | fiction | counterfactual | hypothetical | projection | belief_of
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE world_edges (
    parent_id   TEXT NOT NULL,
    child_id    TEXT NOT NULL,
    edge_type   TEXT NOT NULL CHECK (edge_type IN ('overlays', 'refines', 'branches-from')),
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (parent_id, child_id, edge_type),
    FOREIGN KEY (parent_id) REFERENCES worlds(id),
    FOREIGN KEY (child_id)  REFERENCES worlds(id)
);
```

### Identity vs address

The `world` field on every quad is an **opaque atom** — `w_3f9a`. It never changes for the life of the world. Survives renames. Survives merges. Identity is the one thing that stays stable so the dangerous operations can stay local and reversible.

The rich multi-part view (`actuality › fiction › potterverse(books) › hogwarts-1990s`) lives in the `worlds` + `world_edges` registry as a small DAG. The legible path is a **rendered traversal** of that graph — never the stored key.

```
                actuality (w_root)
                  │  overlays
        ┌─────────┼──────────┐
        │         │          │
   fiction   counterfactual  hypothetical
        │
        │ branches-from
        │
   potterverse(books) (w_3f9a)
        │
        │ refines
        │
   hogwarts-1990s
```

### Edge types

Two child relationships behave **opposite** under inheritance:

| Edge type        | Semantics                                                | Inheritance behavior |
|------------------|----------------------------------------------------------|----------------------|
| `overlays`       | Generic parent-child where child can add or override     | Inherit, add, override |
| `refines`        | Child narrows scope of parent (same physics, subset)     | Inherit; child cannot contradict parent |
| `branches-from`  | Child is a variant universe of parent                    | Inherit baseline; child may shadow parent |

Storing this as edge type (not as a path segment) means inheritance semantics travel with the structure, not with the key.

### Modality

Each world carries a typed modality that governs query leakage:

| Modality           | Meaning                                          | Leakage into actuality queries |
|--------------------|--------------------------------------------------|----------------------------------|
| `actuality`        | Consensus reality                                | (this IS the default)           |
| `fiction`          | Explicitly imagined                              | Never                            |
| `counterfactual`   | What-if branch from actuality                    | Never                            |
| `hypothetical`     | Transient working set                            | Never                            |
| `projection`       | Modeled future state                             | Never                            |
| `belief_of:<agent>` | Another agent's mental model                    | Never                            |

Stage 4 enforces this. Without it, "Hermione cast Lumos" can leak into spellcraft queries about real chemistry.

## Query semantics

### Default behavior

```python
kg.query_entity("Hogwarts")                # → only actuality facts
kg.query_entity("Hogwarts", world="potterverse")  # → only potterverse facts
kg.query_entity("Hogwarts", world=None)    # → all worlds; result rows include 'world' field
```

Every read API takes `world` with default `"actuality"`. Pass `None` for cross-world search.

### Inheritance resolution (Stage 3+)

When a query targets a child world, the resolver walks child-first then falls back to the parent:

```
query Hogwarts in potterverse:
  1. Look in potterverse → has Hogwarts triples? Use those.
  2. No → walk to parent (fiction) → look there.
  3. No → walk to actuality → look there.
```

Triples are surfaced from the closest ancestor that has them. A `branches-from` child can shadow a parent's triple (model an alternate universe). A `refines` child cannot contradict its parent but adds further detail.

### Contradiction detection

Contradictions are **world-local**. Adding `(Sun, color, purple)` in world `alien_perspective` does NOT conflict with `(Sun, color, yellow)` in `actuality`. The contradiction detector at `KnowledgeGraph.add_triple` only fires when the same world has a same-subject-same-predicate-different-object current triple.

## Saṃvega taxonomy

Three subtypes of artifact, each captures a different class of cognitive failure:

| Type                    | Trigger                                                           | Action                                              |
|-------------------------|-------------------------------------------------------------------|-----------------------------------------------------|
| `fact_gap`              | Known world, missing knowledge                                    | Existing thought-seed mechanism                     |
| `frame_gap`             | Claims presuppose a world that hasn't been instantiated           | Spawn a candidate world (Stage 6)                    |
| `consistency_violation` | Claim contradicts an established triple in its own world          | Flag at output time (Stage 2: shipped)              |

Stage 2 is shipped — `consistency_detector.py` runs at output time, extracts plausible named entities from the response, and flags any that don't appear in the user's prompt, conversation history, grounding data fields, or the KG. The samvega artifact lands in `/knowledge/samvega/` with the standard `cross_tier_audit` review queue.

## The risky operation: merge

Two worlds turning out to be the same is frame-level belief revision. It is the structural twin of the failure Marcus describes when LLMs commit to an identity on the strength of similarity ("statistically he's probably British").

The merge contract (Stage 5):

- **Proposal**: similarity may propose, only evidence + grounded coreference commits.
- **Coreference resolution**: identify which entities in world A correspond to which entities in world B. Record linkage problem — 40 years of literature, still hard. GAIA currently matches entities by string equality.
- **Approval queue**: every merge lands in `candidates/` via gaia-study, requires explicit Architect (Azrael) approval. Not auto-promotable.
- **Reversibility**: pre-merge snapshot of both worlds is preserved. Merge record kept. Reverse-merge is a first-class operation.
- **Locality**: the merge is a graph operation on two world atoms — collapse two nodes, reconcile their edges and quads. The atomic-ID design means it's local; no global string rewrite of every quad.

## Lifecycle

Most worlds are ephemeral. A hypothetical you reason about for five minutes shouldn't calcify into a permanent named graph.

- **Ephemeral worlds**: in-memory only, scoped to a session or single turn, never persisted.
- **Durable worlds**: persisted to disk, available across sessions.

Promotion criterion: ephemeral becomes candidate-for-durable when triple count exceeds threshold OR the world is referenced across sessions. Promotion lands in the review queue (Stage 6).

Auto-GC: ephemeral worlds older than session lifetime get dropped.

## What this doesn't solve

Stated plainly so the ambition stays honest:

- **Induction** — reading a new novel and building the right model with the right inferences is still done by the extractor and the base model, with all the out-of-distribution weakness intact.
- **Model-quality bound** — every quality of any given world model is still bounded by the LLM that populated it.
- **Theory-of-mind** at scale — `belief_of:<agent>` is a modality, but the actual representation of what another agent believes is still a flat set of triples, not a structured agent model.

What partially redeems the LLM bound: the consistency check. Even a sloppy extractor now gets caught when it asserts something a world has already ruled out. That's more than it could do alone.

## Staged rollout status

| Stage | Issue | Status   | Description |
|-------|-------|----------|-------------|
| 2     | (n/a) | shipped  | Consistency-violation detector at output time (built first as proof of value) |
| 0     | hrp   | shipped  | KG entity extraction cleanup — token-fragment noise removed, real confidence/source |
| 1     | t2m   | shipped  | `world` column on triples; default `actuality`; all read/write methods world-scoped |
| doc   | o8a   | shipped  | This blueprint |
| 3     | 4da   | open     | World registry as DAG with typed edges; Heimric campaign migration |
| 4     | 80o   | open     | Modality enforcement (fiction triples can't leak to actuality) |
| 5     | 8pk   | open     | Merge mechanism with coreference resolver |
| 6     | azr   | open     | Ephemeral vs durable world lifecycle |

## Affected files

| Path                                                        | Role |
|-------------------------------------------------------------|------|
| `gaia-common/gaia_common/utils/knowledge_graph.py`          | Quad store, schema migration, world-scoped queries |
| `gaia-common/gaia_common/utils/mempalace.py`                | Cleaned entity extractor that feeds the KG |
| `gaia-core/gaia_core/cognition/consistency_detector.py`     | Stage 2 consistency-violation detector |
| `gaia-mcp/gaia_mcp/tools.py`                                | `kg_query` / `kg_add` / `kg_invalidate` / `kg_timeline` accept `world` param |
| `gaia-common/tests/utils/test_knowledge_graph_worlds.py`    | Stage 1 lock-in tests |
| `gaia-common/tests/utils/test_mempalace_extractor.py`       | Stage 0 lock-in tests |
| `scripts/cleanup_kg_noise.py`                               | One-shot cleanup utility for legacy noise |

## See also

- `knowledge/Dev_Notebook/2026-05-21_world_model_design.md` — design conversation and decision rationale (gitignored, lives only on disk)
- `gaia-common/gaia_common/utils/knowledge_graph.py` docstrings — current API and parameter conventions
- bd issues `hrp`, `t2m`, `o8a`, `4da`, `80o`, `8pk`, `azr` — staged work tracker

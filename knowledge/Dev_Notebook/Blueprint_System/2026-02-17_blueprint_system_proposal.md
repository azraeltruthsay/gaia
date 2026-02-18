# GAIA — Architectural Proposal
## Blueprint System · Web UI · Self-Modelling

> Distilled from design session, February 2026.
> Blueprint schema files (`blueprint.py`, `blueprint_io.py`, `gaia-core.yaml`) delivered separately.

---

## What This Is

A unified plan covering three interconnected systems that must be built together because each one depends on the others:

- **Blueprint System** — GAIA's machine-readable self-model
- **Web UI** — mission control dashboard, not a chatbot wrapper
- **Self-Reflection in Study** — autonomous blueprint maintenance during sleep cycles

The guiding principle throughout: **the graph only ever shows what's real.** Everything flows from that.

---

## Core Concepts

### Blueprints Are Not Documentation

Blueprints are GAIA's proprioceptive self-model — a continuous, structured understanding of what she is made of. The human-readable `.md` files are *generated from* the YAML. You never edit the markdown directly. This means docs cannot go stale.

### Two Epistemic States

Every blueprint is either **prescriptive** or **descriptive**. They share a schema but mean different things:

| | Candidate Blueprint | Live Blueprint |
|---|---|---|
| **What it is** | What we intend to build | What actually exists |
| **Location** | `knowledge/blueprints/candidates/` | `knowledge/blueprints/` |
| **Written by** | Builder panel / LLM / human | Study (discovery / reflection) |
| **Graph renders it?** | ❌ Never | ✅ Always |
| **Promotion gate?** | ✅ Required before promotion | N/A |

### Graph Edges Are Derived, Never Stored

Connections between services are computed at render time by matching interface definitions across blueprints. Adding a new service with matching interfaces automatically wires it into the graph. No manual topology definition, ever.

---

## Blueprint Schema

### Key Design Decisions

**Interface-first contract model.** The contract section models *interfaces*, with transport as a property of each. This means the same logical connection can be expressed over REST today and upgraded to gRPC later without restructuring the blueprint — just add a `NegotiatedTransport` with a preference order. The graph shows "upgrade available" in a lighter style.

**Supported transports:** `http_rest` · `websocket` · `sse` · `event` (topic-based) · `mcp` · `grpc` · `direct_call`

**Per-section confidence scores.** Discovery produces blueprints where `runtime` and `contract` are `HIGH` confidence (port and routes are unambiguous in code) but `intent` is `LOW` (requires inference). Confidence upgrades over reflection cycles as evidence accumulates. The dashboard surfaces low-confidence sections as review prompts, not errors.

**The `genesis` flag.** All new blueprints start with `genesis: true`. This means "first impression, not yet validated by experience." The flag clears after the first reflection cycle that checks the blueprint against real service behaviour. That transition — first impression to validated self-knowledge — is logged in the dev journal.

**The `intent` layer.** Design rationale, decisions, and open questions live *inside* the blueprint, not in separate documentation. `open_questions` is where GAIA surfaces her own uncertainty about her design. Populated by the builder session (seeded by the LLM during design) or inferred by study during reflection.

### Meta Fields

```yaml
meta:
  status: candidate | live | archived
  genesis: true                    # clears after first successful reflection
  generated_by: manual_seed | builder_seed | discovery | reflection | gaia_initiated
  blueprint_version: "0.1"
  schema_version: "1.0"
  last_reflected: null
  confidence:
    runtime: high | medium | low
    contract: high | medium | low
    dependencies: high | medium | low
    failure_modes: high | medium | low
    intent: high | medium | low
  reflection_notes: null           # natural language summary written to prime.md on wake
  divergence_score: null           # 0.0–1.0, candidate vs live diff after promotion
```

---

## Two Modes of Blueprint Genesis

### Mode 1 — Reflection (existing service)
- **Trigger:** Sleep cycle (`SLEEPING` state in study)
- **Input:** Prior blueprint as context + current source files
- **Output:** A *diff* — what changed, what's inconsistent, new open questions
- **Confidence:** High — has a baseline to reason against
- **Clears `genesis`:** Yes, on first successful run

### Mode 2 — Discovery (new service, no prior blueprint)
- **Trigger:** Promotion pipeline detecting no existing blueprint, or builder panel requesting it
- **Input:** Source files only — cold start
- **Output:** A *first impression* — full blueprint, lower confidence in intent sections
- **`genesis: true`** until a reflection cycle validates it

### Three Places a New Service Can Appear

1. **Manual development** — developer writes service in `candidates/`. Promotion pipeline detects no blueprint, triggers discovery as a gate. Blueprint generation is a prerequisite for promotion, not an afterthought.

2. **Builder panel** — LLM-assisted composition session. Builder emits a `BLUEPRINT_SEED` alongside generated code. Discovery validates and enriches the seed rather than starting blind. The LLM's design intent is preserved, not thrown away and re-inferred.

3. **GAIA-initiated** (future) — GAIA identifies a capability gap, drafts a service. Same path as builder panel, but GAIA is the author of the seed. Surfaces in the review queue for human approval before promotion.

---

## The Service Lifecycle Arc

Every service follows this arc regardless of how it was created:

```
GENESIS  (candidate blueprint, genesis: true)
    ↓  first reflection cycle validates against reality
ESTABLISHED  (candidate → live via promotion pipeline)
    ↓  ongoing sleep-cycle reflection keeps blueprint current
MAINTAINED  (live, continuously accurate)
    ↓  service retired
ARCHIVED
```

### The Divergence Score

After promotion, study runs discovery on the live code and generates the live blueprint. This is then automatically diffed against the candidate blueprint (the prescriptive design) to produce a `divergence_score` (0.0–1.0):

- **Low score** — code faithfully implements the design ✓
- **High score** — implementation drifted from design, flagged for human review

Over time, the divergence score history tells you how well the LLM-assisted generation process is working.

---

## Self-Reflection in Study

Blueprint maintenance lives in `gaia-study` as a new task type: `SELF_MODEL_UPDATE`. It runs during the `SLEEPING` state after any pending embeddings are queued — giving the sleep cycle genuine cognitive character.

### Why Study, Not a Script

An external scanner *looks at* GAIA. Self-reflection is something GAIA *does*. The distinction affects where the capability lives (inside study, not in `scripts/`), what triggers it (sleep cycle, not cron), and what it produces (a cognitive artefact in `prime.md`, not a log file).

### What Happens During Sleep

```
AWAKE → DROWSY → SLEEPING
                    ↓
           SELF_MODEL_UPDATE task
           ├── for each service with no blueprint → DISCOVERY
           └── for each service with existing blueprint → REFLECTION
                    ↓
           write updated YAML to candidates/
           generate markdown from YAML
           write reflection_notes to prime.md
                    ↓
           WAKING (Prime has narrative awareness of what changed)
```

### The prime.md Integration

Study writes a natural language summary to `prime.md` — not a file diff, a cognitive note. Example:

> *"During this cycle I noticed gaia-core has started calling a new endpoint on gaia-mcp that doesn't appear in either service's blueprint. Flagged as open question in both blueprints."*

Prime reads this on wake and has contextual awareness of what changed while it was sleeping.

### The Human Is Still the Gate

Study writes to `candidates/` only. Blueprint changes surface in the dashboard review queue before propagating to the live graph. Human approves individually or in bulk. On approval, candidate blueprint is promoted to live and the graph updates. Self-reflection that can directly modify the system's self-model without oversight is explicitly not what we want.

---

## Web UI — Mission Control Dashboard

This is not a chatbot wrapper. The chat panel is one instrument among many. Think mission control, not messaging app.

### Panels

| Panel | Description |
|---|---|
| **Chat** | Talk to GAIA. SSE streaming for token output. |
| **System State** | Service health, GPU ownership, sleep state. Live polling from orchestrator. |
| **Action Buttons** | Named cognitive modes — `Sleep Cycle`, `Wake`, `Study`, `Embed`, `History Observer Review`. Calls to existing orchestrator endpoints, labelled with cognitive framing not HTTP routes. |
| **Graph** | Visual execution graph derived from live blueprints. Read-only. |
| **Blueprint Detail** | Click any node → drawer showing rendered markdown alongside raw YAML. |
| **Review Queue** | Blueprint changes from study's last sleep cycle, pending human approval. |
| **Builder Panel** | Separate — see below. |

### Graph Rendering Rules

| Element | Visual treatment |
|---|---|
| Live service, validated | Normal node |
| Live service, `genesis: true` | Node with ⚠️ indicator |
| REST edge | Solid arrow |
| WebSocket edge | Double-headed arrow |
| Event edge | Dashed arrow |
| SSE edge | Animated arrow |
| Planned edge | Grey, lighter weight |
| Edge with fallback available | Amber (degraded-but-alive path) |
| Upgrade available (NegotiatedTransport) | Edge badge |

### Tech Stack

- **Phase 2** (minimal viable UI): Vanilla JS / HTMX served as static files from gaia-web. No build pipeline. Gets something in a browser fast.
- **Phase 4** (graph panel): Migrate to React + Cytoscape.js or similar graph library. Complexity justifies it at this point.

### New API Endpoints Needed in gaia-web

```
GET  /api/blueprints                    → list all live blueprints with meta
GET  /api/blueprints/{service_id}       → full blueprint JSON
GET  /api/blueprints/{service_id}/md    → rendered markdown
GET  /api/blueprints/graph              → derived topology (nodes + edges)
POST /api/blueprints/review/{id}/approve → promote candidate blueprint to live
```

---

## Builder Panel

Available only once the live graph is trustworthy — it depends on accurate blueprints for grounding the LLM.

### Workflow

1. Human opens builder panel (separate from main dashboard)
2. Drags components from the **read-only live graph** as a palette
3. Chats with the LLM — which has full context of all live blueprints loaded automatically
4. LLM assists in designing the new service, producing a **BLUEPRINT_SEED** (candidate YAML)
5. LLM generates candidate service code *from* the blueprint seed
6. Code lands in `candidates/{service_id}/`
7. Promotion pipeline runs: tests → discovery → divergence check → promote
8. New live blueprint generated from promoted code
9. New node appears in the graph

### The Key Invariant

The builder LLM writes code *to* the blueprint. Study's discovery writes the live blueprint *from* the code. The divergence score measures how faithfully those two match. This creates a feedback loop on generation quality.

### Future: GAIA-Initiated Services

Once this loop exists, GAIA can theoretically run it herself — identifying a capability gap, drafting a service blueprint, surfacing it in the builder panel with her reasoning annotated, and awaiting human approval before the promotion pipeline runs. Human oversight remains the gate. The architecture is ready for this; it doesn't need to be built yet.

---

## Phased Implementation Plan

### Phase 1 — Blueprint Foundation
*Prerequisite for everything. No graph without data.*

- [x] Blueprint schema designed (`blueprint.py`, `blueprint_io.py`)
- [ ] Add to `gaia-common/gaia_common/models/` and `utils/`
- [ ] Add `pyyaml` to `gaia-common/requirements.txt`
- [ ] Hand-author seed YAMLs for all 6 services (`gaia-core.yaml` done)
- [ ] Add blueprint validation gate to `promote_candidate.sh`

**Remaining seed YAMLs needed:** `gaia-prime`, `gaia-web`, `gaia-mcp`, `gaia-study`, `gaia-orchestrator`

---

### Phase 2 — Minimal Functional Web UI
*Get something in a browser. Polish later.*

- [ ] Static file serving from gaia-web (`StaticFiles` mount)
- [ ] Chat panel over existing `/process_user_input` endpoint
- [ ] SSE streaming for token output
- [ ] System state panel polling orchestrator health endpoints
- [ ] Action buttons as named cognitive modes

**Deliverable:** A browser you can talk to GAIA through, with live system state visible.

---

### Phase 3 — Blueprint Self-Reflection in Study
*The cognitive layer.*

- [ ] `SELF_MODEL_UPDATE` task type in gaia-study scheduler
- [ ] Discovery mode — cold-start blueprint generation from source files
- [ ] Reflection mode — diff-shaped warm update against prior blueprint
- [ ] Confidence scoring per section
- [ ] `prime.md` integration — natural language wake notes
- [ ] `genesis` flag lifecycle (set on creation, cleared on first successful reflection)
- [ ] Candidate blueprint write path (study never writes to live directory)

---

### Phase 4 — Graph Rendering
*Now that blueprints are being maintained, render them.*

- [ ] Blueprint API endpoints in gaia-web
- [ ] Graph topology endpoint (`/api/blueprints/graph`)
- [ ] Migrate UI to React + graph library (Cytoscape.js or similar)
- [ ] Graph panel with typed edge rendering
- [ ] Blueprint detail drawer (markdown + YAML on node click)
- [ ] Review queue panel for pending blueprint approvals

---

### Phase 5 — Builder Panel
*Only after graph is trustworthy.*

- [ ] Builder panel UI (separate from main dashboard)
- [ ] Live graph as drag-and-drop component palette
- [ ] LLM composition session with blueprint context loaded
- [ ] Blueprint seed generation
- [ ] Code generation from seed
- [ ] Promotion pipeline integration with divergence scoring

---

## Dependency Order Summary

```
blueprint.py + blueprint_io.py   (done)
        ↓
seed YAMLs for all 6 services   ← parallel with Phase 2
        ↓
promote_candidate.sh gate
        ↓
Phase 2: minimal web UI          ← can start immediately
        ↓
Phase 3: study self-reflection
        ↓
Phase 4: graph rendering         ← needs blueprint API
        ↓
Phase 5: builder panel           ← needs trustworthy graph
```

Phases 2 and the seed YAML work can run in parallel — neither depends on the other. Everything else is sequential.

---

## Files Delivered

| File | Destination in project |
|---|---|
| `blueprint.py` | `gaia-common/gaia_common/models/blueprint.py` |
| `blueprint_io.py` | `gaia-common/gaia_common/utils/blueprint_io.py` |
| `gaia-core.yaml` | `knowledge/blueprints/gaia-core.yaml` |

The `gaia-core.yaml` serves as the template for the five remaining manual seed blueprints. All have `generated_by: manual_seed` and `genesis: true` until study's first reflection cycle validates them.

---

*This document should be stored at `knowledge/Dev_Notebook/2026-02-17_blueprint_system_proposal.md`*

# GAIA Tiers of Knowledge

*tiers_of_knowledge.md — Where knowledge lives, how it moves, and what gets loaded when.*

---

This document supersedes and extends `memory_tiers_spec.md` (January 2026) to account for the Cognitive Index Layer, KV cache persistence, pre-inference grounding, dynamic adapter management, and the three-layer persistence model discovered during the April 2026 epistemic honesty work.

The core insight has not changed: **knowledge is not monolithic**. What has changed is our understanding of how knowledge *moves* between layers — and that the boundaries between them are not walls but membranes, with active transport mechanisms that promote, compact, and refresh knowledge across the stack in real time.

---

## The Stack

Knowledge lives at seven tiers, from most volatile to most permanent. Each tier has a different lifespan, cost to access, cost to update, and role in cognition.

```
┌─────────────────────────────────────────────────────┐
│  Tier 6: Weights (permanent — survives everything)  │  Training, ROME
├─────────────────────────────────────────────────────┤
│  Tier 5: KV Prefix State (persistent — survives     │  Samvega fold,
│          restarts, updatable each sleep cycle)       │  adapter swap
��─────────────────────────────────────────────────────┤
│  Tier 4: Reflective Artifacts (persistent files)    │  Samvega, journals
├─────────────────────────────────────────────────────┤
│  Tier 3: Structured Knowledge (persistent files)    │  Blueprints, topics
├─────────────────────────────────────────────────────┤
│  Tier 2: Semantic Memory (vector store)             │  Embeddings, RAG
├─────────────────────────────────────────────────────┤
│  Tier 1: Session Memory (conversation logs)         │  Chat history
├─────────────────────────────────────────────────────┤
│  Tier 0: Active Context (the context window)        │  What the model
│          ├── Cognitive Index (always loaded, <2KB)   │  sees RIGHT NOW
│          ├── Awareness packages (dynamic, ~500 tok)  │
│          ├── Pre-inference grounding (on demand)     │
│          └── User prompt + conversation history      │
└─────────────────────────────────────────────────────┘
```

---

## Tier Definitions

### Tier 0: Active Context *(The Context Window)*

**What the model sees on this turn.** Everything in Tier 0 is loaded into the token context for the current inference call. This is the most expensive tier — every token here costs compute and displaces room for generation.

| Component | Budget | Refresh Rate | Source |
|-----------|--------|-------------|--------|
| Identity anchor + persona | ~150 tokens | Per session | Config |
| Epistemic rules + directives | ~300 tokens | Static | Constants |
| Cognitive Index (gaia-index.md) | ~400 tokens | 60s cache | /shared/memory/ |
| Awareness packages (time, state) | ~200 tokens | Per turn | awareness.py |
| World state snapshot | ~150 tokens | Per turn | world_state.py |
| Temporal context | ~100 tokens | Per turn | temporal_context.py |
| CIL grounding (topic snippets) | 0-500 tokens | Per turn, on demand | Topic files |
| Web search grounding | 0-300 tokens | Per turn, on demand | DuckDuckGo |
| Semantic probe context | 0-300 tokens | Per turn, on demand | Vector store |
| RAG documents | 0-2000 tokens | Per turn, on demand | Vector store |
| Conversation history | ~1000-3000 tokens | Per turn | Session memory |
| User prompt | Variable | Per turn | User input |
| **TOTAL BUDGET** | **≤6000 tokens** | | |

**Critical rule**: Tier 0 has a hard budget. When the budget is exceeded, components must be trimmed in reverse priority order (conversation history first, then RAG, then web grounding, then CIL grounding). The Cognitive Index, identity, and user prompt are never trimmed.

**KV Prefix Cache**: The static portions of Tier 0 (identity, epistemic rules, directives) are pre-computed into a KV prefix cache. This means they cost compute only once, not on every turn. The prefix cache is segmented:

| Segment | Content | Recompute Trigger |
|---------|---------|-------------------|
| `identity` | Persona, rules, directives | Persona change |
| `tools` | Tool calling convention | Never (static) |
| `world_state` | Awareness + temporal | Every turn |
| `behavioral` | Samvega corrections | Sleep cycle fold |

The `behavioral` segment is where Tier 5 (KV prefix state) meets Tier 0 — corrections folded during sleep persist here across turns.

### Tier 1: Session Memory *(Conversation Logs)*

**What happened in this conversation and recent conversations.** Persisted as markdown or JSON files. Read by the prompt builder to populate conversation history in Tier 0.

- **Format**: Markdown transcripts in `/logs/chat_history/`
- **Lifespan**: Persisted indefinitely, archived periodically
- **Promotion**: Interesting exchanges promoted to Tier 4 (reflective) during sleep
- **Compaction**: Old turns summarized to save context budget when injected into Tier 0
- **Writer**: gaia-core (conversation logger)
- **Reader**: prompt_builder (loads recent turns into Tier 0)

### Tier 2: Semantic Memory *(Vector Embeddings)*

**Concept-level retrieval across all stored knowledge.** The vector store holds embeddings of documents, conversation excerpts, and knowledge base content. Queried by the semantic probe and RAG pipeline.

- **Format**: ChromaDB collections with metadata
- **Lifespan**: Persistent, grows continuously
- **Access**: Semantic probe (pre-inference) and explicit RAG queries
- **Injection**: Results injected into Tier 0 as semantic probe context or RAG documents
- **Writer**: gaia-study (sole writer — indexes during idle/sleep)
- **Reader**: semantic_probe.py, agent_core RAG pipeline

### Tier 3: Structured Knowledge *(Files and Blueprints)*

**Formalized, human-readable knowledge.** This is the "card catalogue content" — the actual documents that the CIL index points to.

- **Format**: Markdown, YAML, JSON files
- **Locations**:
  - `/knowledge/blueprints/` — Service contracts and architecture
  - `/knowledge/system_reference/` — Core identity documents, specs
  - `/knowledge/personas/` — Persona definitions
  - `/shared/memory/topics/` — CIL topic files (NEW)
- **Lifespan**: Persistent, versioned, human-approved
- **Access**: On demand via CIL routing or direct file read
- **Injection**: CIL grounding (Tier 6.7 in prompt builder) — only loaded when the CIL index matches an extracted entity
- **Writer**: gaia-study (topic files), human (blueprints, specs)

### Tier 4: Reflective Artifacts *(Self-Generated Insights)*

**What GAIA has observed about herself.** Produced during conversation (samvega artifacts, thought seeds) and during sleep (journal entries, self-model updates).

- **Format**: JSON (samvega), Markdown (journals)
- **Locations**:
  - `/knowledge/samvega/` — Discernment artifacts
  - `/shared/memory/prime.md` — Prime cognitive journal
  - `/shared/memory/core.md` — Core cognitive journal
- **Lifespan**: Persistent with review windows
- **Promotion paths**:
  - High-weight samvega → Tier 5 KV fold (corrections into prefix cache)
  - Tier 5 promoted samvega → Tier 6 (QLoRA training pairs)
  - Journal observations → Tier 3 topic files (via autoDream reconciliation)
- **Writer**: gaia-core (samvega), gaia-study (journals during SELF_MODEL_UPDATE)

### Tier 5: KV Prefix State *(Persistent Attention)*

**The middle layer — learning without training.** Corrections, preferences, and accumulated context baked into the KV prefix cache during sleep cycles. More persistent than Tier 0 (survives restarts), less permanent than Tier 6 (reversible, updatable).

- **Format**: PyTorch tensor checkpoint (`.pt` file)
- **Location**: `/shared/kvcache/samvega_prefix_state.pt`
- **Lifespan**: Survives restarts, updated each sleep cycle
- **Content**: Samvega corrections formatted as the `behavioral` segment of the prefix cache
- **Portability**: Cross-device (GPU↔CPU) via dtype casting in save/load
- **Mechanism**:
  1. Sleep task queries high-weight samvega artifacts
  2. Formats as correction context
  3. Injects into prefix cache `behavioral` segment
  4. Forward pass processes corrections into attention state
  5. New KV state saved to disk
  6. Next wake: model loads with corrections in prefix
- **Reversibility**: Delete the file → model reverts to base weights behavior
- **Writer**: `samvega_kv_fold` sleep task

**LoRA Adapter State** also lives at Tier 5 — dynamic adapters that modify the model's behavior without changing base weights:

| Adapter | Purpose | Loaded When |
|---------|---------|------------|
| `primary_school` | Identity + voice + tools + epistemic | Always (merged into base) |
| `conversational` | Social personality tuning | On demand per persona |
| `gaia_architecture` | Technical architecture knowledge | When discussing GAIA internals |
| *(future)* `code_replace` | Tool calling format | When tool routing active |

Adapters are loaded/unloaded dynamically via the GAIA Engine's `/adapter/load` and `/adapter/set` endpoints. The active adapter set is part of the model's runtime state — changing it changes behavior immediately without retraining.

### Tier 6: Weights *(Permanent Knowledge)*

**What the model IS.** The base model weights plus any ROME edits or merged adapter weights. This is the most expensive tier to update (requires training or surgery) but the most permanent — it survives everything.

- **Format**: Safetensors (GPU) / GGUF (CPU)
- **Location**: `/models/prime`, `/models/core`, `/models/nano`
- **Lifespan**: Permanent until next training cycle
- **Update mechanisms**:
  - **Primary School QLoRA**: Unified training curriculum → adapter → merge
  - **ROME weight surgery**: Targeted single-layer edits for residual refusals
  - **SAE-guided abliteration**: Precision feature suppression with safety
- **Update frequency**: Weeks to months
- **Validation required**: 4-axis validation (knowledge, refusal, epistemic, anti-confabulation)
- **Writer**: Training pipeline (`clean_abliterate_prime.py`)

---

## Knowledge Transport Mechanisms

Knowledge doesn't just sit in tiers — it moves between them through active transport:

### Upward Promotion (volatile → permanent)

```
Tier 0 (context) → Tier 1 (session log)
  Mechanism: conversation_logger writes every exchange

Tier 1 (session) → Tier 2 (semantic)
  Mechanism: gaia-study indexes session transcripts during idle

Tier 1 (session) → Tier 4 (reflective)
  Mechanism: samvega detects corrections, saves artifacts

Tier 4 (reflective) → Tier 5 (KV state)
  Mechanism: samvega_kv_fold sleep task

Tier 4 (reflective) → Tier 6 (weights)
  Mechanism: Tier 5 promoted artifacts → QLoRA training pairs
```

### Downward Retrieval (permanent → volatile)

```
Tier 6 (weights) → Tier 0
  Mechanism: The model's behavior IS the weights. Always present.

Tier 5 (KV state) → Tier 0
  Mechanism: Prefix cache loaded on wake. Corrections in `behavioral` segment.

Tier 5 (adapters) → Tier 0
  Mechanism: LoRA adapter loaded/swapped, changes inference immediately.

Tier 3 (structured) → Tier 0
  Mechanism: CIL lookup → topic file fetch → inject as grounding.

Tier 2 (semantic) → Tier 0
  Mechanism: Semantic probe + RAG → inject as retrieved documents.

Tier 1 (session) → Tier 0
  Mechanism: Conversation history loaded by prompt builder.
```

### Lateral Refresh (same tier, updated content)

```
Tier 3 topic files ← autoDream reconciliation (planned)
  Stale topics updated with fresh observations from Tier 4 journals.

Tier 5 KV state ← samvega_kv_fold (each sleep cycle)
  New corrections layered onto existing prefix state.

Tier 0 awareness ← world_state.py (each turn)
  Clock, system metrics, immune status refreshed.
```

---

## Context Budget Management

The fundamental constraint: **Tier 0 has a finite token budget.** Everything that enters Tier 0 displaces room for the model to think and generate.

### Priority Classes

When the context budget is exceeded, components are trimmed in this order (lowest priority trimmed first):

| Priority | Component | Trim Strategy |
|----------|-----------|---------------|
| P0 (never trim) | User prompt | Truncate if >2000 tokens |
| P0 (never trim) | Identity + persona | Fixed, small |
| P1 (trim last) | Cognitive Index | Fixed, <2KB |
| P2 | Epistemic rules | Can compress to essentials |
| P3 | Awareness + world state | Can omit temporal details |
| P4 | CIL grounding | Trim snippets to titles |
| P5 | Web search grounding | Trim to titles only |
| P6 | Semantic probe context | Reduce hit count |
| P7 | RAG documents | Reduce per-doc cap |
| P8 (trim first) | Conversation history | Summarize older turns |

### Budget Enforcement

The prompt builder should enforce the budget BEFORE assembling the final prompt:

1. Calculate fixed costs (identity, persona, rules, CIL index, awareness)
2. Reserve space for user prompt + minimum generation room (~1000 tokens)
3. Allocate remaining budget to variable components by priority
4. If over budget: trim from P8 upward until within budget

### Tier-Appropriate Responses

Not every request needs every tier. The cognitive tier determines what to load:

| Cognitive Tier | Tier 0 Components Loaded |
|---------------|--------------------------|
| **Nano (slim mode)** | Identity + 8 few-shot examples + user prompt. Nothing else. |
| **Core (standard)** | Identity, rules, CIL index, awareness, semantic probe, conversation history, user prompt |
| **Core (RAG)** | Standard + RAG documents (reduce history to compensate) |
| **Prime (focus)** | Standard + full RAG + web grounding + extended history |

---

## Dynamic Swapping

### KV Cache Swapping

The KV prefix cache can be saved and loaded across device boundaries (GPU↔CPU):

- **Save**: `POST /cache/save` — serializes to CPU float32, device-agnostic
- **Load**: `POST /cache/load` — casts to target device dtype automatically
- **Use case**: FOCUSING transition saves Core's warm cache, loads Prime's. When FOCUSING ends, restore Core's cache for instant resume.

### LoRA Adapter Swapping

Adapters are loaded/unloaded without model restart:

- **Load**: `POST /adapter/load {"name": "...", "path": "..."}`
- **Set active**: `POST /adapter/set {"name": "..."}`
- **Unload**: `POST /adapter/unload {"name": "..."}`
- **Use case**: Switch from `conversational` adapter to `gaia_architecture` adapter when the topic changes from casual chat to technical discussion.

### Journal Refresh Cycle

The cognitive journals (`prime.md`, `core.md`) are read on wake and written during sleep:

- **Read**: Part of the golden thread — model's first perception on wake
- **Write**: SELF_MODEL_UPDATE during sleep (when active)
- **Frequency**: Every sleep cycle (approximately every 30-60 minutes of active use)
- **Purpose**: Continuity. The journals carry forward observations, preferences, and context that don't fit in the KV prefix cache but need to persist across sessions.

---

## Relationship to Other Specs

| Document | Relationship |
|----------|-------------|
| `memory_tiers_spec.md` | **Predecessor**. This document extends Tiers 0-5 with Tier 5 (KV persistence) and Tier 6 (weights), and adds transport mechanisms. |
| `tiers_of_cognition.md` | **Sibling**. Cognitive tiers determine which knowledge tiers are accessed per request. |
| `layered_identity_model.md` | **Input**. Identity Tier I is always in Tier 0 context. Identity is knowledge that never leaves. |
| `gaia_cognitition_protocol.md` | **Specification**. The CognitionPacket format and MTU budget rules. This document operationalizes those budgets. |
| CIL proposal | **Implementation**. The CIL is the navigational tier between Tier 0 and Tiers 2-3. |

---

## Design Principles

1. **Weights hold behaviors, context holds facts.** Never train port numbers into weights. Never rely on context for identity.

2. **The index is always loaded, the content is fetched on demand.** The CIL index costs ~400 tokens always. Topic files cost 0 tokens until needed.

3. **Budget is finite, prioritize ruthlessly.** The user's prompt and the model's identity are sacred. Everything else can be trimmed.

4. **Knowledge moves upward through observation and downward through retrieval.** The system is a living cycle, not a static archive.

5. **Reversibility scales inversely with permanence.** Tier 0 is gone next turn. Tier 5 survives restarts but can be deleted. Tier 6 survives everything but requires retraining to change.

6. **Every tier has a sole writer.** gaia-core writes Tiers 0, 1, 4. gaia-study writes Tiers 2, 3, 5. Training pipeline writes Tier 6. No conflicts.

7. **Refresh often enough for continuity, rarely enough for stability.** Journals every sleep cycle. KV fold every sleep cycle. Adapters on persona change. Weights on major training cycles.

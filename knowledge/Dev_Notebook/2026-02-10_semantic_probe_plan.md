# Pre-Cognition Semantic Probe — Design & Implementation Plan

**Date:** 2026-02-10
**Status:** Proposed
**Author:** Azrael + Claude
**Scope:** candidates/gaia-core, candidates/gaia-common

---

## The Idea

Before intent detection runs, extract "interesting" words and phrases from the user's input and run them through the vector store. If any hit with sufficient similarity, the **source collection and context** of those hits become part of the cognition packet — informing intent detection, model selection, persona, and tool routing downstream.

This generalizes the current keyword-based persona switching (`get_persona_for_request`) into a **semantic** approach. Instead of maintaining hardcoded keyword lists for every domain, GAIA discovers relevance by what's actually indexed.

### Why This Matters

1. **DND lore problem (the catalyst):** A user says "What happened at Rogue's End?" — currently this only gets routed to the D&D persona if "Rogue's End" appears in the `PERSONA_KEYWORDS` list. If the user mentions a location that was ingested but not in the keyword list, GAIA misses it entirely. The semantic probe would catch it because Rogue's End is in the vector store.

2. **Universal applicability:** This works for ANY knowledge domain — system docs, project notes, meeting logs, code documentation. If a phrase has semantic proximity to indexed content, GAIA discovers context automatically.

3. **Intent enrichment:** Knowing *which collection* a phrase matched against tells the intent detector what domain the user is operating in, without needing LLM classification.

---

## Current Architecture (What Exists)

### Keyword-Based Persona Routing
```
user_input → get_persona_for_request() → (persona_name, knowledge_base_name)
                ↓ hardcoded keyword match
         PERSONA_KEYWORDS dict in persona_switcher.py
```
- Brittle: requires manual keyword maintenance
- Misses anything not in the list
- Binary: either matches or doesn't (no confidence/relevance)

### RAG Query (Post-Intent)
```
user_input → knowledge_enhancer.enhance_packet() → embedding_query()
                ↓ runs AFTER intent detection
         Only fires if knowledge_base_name was already set by persona routing
```
- Chicken-and-egg: RAG only runs if persona routing already identified the KB
- The full user input is the query — not optimized for entity/phrase extraction

### Intent Detection
```
user_input → detect_intent() → Plan(intent, read_only)
                ↓ reflex patterns + LLM classification
         No awareness of what's in the vector store
```
- Intent detection operates without knowledge of what domains have relevant context
- Can't factor in "this message mentions 3 D&D entities" as a signal

---

## Proposed Architecture: The Semantic Probe

### New Component: `SemanticProbe`

A lightweight pre-cognition step that runs **before** intent detection and persona selection. It:

1. **Extracts candidate phrases** from user input (NER-lite + noun chunk extraction)
2. **Probes all active vector collections** with those phrases
3. **Returns hits** with collection source, similarity score, and matched chunk metadata
4. **Injects results** into the cognition packet as a new DataField

### Flow Change

```
BEFORE (current):
  user_input → persona_keywords → intent_detect → RAG (maybe) → model

AFTER (proposed):
  user_input → SemanticProbe → [enriched context] → persona_select → intent_detect → RAG → model
                   ↓
            probe all vector collections
            with extracted phrases
                   ↓
            hits inform persona, intent, and tool routing
```

### Phrase Extraction Strategy

Not full NER (too heavy for pre-cognition). Instead, a fast heuristic pipeline:

1. **Capitalized sequences** — "Rogue's End", "Tower Faction", "Jade Phoenix"
   Regex: `[A-Z][a-z]+(?:['']s)?\s+[A-Z][a-z]+` and similar patterns
2. **Quoted strings** — `"the Maid"`, `'BlueShot'`
3. **Unusual/rare words** — words not in a small common-word set (stopwords + top-1000 English words)
4. **Named entity patterns** — D&D notation (d20, AC 15), proper nouns, titles
5. **Full input fallback** — if no interesting phrases extracted, probe with the full input at lower weight

The extraction should be **fast** (pure regex/set operations, no model calls) and **over-inclusive** (better to probe an extra phrase than miss a real entity).

### Multi-Collection Probing

Current architecture has knowledge bases defined in `gaia_constants.json`:
```json
{
  "KNOWLEDGE_BASES": {
    "dnd_campaign": { "vector_store_dir": "dnd_campaign/vector_store" },
    "system": { "vector_store_dir": "vectordb/system" }
  }
}
```

The probe checks **all** indexed collections, not just the one currently active. This is key — the probe *discovers* which collection is relevant rather than relying on keyword routing to tell it.

### Probe Result Structure

```python
@dataclass
class ProbeHit:
    phrase: str              # The extracted phrase that matched
    collection: str          # Which knowledge base ("dnd_campaign", "system", etc.)
    chunk_text: str          # The matched chunk (truncated for context)
    similarity: float        # Cosine similarity score
    filename: str            # Source document filename
    chunk_idx: int           # Position in source document

@dataclass
class SemanticProbeResult:
    hits: List[ProbeHit]
    primary_collection: Optional[str]       # Collection with most/strongest hits (drives persona)
    supplemental_collections: List[str]     # Other collections with hits above threshold
    probe_time_ms: float                    # Performance tracking
    phrases_tested: List[str]               # What was extracted
    from_cache: int                         # Number of phrases resolved from session cache
```

### Integration Points

#### 1. Injection into CognitionPacket

New DataField key: `semantic_probe_result`

```python
packet.content.data_fields.append(DataField(
    key='semantic_probe_result',
    value=probe_result.to_dict(),
    type='json',
    source='semantic_probe'
))
```

#### 2. Context-Driven Persona Selection

The probe drives persona selection. Keyword matching (`PERSONA_KEYWORDS`) becomes a fallback:

```python
# In agent_core.py, after probe runs:
if probe_result.primary_collection:
    # Probe found a dominant domain — adopt that persona
    kb_name = probe_result.primary_collection
    persona_name = get_persona_for_knowledge_base(kb_name)
else:
    # No probe hits — fall back to keyword matching
    persona_name, kb_name = get_persona_for_request(user_input)
```

This is the **reverse** of the current flow. Instead of persona → KB, it's KB → persona. GAIA chooses her own hat based on what the conversation is about.

#### 3. Intent Detection Enhancement

Pass probe results to `detect_intent()` so the LLM classifier has domain context:

```python
plan = detect_intent(
    user_input,
    config,
    lite_llm=lite_llm,
    probe_context=probe_result  # NEW: semantic context
)
```

The intent prompt can include: "The user's message references content from the D&D campaign knowledge base (matched entities: Rogue's End, Tower Faction)."

#### 4. RAG Optimization

When the probe has already found relevant chunks, the later RAG step can:
- Skip re-querying the same collection (probe already found the hits)
- Use probe hits as seed context
- Focus on retrieving adjacent/complementary chunks

#### 5. Prompt Builder

`build_from_packet()` picks up `semantic_probe_result` and formats it:

```
[SEMANTIC CONTEXT]
The user's message references content from your knowledge bases:
- "Rogue's End" matched dnd_campaign (similarity: 0.87) — from locations_braeneage.md
- "Tower Faction" matched dnd_campaign (similarity: 0.82) — from factions_overview.md
```

---

## Performance Budget

The probe must be **fast** — it runs before intent detection on every turn.

| Step | Budget | Notes |
|------|--------|-------|
| Phrase extraction | < 5ms | Pure regex/set ops, no model |
| Embedding phrases | < 50ms | MiniLM is fast, batch embed 3-8 phrases |
| Vector search per collection | < 20ms | NumPy cosine similarity on small indices |
| Total (2 collections, 5 phrases) | < 100ms | Well within acceptable latency |

### Safeguards

- **Max phrases to probe:** 8 (cap extraction output)
- **Min phrase length:** 3 chars (skip noise)
- **Similarity threshold:** 0.40 (below this, not a meaningful hit)
- **Max collections to probe:** configurable, default all
- **Cache:** If the same session has probed recently, skip phrases already tested
- **Short-circuit:** If input is < 3 words or a known reflex command (exit, help), skip probe entirely

---

## Implementation Phases

### Phase 1: Core Probe Engine
- [ ] New file: `candidates/gaia-core/gaia_core/cognition/semantic_probe.py`
- [ ] `extract_candidate_phrases(text: str) -> List[str]` — regex/heuristic extraction
- [ ] `probe_collections(phrases, config) -> SemanticProbeResult` — multi-collection vector search
- [ ] Unit tests for phrase extraction (D&D names, general text, edge cases)

### Phase 2: Packet Integration
- [ ] Wire probe into `agent_core.run_turn()` — after input received, before persona selection
- [ ] Add `semantic_probe_result` DataField to packet
- [ ] Update `prompt_builder.py` to format probe results into system prompt

### Phase 3: Persona & Intent Enhancement
- [ ] Modify persona selection to consider probe's `dominant_collection`
- [ ] Pass probe context to `detect_intent()` for richer classification
- [ ] Deprecate some hardcoded keywords that are now covered by semantic matching

### Phase 4: RAG Dedup & Optimization
- [ ] If probe already found chunks from a collection, skip redundant RAG query
- [ ] Use probe hits as seed context for knowledge_enhancer
- [ ] Merge probe + RAG results into a unified retrieved_documents set

### Phase 5: Observability & Tuning
- [ ] Log probe results to thoughtstream for debugging
- [ ] Track hit rate metrics (how often probe finds something useful)
- [ ] Tune similarity threshold based on observed precision/recall
- [ ] Add probe timing to packet.metrics

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Probe adds latency to every turn | Performance budget enforced; short-circuit for trivial inputs |
| False positives (common words matching indexed content) | Similarity threshold (0.40); filter common English words before probing |
| Too many collections slow probing | Max collection limit; lazy-load indices only when needed |
| Probe results confuse intent detector | Probe is context, not directive — intent detector can ignore low-confidence hits |
| Phrase extraction misses important terms | Full-input fallback as last resort; iterative tuning of extraction patterns |

---

## Example Walkthrough

**User input:** "What happened at Rogue's End after the Tower Faction arrived?"

1. **Phrase extraction:**
   - "Rogue's End" (capitalized sequence with possessive)
   - "Tower Faction" (capitalized sequence)

2. **Probe collections:**
   - `dnd_campaign`: "Rogue's End" → 0.89 similarity (locations_braeneage.md, chunk 3)
   - `dnd_campaign`: "Tower Faction" → 0.85 similarity (factions_overview.md, chunk 1)
   - `system`: "Rogue's End" → 0.12 similarity (no meaningful match)
   - `system`: "Tower Faction" → 0.08 similarity (no meaningful match)

3. **Dominant collection:** `dnd_campaign` (2 strong hits)

4. **Persona selection:** `dnd_player_assistant` (derived from dominant collection)

5. **Intent detection receives:** "User references D&D campaign entities (Rogue's End, Tower Faction)"

6. **RAG step:** Skips probing dnd_campaign again; uses probe chunks as seed context; retrieves adjacent chunks for fuller picture.

7. **Prompt to model includes:**
   ```
   [SEMANTIC CONTEXT]
   Matched knowledge: dnd_campaign
   - "Rogue's End" (0.89) — from locations_braeneage.md
   - "Tower Faction" (0.85) — from factions_overview.md
   ```

---

## Design Decisions (Resolved 2026-02-10)

### 1. Persona is context-driven, not gated

**Decision:** The probe drives persona selection. Persona becomes a **contextual parameter** that GAIA adopts based on what the probe finds — not a rigid gate controlled by keyword lists.

- `PERSONA_KEYWORDS` becomes a fallback for when the probe has no hits (cold start, trivial inputs, empty vector stores)
- When the probe finds a dominant collection, the associated persona is adopted naturally
- No override logic needed — the probe *is* the persona signal
- Future option: tighter persona control can be layered on later if needed

**Rationale:** GAIA should choose her own hat based on what the conversation is about. The knowledge shapes the identity, not the other way around.

### 2. Multi-domain: Primary + Supplemental

**Decision:** The collection with the strongest/most hits becomes **primary** (drives persona, main context). Other matched collections are included as **supplemental** context, clearly labeled by source.

- Primary collection: highest aggregate similarity score or most hits above threshold
- Supplemental collections: any other collections with hits above threshold, tagged as secondary
- Prompt builder formats them distinctly:
  ```
  [PRIMARY CONTEXT — dnd_campaign]
  - "Rogue's End" (0.89) — locations_braeneage.md
  - "Tower Faction" (0.85) — factions_overview.md

  [SUPPLEMENTAL — system]
  - "character sheet update" (0.52) — system_logs_feb.md
  ```
- The model sees both but understands the main frame

**Rationale:** Full merge creates context soup that can confuse the model about what kind of answer is expected. Primary + supplemental preserves hierarchy while retaining cross-domain awareness. Can graduate to full merge later if this feels limiting.

**Example:** "Check the system logs for when Rupert's character sheet was last updated"
- Primary: `system` (the user wants logs — action is system-oriented)
- Supplemental: `dnd_campaign` (who Rupert is, character context)

### 3. Adaptive probing with session cache

**Decision:** Probe adaptively, not on every turn unconditionally.

**Mechanism:**
- **Session-level phrase cache:** Map of `phrase → ProbeHit[]` with TTL
- First time a phrase is seen in a session → full probe, cache results
- Subsequent turns mentioning the same phrase → reuse cached hits, skip re-embedding
- Cache eviction: after `N` turns (default 10) or explicit session reset
- **Short-circuit rules** (skip probe entirely):
  - Reflex commands: exit, help, status, list_tools
  - Messages under 3 words
  - Messages that are exact duplicates of the previous turn (loop protection)
- **New phrase detection:** Only probe phrases not already in the session cache — avoids redundant work while still catching new entities as they appear in conversation

**Performance impact:** First turn with new entities pays ~100ms. Subsequent turns referencing the same entities pay ~0ms (cache hit). Net effect: near-zero overhead for ongoing conversations after the first probe.

**Data structure:**
```python
@dataclass
class SessionProbeCache:
    phrase_hits: Dict[str, List[ProbeHit]]  # phrase → cached hits
    turn_ages: Dict[str, int]               # phrase → turn number when cached
    current_turn: int                        # incremented each turn
    max_age: int = 10                        # evict after N turns
```

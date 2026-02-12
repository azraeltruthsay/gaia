# Dev Journal Entry: 2026-02-10 - write_file End-to-End Integration

**Date:** 2026-02-10
**Author:** Claude Code (Opus 4.6) via Happy

## Context

The `write_file` MCP tool was already implemented and tested at the MCP layer (JSON-RPC), but GAIA couldn't invoke it through natural language because three integration points in gaia-core were broken:

1. **Safety gate permanently locked** — every packet was created with `execution_allowed=False` and nothing ever unlocked it, so all sidecar actions were denied.
2. **EXECUTE parser produced wrong params** — put everything into `{"command": ...}` instead of structured `{"path": ..., "content": ...}`.
3. **Tool selector didn't know about MCP tools** — hardcoded catalog only had legacy `ai.*` primitives.

Plan file: `/home/azrael/.claude/plans/nested-cooking-robin.md`

## Changes Implemented

### 1. Tiered Safety Gate (`packet_utils.py`)

**Files:** `gaia-common/gaia_common/utils/packet_utils.py`, `gaia-core/gaia_core/cognition/packet_utils.py`

Replaced the all-or-nothing `is_execution_safe()` with a tiered check:
- **Explicit governance allow** — if `execution_allowed=True` AND a whitelist ID is set, all actions pass (unchanged behavior).
- **Safe tool tier** — if governance hasn't explicitly allowed, check if ALL actions are in `SAFE_SIDECAR_TOOLS` (read-only, memory, fragment operations). These pass without human approval.
- **Sensitive tools** — anything not in the safe set (write_file, run_shell, etc.) is blocked at the safety gate. These get routed to MCP approval via the 403 handling in dispatch.

### 2. Structured EXECUTE Parser (`output_router.py`)

**File:** `gaia-core/gaia_core/utils/output_router.py`

Updated `_parse_llm_output_into_packet()` to support two EXECUTE formats:
- **Structured:** `EXECUTE: write_file {"path": "/knowledge/doc.txt", "content": "..."}`
- **Legacy:** `EXECUTE: run_shell ls -la /knowledge`

Parser uses `split(None, 1)` to separate tool name from args, then tries `json.loads()` on the args. Falls back to `{"command": raw_args}` if JSON parsing fails.

### 3. Tool Selector Catalog (`tool_selector.py`)

**File:** `gaia-core/gaia_core/cognition/tool_selector.py`

Added `write_file` and `read_file` to `AVAILABLE_TOOLS` with proper param descriptions and `requires_approval` flags. This lets the LLM-based tool selector recognize and route to these MCP tools.

### 4. MCP JSON-RPC Fallback (`agent_core.py`)

**File:** `gaia-core/gaia_core/cognition/agent_core.py`

Replaced the "Unknown tool" error in `_execute_mcp_tool()` with a JSON-RPC dispatch fallback. Tools not handled by the local if/elif chain now get forwarded to the MCP server via `mcp_client.call_jsonrpc()`. This means any tool registered in the MCP server is automatically available through the pre-generation path.

### 5. 403 Approval Routing (`mcp_client.py`)

**File:** `gaia-core/gaia_core/utils/mcp_client.py`

Updated `dispatch_sidecar_actions()` to detect HTTP 403 from the MCP server (sensitive tool rejection) and route through `request_approval_via_mcp()` with `_allow_pending=True`. This creates a pending approval that humans can approve via Discord or the web UI, rather than silently failing.

### 6. Cheat Sheet Documentation (`cheat_sheet.json`)

**File:** `knowledge/system_reference/cheat_sheet.json`

Updated the EXECUTE definition to show both structured and shell formats. Added `write_file` and `read_file` usage examples to `tool_use_guidance`.

## Data Flow (Post-Implementation)

### Pre-generation path (tool routing loop)
```
User input → needs_tool_routing() → select_tool() [now knows write_file/read_file]
  → _execute_mcp_tool() [JSON-RPC fallback for MCP tools]
```

### Post-generation path (EXECUTE directives)
```
LLM emits EXECUTE: write_file {"path":..., "content":...}
  → _parse_llm_output_into_packet() [JSON param extraction]
  → is_execution_safe() [tiered: safe tools pass, sensitive blocked]
  → dispatch_sidecar_actions() [403 → approval flow]
```

## Process Note

Changes were edited in production (`gaia-core/`, `gaia-common/`) first, then mirrored to candidates (`candidates/gaia-core/`, `candidates/gaia-common/`). The correct workflow per the candidate SDLC should have been: edit candidates → validate → promote. Both directions are now synced. Future sessions should start in candidates.

## Files Modified

- `gaia-common/gaia_common/utils/packet_utils.py` — tiered safety gate + SAFE_SIDECAR_TOOLS
- `gaia-core/gaia_core/cognition/packet_utils.py` — same (local copy)
- `gaia-core/gaia_core/utils/output_router.py` — JSON EXECUTE parser
- `gaia-core/gaia_core/cognition/tool_selector.py` — write_file + read_file in catalog
- `gaia-core/gaia_core/cognition/agent_core.py` — MCP JSON-RPC fallback
- `gaia-core/gaia_core/utils/mcp_client.py` — 403 approval routing
- `knowledge/system_reference/cheat_sheet.json` — documentation

All mirrored to `candidates/` counterparts.

## Verification Steps

1. **Unit check**: Restart gaia-core, confirm no import errors
2. **Tool routing path**: Ask GAIA "write a test note to /knowledge/test_write.txt" — should trigger tool_selector → write_file → MCP JSON-RPC → approval pending
3. **EXECUTE path**: If LLM emits `EXECUTE: write_file {"path":..., "content":...}`, verify the safety gate allows non-sensitive tools and routes sensitive ones to approval
4. **Regression**: Ask GAIA a normal question (no tools) — should work unchanged
5. **Safety**: Confirm `EXECUTE: run_shell rm -rf /` still gets blocked (sensitive tool → approval required)

---

# Dev Journal Entry: 2026-02-10 — Pre-Cognition Semantic Probe (Concept)

**Date:** 2026-02-10 (evening)
**Author:** Azrael + Claude Code (Opus 4.6) via Happy
**Status:** Design complete, implementation pending

## Origin

Azrael's observation: for D&D lore, GAIA should naturally vector-lookup any odd words or phrases from the prompt *before* deciding what to do. If "Rogue's End" is in the vector store but not in the hardcoded keyword list, GAIA currently misses it. The insight: **this concept works for everything, not just D&D.**

## Problem Statement

GAIA's current pipeline has a sequencing problem:

1. **Persona/KB selection** happens via keyword matching (`PERSONA_KEYWORDS` dict)
2. **Intent detection** runs without knowledge of what's in the vector store
3. **RAG retrieval** only fires *if* step 1 already identified a knowledge base

This means:
- New entities added to the vector store (via ingestion) aren't discoverable unless someone also adds them to the keyword list
- Intent detection can't factor in domain context ("this mentions 3 D&D entities" is a strong signal)
- The system is brittle — keyword lists must be maintained manually

## Proposed Solution: Semantic Probe

A new pre-cognition step that runs **before** persona selection and intent detection:

1. **Extract interesting phrases** from user input (capitalized sequences, quoted strings, rare words) — pure regex, no model call, < 5ms
2. **Probe all vector collections** with extracted phrases — batch embed via MiniLM, cosine similarity search, < 100ms total
3. **Inject results** into the cognition packet as `semantic_probe_result` DataField
4. **Downstream systems** (persona selection, intent detection, RAG, prompt builder) consume probe results as enrichment

### Key Design Decisions

- **Probe runs on every turn** (with short-circuit for trivial inputs like "exit", "help")
- **All collections are probed**, not just the currently-active one — the probe *discovers* relevance
- **Phrase extraction is heuristic, not model-based** — must be fast enough to not add perceptible latency
- **Probe is additive** — it enriches context but doesn't override explicit user intent
- **Similarity threshold of 0.40** — below this, hits are noise

### What This Unlocks

1. **Self-maintaining domain routing** — as documents are ingested, they become discoverable automatically
2. **Cross-domain awareness** — a message that references both D&D lore and system config gets context from both
3. **Intent enrichment** — "user references 3 D&D entities" is a strong signal for intent classification
4. **RAG dedup** — probe hits can seed the RAG step, avoiding redundant queries

## Architectural Fit

This slots in cleanly because:
- `VectorIndexer` already supports multi-collection querying (singleton per KB)
- `DataField` injection is the standard packet enrichment pattern
- `prompt_builder.py` already iterates data_fields and formats them
- The probe's output format matches the existing `retrieved_documents` pattern

## Implementation Plan

See detailed plan: `/gaia/GAIA_Project/knowledge/Dev_Notebook/2026-02-10_semantic_probe_plan.md`

**Phases:**
1. Core probe engine (`semantic_probe.py`) — phrase extraction + multi-collection search
2. Packet integration — wire into `agent_core.run_turn()`, add DataField
3. Persona & intent enhancement — use probe results to inform routing
4. RAG dedup — avoid re-querying what the probe already found
5. Observability — logging, metrics, threshold tuning

## Response Header Feature

Also added in this session: a status header prepended to user-facing responses showing model name, packet state, and observer activity. Format:

```
[Model: Thinker (GPU) | State: processing | Observer: streaming]
```

Implemented in `agent_core.py` via `_build_response_header()` method, injected at both the slim-prompt and normal response yield points. Header is yielded but not persisted to session history (doesn't pollute context).

## Implementation Progress

### Phase 1: Core Probe Engine (Complete)
- **Created:** `candidates/gaia-core/gaia_core/cognition/semantic_probe.py` (~340 lines)
  - `extract_candidate_phrases()` — regex/heuristic phrase extraction (quoted strings, proper nouns, D&D notation, rare words)
  - `ProbeHit`, `SemanticProbeResult`, `SessionProbeCache` — data structures
  - `probe_collections()` — multi-collection vector search with session caching
  - `run_semantic_probe()` — top-level orchestrator with short-circuit rules
  - Tested: 14 diverse inputs, all extracting correctly

### Phase 2: Packet Integration (Complete)
- **Modified:** `candidates/gaia-core/gaia_core/behavior/persona_switcher.py`
  - Added `get_persona_for_knowledge_base(kb_name)` — reverse lookup (KB → persona)
- **Modified:** `candidates/gaia-core/gaia_core/cognition/agent_core.py`
  - Probe runs before persona selection in `run_turn()` (~line 635)
  - Probe-driven persona/KB selection with keyword fallback (~line 668)
  - Probe result injected as `semantic_probe_result` DataField (~line 852)
  - Thoughtstream event logged for probe results
- **Modified:** `candidates/gaia-core/gaia_core/utils/prompt_builder.py`
  - Extracts `semantic_probe_result` DataField and formats as tier 6.5
  - Primary + supplemental collection hierarchy in prompt

### Phase 3: Intent Enrichment (Complete)
- **Modified:** `candidates/gaia-core/gaia_core/cognition/nlu/intent_detection.py`
  - Added `probe_context` parameter to `detect_intent()` and `model_intent_detection()`
  - Probe context injected as a `Context:` line in the LLM intent prompt
  - Example: `Context: User references dnd_campaign entities (Rogue's End, Tower Faction)`
- **Modified:** `candidates/gaia-core/gaia_core/cognition/agent_core.py`
  - Formats probe hits into a short context string before calling `detect_intent()`
  - Lists primary collection and top matched entity names

### Phase 4: RAG Dedup (Complete)
- **Modified:** `candidates/gaia-core/gaia_core/cognition/agent_core.py`
  - When probe found >= 2 hits from the same collection RAG would query, probe chunks are converted to `retrieved_documents` format and injected directly — the MCP `embedding_query` call is skipped
  - Deduplicates by filename to avoid repeat chunks from the same source doc
  - When probe found < 2 hits (or none from that collection), RAG runs as normal
  - Tags seed docs with `"source": "semantic_probe"` for observability

### Phase 5: Observability (Complete)
- **`semantic_probe.py` — `to_metrics_dict()`:** Compact metrics summary on every `SemanticProbeResult` — includes phrases extracted/matched, similarity stats (avg/max/min), timing, cache hits, threshold used
- **`semantic_probe.py` — `ProbeSessionStats`:** Cumulative per-session tracker recording total probes, hit rate, avg probe time, cache utilization, collections seen. Logs a summary every 10 probes
- **`semantic_probe.py` — `get_session_probe_stats()`:** Public accessor for session stats (used by agent_core)
- **`semantic_probe.py` — Config-driven thresholds:** All constants (`SIMILARITY_THRESHOLD`, `_MAX_PHRASES`, `_MIN_PHRASE_LEN`, `_MIN_WORDS`, `_CACHE_MAX_AGE`, `top_k_per_phrase`) now read from `SEMANTIC_PROBE` section of `gaia_constants.json` with hardcoded fallbacks
- **`cognition_packet.py` — `Metrics.semantic_probe`:** New optional `Dict[str, Any]` field on the `Metrics` dataclass, populated with `to_metrics_dict()` output
- **`agent_core.py` — Enhanced thoughtstream events:** `semantic_probe` event now includes full `metrics` dict and per-hit details (phrase, collection, similarity, filename). Both hit and no-hit cases emit an event. `turn_end` event includes cumulative `probe_session_stats`
- **`gaia_constants.json` — `SEMANTIC_PROBE` config:** New config section with all tunable thresholds (`similarity_threshold`, `max_phrases`, `min_phrase_len`, `min_words_to_probe`, `cache_max_age_turns`, `top_k_per_phrase`)

## Files Created/Modified This Session

- **Created:** `knowledge/Dev_Notebook/2026-02-10_semantic_probe_plan.md` — full design doc
- **Created:** `candidates/gaia-core/gaia_core/cognition/semantic_probe.py` — probe engine
- **Modified:** `candidates/gaia-core/gaia_core/cognition/agent_core.py` — response header + probe wiring + observability
- **Modified:** `candidates/gaia-core/gaia_core/cognition/cognition_packet.py` — `Metrics.semantic_probe` field
- **Modified:** `candidates/gaia-core/gaia_core/behavior/persona_switcher.py` — reverse KB→persona lookup
- **Modified:** `candidates/gaia-core/gaia_core/utils/prompt_builder.py` — probe result formatting
- **Modified:** `candidates/gaia-core/gaia_core/cognition/nlu/intent_detection.py` — probe context in intent prompt
- **Modified:** `candidates/gaia-common/gaia_common/constants/gaia_constants.json` — `SEMANTIC_PROBE` config section

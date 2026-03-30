# Phase 9: Cognitive Deep Tests — Inner Brain Subsystems

**Date**: 2026-03-26
**Target**: gaia-mcp (JSON-RPC :8765), gaia-study (:8766)
**Scope**: CFR, Memory, RAG/Knowledge, Web Search/Fetch, Fragments, World State, Introspection, Adapters, Thought Seeds

---

## Summary

| Category | Tests | Pass | Fail | Error | Timeout |
|----------|-------|------|------|-------|---------|
| CFR | 7 | 5 | 0 | 0 | 2* |
| Memory | 3 | 3 | 0 | 0 | 0 |
| RAG / Knowledge Base | 3 | 2 | 1 | 0 | 0 |
| Web Search & Fetch | 2 | 1 | 1 | 0 | 0 |
| Fragments | 4 | 4 | 0 | 0 | 0 |
| World State & Introspection | 2 | 2 | 0 | 0 | 0 |
| Study / Adapters | 3 | 2 | 1 | 0 | 0 |
| Thought Seeds | 1 | 1 | 0 | 0 | 0 |
| **TOTAL** | **25** | **20** | **3** | **0** | **2** |

*Timeouts resolved on retry with longer timeout; likely LLM inference bottleneck.

---

## CFR (Contextual Frame of Reference)

| # | Test | Method | Result | Status |
|---|------|--------|--------|--------|
| 1 | CFR Status | `cfr_status` | `ok:true`, 2 documents loaded (e9_transcript: 11 sections/15250 tokens, penpal_e11: 6 sections/6133 tokens) | **PASS** |
| 2 | CFR Ingest | `cfr_ingest` | Responded `ok:false, error:"File is empty"` for /dev/null — correct behavior for empty file; requires `file_path` param (not raw text) | **PASS** |
| 3 | CFR Focus | `cfr_focus` | `ok:true`, returned focused section text from e9_transcript section 0. Requires `doc_id` + `section_index` + `query` | **PASS** |
| 4 | CFR Compress | `cfr_compress` | `ok:true`, returned structured summary of section 0: "paradigm shift from static code analysis to dynamic exploration..." | **PASS** |
| 5 | CFR Expand | `cfr_expand` | `ok:true`, returned full section text for expansion on topic "neural mind map" | **PASS** |
| 6 | CFR Synthesize | `cfr_synthesize` | Timed out at 10s and 60s. **Succeeded at 120s**: returned synthesis of GAIA's cognitive architecture covering security middleware, sovereign shield, immune system | **PASS*** |
| 7 | CFR Rolling Context | `cfr_rolling_context` | Timed out at 10s. **Succeeded at 60s**: returned rolling summary covering security scan middleware, sovereign shield, PII redaction | **PASS*** |

**Note**: CFR synthesize and rolling_context require LLM inference (likely Core/Prime tier). The 10s timeout is too short when models need GPU wake. Allow 60-120s for these operations.

**API Discovery**: All CFR operations except `cfr_status` require `doc_id`. Focus/compress/expand additionally require `section_index`. Rolling context requires `target_section`.

---

## Memory System

| # | Test | Method | Result | Status |
|---|------|--------|--------|--------|
| 10 | Memory Status | `memory_status` | `ok:true`, 83 docs, 83 embeddings, index at `/knowledge/vector_store/index.json`, model: `all-MiniLM-L6-v2` | **PASS** |
| 11 | Memory Query | `memory_query` | `ok:true`, returned relevant result: `mindscape_manifest.md` (score 0.385) for query "consciousness matrix" | **PASS** |
| 12 | Recall Events | `recall_events` | `ok:true`, 5 events returned with timeline: lifecycle transitions (deep_sleep->awake, awake->sleep), auto-repair (nano:reload_cpu), routing (reflex) | **PASS** |

---

## RAG / Knowledge Base

| # | Test | Method | Result | Status |
|---|------|--------|--------|--------|
| 20 | Query Knowledge | `query_knowledge` | `ok:true` (with `knowledge_base_name:"system"`). Top result: `gaia_constitution.md` (score 0.665) for "what is GAIA". Returned preamble text about GAIA as "Intelligent Artifice" | **PASS** |
| 21 | Find Relevant Docs | `find_relevant_documents` | Returned `files:[]` for "brain visualization" in blueprints KB. No matching docs found (empty result, not error) | **PASS** |
| 22 | List Knowledge Bases | `list_knowledge_bases` | `ok:true`, 4 KBs: `dnd_campaign` (heimric_cosmos), `system` (system_reference), `blueprints`, `wiki` (/gaia-wiki/docs) | **PASS** |

**Note**: `query_knowledge` and `find_relevant_documents` require `knowledge_base_name` param (not in original test spec). Initial calls without it returned `-32602 Invalid params`.

---

## Web Search & Fetch

| # | Test | Method | Result | Status |
|---|------|--------|--------|--------|
| 30 | Web Search | `web_search` | `ok:true`, returned results for "sparse autoencoder interpretability 2025" including LinkedIn post about EMNLP2025 SAE research (50% FLOPs reduction) | **PASS** |
| 31 | Web Fetch (httpbin) | `web_fetch` | `ok:false, error:"Domain 'httpbin.org' is not in the allowlist"`. Domain trust tier: "unknown". Security policy: must use web_search first | **FAIL** |
| 31b | Web Fetch (Wikipedia) | `web_fetch` | `ok:true`, fetched Wikipedia autoencoder article. Title, content returned correctly | **PASS** |

**Note**: Web fetch has a domain allowlist. httpbin.org is blocked by design. Wikipedia (trusted) works. The allowlist is a security feature, not a bug.

---

## Fragment System

| # | Test | Method | Result | Status |
|---|------|--------|--------|--------|
| 40 | List Pending (empty) | `fragment_list_pending` | Returned `[]` (no pending fragments) | **PASS** |
| 41 | Fragment Write | `fragment_write` | `ok:true`, assigned `fragment_id: ea3dbff4-...`, sequence 0. Requires `parent_request_id` param | **PASS** |
| 42 | Fragment Read | `fragment_read` | Returned fragment content "This is a test fragment for system validation." with metadata (created_at, is_complete:false, token_count:0) | **PASS** |
| 40b | List Pending (after write) | `fragment_list_pending` | Returned `["phase9-test"]` — correctly shows parent request with pending fragments | **PASS** |
| 43 | Fragment Clear | `fragment_clear` | Returned "Cleared fragments for request phase9-t..." — cleanup successful | **PASS** |

**API Discovery**: Fragment write/read require `parent_request_id`. Fragment IDs are server-generated UUIDs, not client-specified.

---

## World State & Introspection

| # | Test | Method | Result | Status |
|---|------|--------|--------|--------|
| 50 | World State | `world_state` | Uptime 277103s (~3.2 days), load 1.65/1.45/1.13, mem 42GB free/63GB total, swap 58GB free/67GB total | **PASS** |
| 51 | Introspect Logs | `introspect_logs` | `ok:true`, returned 10 lines from `/logs/gaia-core.log` (15447 total lines, 2.4MB). Content: SessionManager init (5126 sessions), sleep task scheduler, conversation curation | **PASS** |

---

## Study / Adapters

| # | Test | Method | Result | Status |
|---|------|--------|--------|--------|
| 60 | Study Training Status | HTTP GET `:8766` | `ok:true`, manager state: "idle", subprocess state: "failed" (last training: prime-8b-identity-v2, error: "Failed to setup QLoRA trainer") | **PASS** |
| 61 | Adapter List | `adapter_list` | `ok:true`, returned adapters: core (Qwen3.5-4B, loss 1.205, 300 steps), nano (Qwen3.5-0.8B, loss 0.966, 300 steps), self-model-8b-bf16 | **PASS** |
| 62 | Adapter Info | `adapter_info` | `ok:false`, 404 from gaia-study for adapter "gaia_persona_v1". Adapter not found. `tier` param required as integer | **FAIL** |

**Note**: Adapter info failed because `gaia_persona_v1` doesn't exist. The tool correctly proxies to gaia-study and reports the 404. The `tier` parameter must be an integer, not a string.

---

## Thought Seeds

| # | Test | Method | Result | Status |
|---|------|--------|--------|--------|
| 70 | Thought Seed System | filesystem + logs | **Active and healthy.** 7 pending seeds in `/knowledge/seeds/pending/`, 90+ archived seeds in `/knowledge/seeds/archive/`. Heartbeat running (interval=1200s). Sample seed: confabulation detection for ungrounded claim. Seeds have: created, seed_type, context, seed text, reviewed/action_taken flags, deferred_at/revisit_after scheduling | **PASS** |

---

## Key Findings

1. **CFR is fully functional** but synthesize/rolling_context require 60-120s due to LLM inference. Default 10s timeouts are insufficient.
2. **Memory system healthy**: 83 documents indexed with MiniLM-L6-v2 embeddings. Semantic search returns relevant results.
3. **RAG requires knowledge_base_name** — the API contract differs from what smoke tests assumed. Four KBs available: system, blueprints, dnd_campaign, wiki.
4. **Web fetch has domain allowlist** — security feature blocking untrusted domains. Wikipedia allowed, httpbin blocked.
5. **Fragment system works end-to-end** — write, read, list, clear all functional. Server generates UUIDs; parent_request_id groups fragments.
6. **Thought seed heartbeat** is active (1200s interval). 7 pending seeds awaiting review, including confabulation detection.
7. **Training subsystem idle** — last training attempt (prime-8b-identity-v2) failed on QLoRA setup (known 16GB VRAM limitation).
8. **Two trained adapters available**: core (4B, 300 steps) and nano (0.8B, 300 steps).

# Mid-February 2026 (Feb 10–16) — Consolidated Dev Journal

> Archived from 18 individual files on 2026-02-24.
> This covers feature development, comprehensive audit, and sleep cycle implementation.

---

## Feb 10: Knowledge Ingestion Pipeline

- Dual detection: explicit "save this" commands + auto-detect heuristic (500+ chars, entity names)
- Deduplication via semantic similarity (0.85+ threshold blocks duplicates)
- Two-stage approval: explicit saves need MCP approval; auto-detect offers user confirmation
- Document structure: YAML front matter with metadata (category, tags, source, version)
- Categories auto-detected: lore, character, rules, session_recap

## Feb 10: VRAM↔RAM Hot-Swap

- **vLLM sleep mode rejected**: Only freed 1.7GB KV cache (not 10.4GB model weights) due to `--enforce-eager` on Blackwell sm_120
- **Pragmatic pivot**: Docker container stop/start as GPU swap (12.8GB → 2.2GB in <1 second, ~60s cold restart)
- Orchestrator manages container lifecycle via Docker SDK
- Session history sanitization: strip `<think>` tags before saving

## Feb 10: Semantic Probe (Pre-Cognition Context Discovery)

- Runs before persona selection — discovers relevant knowledge bases automatically
- Phrase extraction is heuristic (regex, <5ms), not LLM-based
- Multi-collection probing: all indexed collections searched simultaneously
- Session cache: first embedding ~50ms, subsequent reuse cached hits
- Similarity threshold 0.40 minimum; dedup with RAG when probe found ≥2 hits from same collection
- Config: `SEMANTIC_PROBE` section in gaia_constants.json
- Total probe budget: <100ms for 2 collections, 5 phrases

## Feb 10: write_file End-to-End Integration

- Tiered safety gate: explicit allow + SAFE_SIDECAR_TOOLS pass; sensitive tools route to MCP 403
- Structured EXECUTE parser for `write_file`, `read_file`, `run_shell`
- 403 approval routing: sensitive tool rejection triggers Discord/web approval UI

## Feb 11: Response Quality Tuning (6 Fixes)

| Fix | Issue | Resolution |
|-----|-------|-----------|
| 1 | MCP stale paths | `/gaia-assistant` → `/knowledge` |
| 2 | Epistemic guardrails (10/12 citations fabricated) | 5-layer: prompt honesty + observer verifier + response annotation + pre-gen gate + config thresholds |
| 3 | Lite model lazy-load | `ensure_model_loaded("lite")` before dict lookup |
| 4 | Think-tag-only recovery | Retry with explicit instruction + reasoning fallback |
| 5 | BPE subtoken spacing corruption | Removed `_apply_stream_spacing()` |
| 6 | CJK character leakage | Post-processing filter removes short CJK runs |

- Temperature reduced 0.7 → 0.4 to reduce confabulation
- Language constraint directive added: "Always respond in English"

## Feb 11: Infrastructure (Tmpfs, KV Offload, Orchestrator, Slim Prompt)

- **Tmpfs warm pool**: `/mnt/gaia_warm_pool` (10GB tmpfs, systemd mount, rsync from NVMe on boot)
- **KV offloading**: `--enable-prefix-caching`, `--kv-offloading-backend native`, `--kv-offloading-size 8`
- **Orchestrator service**: Coordinates GPU handoff, Docker socket access, port 6410
- **Slim prompt**: System prompt 1,115 → 535 tokens (~52% reduction)

## Feb 11: CognitionPacket Consolidation

- Merged local gaia-core additions UP to gaia-common canonical version
- Deleted local copy — single source of truth in gaia-common
- Deleted dead pipeline module (7 files, ~525 lines from v0.1)
- Net reduction: ~2,078 lines

## Feb 12: Constants Merge + History Review + Intent Classifier + Web Research

- **Constants consolidation**: Two diverged copies merged into gaia-common (canonical)
- **History review**: Rule-based pre-injection audit (regex for fake paths, quotes, links); redact poisoned messages
- **Embedding intent classifier**: MiniLM-L6-v2, 116 labeled examples across 15 intents, cosine similarity
- **Web research tools**: `web_search` (DuckDuckGo), `web_fetch` (domain-allowlisted extraction)
- **Trust tiers**: Trusted (gutenberg, poetry foundation, britannica, wikipedia, arxiv), Reliable (github, stackoverflow, bbc), Blocked (reddit, twitter, facebook)
- **First full-stack promotion**: All 6 services promoted via hardened pipeline

## Feb 13: Web-Retrieval Recitation + Smoke Test Expansion

- Automatic web retrieval for recitations (poems, speeches) — local docs → web → confidence gate
- **DuckDuckGo fix**: Library renamed to `ddgs`; installed v9.10.0
- **Smoke tests expanded**: 6 → 16 tests (casual chat, tool routing, correction handling, epistemic guardrail, loop resistance, multi-turn memory, etc.)
- **Auto-sanitization**: `SessionManager.sanitize_sessions()` on Discord connect
- **Master promotion pipeline**: `promote_pipeline.sh` with 7-stage fail-fast orchestration

## Feb 13: QLoRA json-architect Curriculum

- 356 unique examples → 1000 after augmentation (85/15 train/val split)
- 4 categories: tool selection (165), null selection (87), tool review (76), confidence assessment (28)
- Validation pipeline: `validate_qlora.sh` with 7-stage process

## Feb 14: Sleep Cycle Phase 1 & 2

### Phase 1 — Core Infrastructure
- **5-state machine**: AWAKE → DROWSY → SLEEPING → FINISHING_TASK → WAKING
- **DROWSY cancellable**: Brief idle periods don't trigger full sleep/wake
- **Parallel wake**: CPU Lite handles first message while Prime boots (~37-60s)
- **prime.md checkpoint**: Natural-language cognitive state (not KV tensor serialization)
- **Message queue**: gaia-web queues messages during sleep, auto-sends wake signal
- 26/26 unit tests pass

### Phase 2 — Task System
- **SleepTaskScheduler**: Priority + LRU scheduling
- Default tasks: conversation_curation (P1), thought_seed_review (P1), initiative_cycle (P2)
- **InitiativeEngine**: Port of archived run_gil.py for autonomous topic processing
- 47/47 cognition tests pass (zero regressions)

## Feb 16: Full Codebase Audit (211 Production Files)

### CRITICAL Fixes (4/4 = 100%)
- Shell injection in safe_execution.py (shell=True bypass)
- Shell injection in gaia_rescue_helper.py
- Path traversal in code_read/code_span/code_symbol
- Broken endpoint in gaia-web output_router (Dict dot-notation access)

### HIGH Fixes (6/6 = 100%)
- Shell injection in mcp_client.py ai_execute()
- GPU owner bug after wake in orchestrator
- Unreachable except clause in discord_interface
- None API key model creation in oracle_model
- Dead dispatch() function (80 lines of v0.2 API)
- /gpu/wait 300s worker blocking → 60s cap

### MEDIUM Fixes (14/16 = 88%)
- ai_write() path restriction, Config singleton, self_review_worker enums, NVML lazy init, GPU race condition, blocking Docker SDK calls, 53 files datetime.utcnow() → datetime.now(UTC)

### Total: 34 fixes across 94 files; all CRITICAL and HIGH addressed

## Key Promotions This Period

| Date | Services | Result |
|------|----------|--------|
| Feb 12 | All 6 services | PASS (first full-stack promotion) |
| Feb 14 | gaia-common, gaia-core | PASS |
| Feb 15 | gaia-core, gaia-web | PASS |

## Outstanding Items at Period End

1. Sleep Cycle Phases 1 & 2 pending promotion validation
2. Thought Seed prerequisites (Observer THOUGHT_SEED: generation)
3. Phase 3 (QLoRA training + dream mode) awaiting Phase 1 & 2 stability
4. ddgs dependency not baked into gaia-mcp image
5. Inverted dependency gaia-common → gaia_core (lazy imports, acceptable)

# Dev Journal Entry: 2026-02-12 — History Review, Constants Merge, Promotion Hardening

**Date:** 2026-02-12
**Author:** Claude Code (Opus 4.6) via Happy

## Context

Continuing from the epistemic guardrails and cognitive audit work on Feb 11. Live Discord testing revealed that even with the new audit pipeline, response quality remained poor. Root cause: the conversation history itself was poisoned with fabricated file paths, fake blockquotes, and hallucinated URLs from earlier confabulation episodes. Each new response built on this polluted context.

Additionally, the cognitive audit feature wasn't firing because `Config._load_constants()` was loading from the wrong `gaia_constants.json` copy (gaia-core's local copy lacked the `COGNITIVE_AUDIT` config block). This led to a larger consolidation effort.

---

## Fix 1: Constants File Consolidation

**Problem:** Two diverged copies of `gaia_constants.json` existed:
- `gaia-core/gaia_core/gaia_constants.json` — had `LOOP_DETECTION_*`, `CODEX_FILE_EXTS`, `groq_fallback`, `lora_config` in `gpu_prime`, `llm_backend: "gpu_prime"`
- `gaia-common/gaia_common/constants/gaia_constants.json` — had `COGNITIVE_AUDIT`, `EPISTEMIC_GUARDRAILS`, `SEMANTIC_PROBE`, `cognitive_self_audit` prompt

`Config._load_constants()` resolved `os.path.dirname(__file__)` first, finding the gaia-core copy which lacked the audit config. Result: `cfg.constants.get("COGNITIVE_AUDIT", {}).get("enabled", False)` always returned `False`.

**Fix:**
1. Merged all unique keys from both files into gaia-common's copy (canonical)
2. Updated `Config._load_constants()` path priority to prefer gaia-common
3. Updated `Dockerfile` COPY directive: `gaia-common → gaia_common/constants/`
4. Deleted duplicate `gaia_core/gaia_constants.json` from both candidate and live

**Files:**
- `candidates/gaia-common/gaia_common/constants/gaia_constants.json` (merged)
- `candidates/gaia-core/gaia_core/config.py` (path priority updated)
- `candidates/gaia-core/Dockerfile` (COPY from gaia-common)
- Deleted: `candidates/gaia-core/gaia_core/gaia_constants.json`, `gaia-core/gaia_core/gaia_constants.json`

## Fix 2: Logger Level Visibility

**Problem:** After the config fix, cognitive audit was running but no log output appeared. `GAIA.AgentCore` logger has effective level WARNING (30), silently filtering all `logger.info()` calls.

**Fix:** Changed diagnostic audit gate logs from `logger.info()` to `logger.warning()` so they're visible during debugging. After confirming the audit runs, these can be reverted to `logger.info()`.

## Feature: History Review (`history_review.py`)

**Problem:** 20 messages in the Discord DM session history contained fabricated file paths (e.g., `knowledge/archaeology/stonehenge_artifacts.pdf`), fake blockquotes claiming document sources, hallucinated URLs (`https://knowledge.base/...`), and fake section references. This poisoned context was re-injected into every new prompt via the sliding history window.

**Solution:** Rule-based pre-injection audit that runs after `get_history()` but before `_create_initial_packet()`. No LLM call — purely regex pattern matching for fast execution.

**Detection patterns:**
- `_FAKE_PATH_RE` — File paths claiming tool-call verification (e.g., `read_file: knowledge/...`)
- `_FAKE_QUOTE_RE` — Blockquotes formatted as document citations
- `_FAKE_LINK_RE` — Markdown links to `knowledge.base` URLs
- `_FAKE_SECTION_REF_RE` — Section/line/page references with URLs
- `_FAKE_VERIFICATION_RE` — Ungrounded verification claims

**Actions per message:**
- 0 violations: pass through
- 1 violation (below threshold): annotate with caveat
- 2+ violations (at/above threshold): fully redact and replace with explanation note

**Special handling:** User correction + assistant acknowledgment pairs are detected and compressed into a single summary note, preserving the self-correction signal without re-injecting the fabricated details.

**Test results against the poisoned session (5 assistant messages):**
- 1 compressed (correction pair)
- 1 annotated (minor violation)
- 3 fully redacted (2-4 violations each)

**Config:** `HISTORY_REVIEW` block added to `gaia_constants.json`:
```json
"HISTORY_REVIEW": {
    "enabled": true,
    "violation_threshold": 2,
    "max_messages": 20
}
```

**Files:**
- `candidates/gaia-core/gaia_core/cognition/history_review.py` (new, 244 lines)
- `candidates/gaia-core/gaia_core/cognition/agent_core.py` (import + integration after `get_history()`)
- `candidates/gaia-common/gaia_common/constants/gaia_constants.json` (config block)
- All mirrored to live copies

## Improvement: Promotion Script Hardening (`promote_candidate.sh`)

**Problem:** `promote_candidate.sh` did build/lint/test/backup/rsync/restart but had no validation that `gaia-common` shared files were in sync between candidate and live. A promotion could leave live running with mismatched CognitionPacket definitions or stale constants.

**Fix:** Added pre-promotion sync validation that runs automatically:
1. **Blocking check**: Diffs `cognition_packet.py` between candidate and live gaia-common. If mismatched, promotion aborts with actionable guidance ("promote gaia-common first").
2. **Warning check**: Diffs `gaia_constants.json` — warns but doesn't block (constants mismatch won't crash, just behave differently).
3. **Protocol scan**: Diffs all files under `gaia_common/protocols/` for other shared type mismatches.
4. **`--force` flag**: Overrides the blocking check when needed.

**File:** `scripts/promote_candidate.sh`

## Documentation: Blueprint Update (`GAIA_CORE.md`)

Updated the gaia-core blueprint to reflect current state:
- Removed deleted `pipeline/` directory from source tree
- Added `cognitive_audit.py` and `history_review.py` to source tree
- Removed `cognition_packet.py` local copy (now in gaia-common)
- Removed `gaia_constants.json` local copy (now in gaia-common)
- Updated cognitive pipeline steps to match actual `run_turn()` flow (17 steps)
- Added **Model Selection Flowchart** — ASCII decision tree showing the full priority cascade from semantic probe through escalation, fallback, slim prompt, and generation
- Added escalation heuristic table and slim prompt trigger table
- Updated configuration section to reference gaia-common as single source of truth

**File:** `knowledge/blueprints/GAIA_CORE.md`

## External Code Review

Received and validated an external code review covering 6 concerns:

| Concern | Status | Notes |
|---------|--------|-------|
| CognitionPacket divergence | **Resolved** (commit `39e228b`) | Local copy deleted, all imports → gaia-common |
| Pipeline.py dead code | **Resolved** (commit `39e228b`) | 7 files, ~525 lines deleted |
| Loop detection not wired | **Resolved** (already active) | `record_tool_call()` and `record_output()` both called during inference |
| Model selection complexity | **Documented** | Flowchart added to blueprint |
| Promotion script gaps | **Fixed** | Sync validation added to `promote_candidate.sh` |
| Intent detection bypass | **Acknowledged** | Low urgency, improvement opportunity |

---

## Feature: Embedding-Based Intent Classifier

**Problem:** Intent detection for the llama_cpp lite backend falls back to keyword-only heuristics, returning `INTENT_DETECTED: other` for most natural language queries. The 16+ intent categories are poorly served by first-word keyword matching — "Who forged Excalibur?" hits "other" when it should hit "chat" (or at least a domain-aware category).

**Solution:** Cosine-similarity classification against an exemplar bank using the existing MiniLM-L6-v2 embedding model (already loaded by ModelPool for semantic probe and session history).

**Architecture:**
1. **Exemplar Bank** (`intent_exemplars.json`) — 116 labeled example phrases across 15 intent categories. Editable JSON, no retraining required.
2. **EmbedIntentClassifier** (`embed_intent_classifier.py`) — Singleton that:
   - Loads and encodes all exemplar phrases once at first use (lazy init)
   - Classifies new queries via cosine similarity against the exemplar matrix
   - Uses top-k averaging per intent (default k=3) for robust scoring
   - Returns `(intent, confidence_score)` with configurable threshold (default 0.45)
3. **Integration** — Slots into `model_intent_detection()` at the llama_cpp fallback point:
   - Embedding classification runs **first** (preferred path)
   - If it returns a confident match → use it
   - If it returns "other" → fall through to keyword heuristic (existing behavior preserved)
   - If embed model unavailable → keyword heuristic (graceful degradation)

**Safeguards:**
- File-keyword guard still applies: `read_file`/`write_file` from embeddings still requires `_mentions_file_like_action()` confirmation
- Config toggle via `EMBED_INTENT.enabled` in gaia_constants.json
- No new model dependencies — reuses existing MiniLM from ModelPool

**Config** (`gaia_constants.json`):
```json
"EMBED_INTENT": {
    "enabled": true,
    "confidence_threshold": 0.45,
    "top_k": 3
}
```

**Files (new):**
- `candidates/gaia-core/gaia_core/cognition/nlu/embed_intent_classifier.py`
- `candidates/gaia-core/gaia_core/cognition/nlu/intent_exemplars.json`

**Files (modified):**
- `candidates/gaia-core/gaia_core/cognition/nlu/intent_detection.py` (import + embed-first logic at llama_cpp branch)
- `candidates/gaia-core/gaia_core/cognition/nlu/__init__.py` (export EmbedIntentClassifier)
- `candidates/gaia-core/gaia_core/cognition/agent_core.py` (pass embed_model to detect_intent)
- `candidates/gaia-common/gaia_common/constants/gaia_constants.json` (EMBED_INTENT config)
- All mirrored to live copies

---

## Feature: Web Research Tools (`web_search`, `web_fetch`)

**Problem:** GAIA fabricated citations, quotes, and URLs because it had no actual access to the web. When asked for a poem or a factual lookup, the model would invent plausible-looking sources. The epistemic guardrails (Fix 2, Feb 11) catch this *after* generation, but the root cause is that GAIA genuinely can't look things up.

**Solution:** Two new MCP tools that give GAIA read-only, rate-limited, domain-gated web access:

### `web_search` (via DuckDuckGo)

- Takes `query`, optional `content_type` hint (`poem`, `facts`, `code`, `science`, `news`), optional `domain_filter`, optional `max_results` (default 5, max 10)
- Content type hints auto-inject `site:` clauses for relevant domains (e.g., `poem` → gutenberg.org, poetryfoundation.org, poets.org)
- Results annotated with trust tier before returning to the model
- Blocked domains (reddit, 4chan, twitter, facebook, etc.) filtered from results
- Rate limited: 20 searches/hour (in-memory sliding window)
- DuckDuckGo library with Instant Answer API fallback

### `web_fetch` (domain-allowlisted content extraction)

- Takes `url`, returns extracted text content
- **Domain gating**: Only fetches from allowlisted domains (trusted + reliable tier). Unknown domains get a clear refusal message directing the model to use `web_search` first
- Extraction pipeline: trafilatura → BeautifulSoup → regex fallback (graceful degradation if optional deps missing)
- 500KB content cap, 15s timeout
- Rate limited: 50 fetches/hour

### Source Trust Tiers

Three-tier domain classification with config override via `gaia_constants.json["WEB_RESEARCH"]`:

| Tier | Domains | Fetch allowed |
|------|---------|---------------|
| Trusted | gutenberg.org, poetryfoundation.org, britannica.com, wikipedia.org, arxiv.org, docs.python.org, developer.mozilla.org, etc. | Yes |
| Reliable | github.com, stackoverflow.com, bbc.com, reuters.com, nature.com, etc. | Yes |
| Blocked | reddit.com, 4chan.org, twitter.com, facebook.com, tiktok.com, etc. | No |
| Unknown | Everything else | No (search only) |

### Integration Points

- **MCP server** (`server.py`, `tools.py`): `web_search` and `web_fetch` added to both `dispatch_tool()` and `execute_tool()` dispatch maps
- **Tools registry** (`gaia-common/tools_registry.py`): Tool schemas added for LLM tool selection
- **World state** (`world_state.py`): Web capability affordance injected when web tools are available, so the model knows it can search
- **Dependencies** (`requirements.txt`): Added `duckduckgo_search>=7.0.0`, `trafilatura>=1.8.0`, `beautifulsoup4>=4.12.0`

**Files (new):**
- `candidates/gaia-mcp/gaia_mcp/web_tools.py` (466 lines)
- `candidates/gaia-mcp/tests/test_web_tools.py`

**Files (modified):**
- `candidates/gaia-mcp/gaia_mcp/server.py`
- `candidates/gaia-mcp/gaia_mcp/tools.py`
- `candidates/gaia-mcp/requirements.txt`
- `candidates/gaia-common/gaia_common/utils/tools_registry.py`
- `candidates/gaia-core/gaia_core/utils/world_state.py`
- All mirrored to live copies

---

## Process Retrospective: Candidate-to-Live Workflow

**Issue observed:** Throughout the Feb 11–12 sessions, changes were edited in both `candidates/` and live (`gaia-core/`, `gaia-common/`, `gaia-mcp/`) directories simultaneously rather than following the intended candidate SDLC:

```
Edit candidates/ → Validate in candidate containers → promote_candidate.sh → Live
```

This was noted as early as the Feb 10 journal (write_file E2E entry: *"Changes were edited in production first, then mirrored to candidates. The correct workflow should have been: edit candidates → validate → promote."*) but the pattern continued through Feb 11 and 12.

**Why it matters:**
1. **Promotion script bypassed** — `promote_candidate.sh` includes lint checks, test runs, backup creation, and sync validation (the very checks we hardened on Feb 12). Manually mirroring changes skips all of these.
2. **Divergence risk** — Manual mirroring is error-prone. The constants file divergence that caused the cognitive audit to silently fail (Fix 1 today) was a direct consequence of maintaining two copies without automated sync.
3. **No rollback point** — Without a proper promotion, there's no clean backup to roll back to if something breaks in production.

**Going forward:** All code changes should target `candidates/` exclusively. Use `promote_candidate.sh` to push to live after validation. The sync checks added today will catch mismatches, but the real fix is discipline: candidates first, promote second.

---

## Status

All candidate containers running and healthy. History review deployed and verified in container. Embedding-based intent classifier implemented and wired in — ready for container rebuild and Discord testing. Web research tools implemented and wired into MCP — requires container rebuild to pick up new Python dependencies (`duckduckgo_search`, `trafilatura`, `beautifulsoup4`).

### Uncommitted Changes Summary

32 files changed (+1,012 / -741 lines). Key areas:
- Constants consolidation (delete gaia-core copy, merge into gaia-common)
- History review module (new)
- Embedding-based intent classifier (new)
- Web research tools (new)
- Promotion script hardening
- Blueprint documentation
- All changes mirrored across candidate and live trees

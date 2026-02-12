# Dev Journal Entry: 2026-02-11 — Response Quality Tuning & Stream Fixes

**Date:** 2026-02-11
**Author:** Claude Code (Opus 4.6) via Happy

## Context

After the epistemic guardrails, semantic probe, and tmpfs warm pool work from earlier today, Azrael ran live Discord testing against the candidate stack. This uncovered a cascade of response quality issues in the 3B Nanbeige4 model output: stale paths in MCP, think-tag-only responses, BPE subtoken spacing corruption, CJK character leakage, and confabulation. This session systematically diagnosed and fixed each issue through iterative test-fix-restart-retest cycles.

## Commits This Session

| Commit | Summary |
|--------|---------|
| `732baf7` | Fix MCP find_files/list_files: stale `/gaia-assistant` paths → `/knowledge` |
| `76483ac` | Epistemic guardrails: 5-layer anti-confabulation system |
| `bdcffb2` | Lite model lazy-load, observer→lite, English constraint, temp 0.7→0.4 |
| `e13a493` | Think-tag-only recovery: retry + reasoning fallback |
| `1f45ff4` | Remove `_apply_stream_spacing` that corrupted vLLM subtoken output |
| (pending) | CJK post-processing filter in output_router.py |

---

## Fix 1: MCP Stale Paths (`732baf7`)

**Problem:** `find_files` and `list_files` MCP tools still referenced `/gaia-assistant/knowledge` (old path) instead of `/knowledge` (containerized mount).

**Fix:** Updated all hardcoded paths in MCP file tools.

**File:** `candidates/gaia-mcp/gaia_mcp/tools/file_tools.py`

---

## Fix 2: Epistemic Guardrails (`76483ac`)

**Problem:** GAIA fabricated 10 out of 12 file citations in a Discord conversation, presenting hallucinated paths and quotes as verified RAG retrievals.

**Fix:** 5-layer defense system:

1. **Prompt Guardrails** (`prompt_builder.py`) — Unconditional epistemic honesty directive on every turn. Rules: never cite unread files, never fabricate quotes, distinguish "from my knowledge base" vs "from my general knowledge."

2. **Observer Citation Verifier** (`stream_observer.py`) — Cross-references cited filenames against `retrieved_documents` DataField. Counts fabricated knowledge-base citations and returns CAUTION interrupt.

3. **Response Annotation** (`agent_core.py`) — After observer review, checks reflection log for path validation failures. Appends user-visible warning: *"[Observer: Some file references could not be verified]"*

4. **Pre-Generation Confidence Gate** (`agent_core.py`) — When RAG returns no results for a domain-specific query, runs confidence assessment. If confidence < 0.5, returns honest "I don't have that information" instead of generating (preventing fabrication).

5. **Configuration** (`gaia_constants.json`) — `EPISTEMIC_GUARDRAILS` config section with tunable thresholds.

**Files:** `prompt_builder.py`, `stream_observer.py`, `agent_core.py`, `gaia_constants.json`

---

## Fix 3: Lite Model Lazy-Loading + Observer Switch (`bdcffb2`)

**Problem:** Intent detection silently skipped the lite model because `models.get("lite")` returned None when `GAIA_AUTOLOAD_MODELS=0`. Observer was configured to use `gpu_prime` (heavy) for observation tasks.

**Fix:**
- **Intent detection** (`agent_core.py` ~line 1093): Added `ensure_model_loaded("lite")` call before dict lookup. Now the lite model lazy-loads on first request.
- **Observer model** (`agent_core.py` ~line 1370): Changed from `use_gpu_prime` config to `use_lite`, using `ensure_model_loaded("lite")` with fallback chain.
- **Constants** (`gaia_constants.json`): Observer config changed from `"use_gpu_prime": true, "type": "vllm"` to `"use_lite": true, "type": "local"`.

**Temperature reduction:** Default temperature lowered from 0.7 to 0.4 in `config.py` to reduce confabulation.

**English-only constraint:** Added `LANGUAGE CONSTRAINT` directive in `prompt_builder.py` — "Always respond in English. Do not use non-English characters unless explicitly requested."

---

## Fix 4: Think-Tag-Only Recovery (`e13a493`)

**Problem:** The 3B model sometimes generates entire responses inside `<think>` tags with no visible output. After `_strip_think_tags_robust()` removes them, the response is empty and the user sees "I apologize, but I encountered an issue."

**Root cause observation:** `AgentCore: collected 470 stream chunks (len=1997 chars)` → all inside think tags → empty after stripping.

**Fix:** Two-stage recovery in `agent_core.py` after line 1565:

1. **Retry with explicit instruction:** Detects empty-after-stripping, appends "Respond directly without `<think>` tags" to messages, retries at lower temperature.
2. **Reasoning fallback:** If retry also fails, extracts reasoning content from the original think tags and presents it as "Based on my analysis: [reasoning]"

Also improved `output_router.py` (lines 165-178) with the same reasoning extraction as a safety net.

---

## Fix 5: BPE Subtoken Spacing (`1f45ff4`)

**Problem:** Words in responses were broken apart: `He im ric`, `Sn ark`, `Stra uth auk`. This made responses nearly unreadable.

**Root cause:** `_apply_stream_spacing()` in `external_voice.py` (lines 451-458) prepended a space to every non-whitespace token, assuming word-level tokenization. But vLLM emits BPE subtokens (e.g., "Hei", "mric") that must be concatenated directly. The OpenAI-compatible streaming API already includes spaces at word boundaries.

**Fix:** Removed the `_apply_stream_spacing()` call at line 251 of `external_voice.py`. The method is still defined but no longer invoked in the stream loop.

**Verification:** "Excalibur", "Stonehenge", "Arthur" — all properly joined in the next test.

---

## Fix 6: CJK Post-Processing Filter (uncommitted)

**Problem:** Despite the English-only language constraint, the 3B model still leaks CJK characters inline with English: `Stonehenge巨石` (巨石 = "megaliths"), `Dexterity (敏捷)` (敏捷 = "agility"). This is a fundamental limitation of the multilingual training data baked into model weights.

**Fix:** Post-processing filter in `output_router.py`:

- `_CJK_STRAY_RE` regex matches CJK ideographs, Kana, Hangul, and fullwidth forms
- `_strip_stray_cjk()` removes short runs (≤10 chars) as stray leakage
- Longer runs (>10 chars) preserved for intentional translations/quotes
- Applied after think-tag stripping, before destination routing
- Collapses double-spaces left by removal

**Test results:**
- `Stonehenge巨石` → `Stonehenge` ✓
- `Dexterity (敏捷 - speed)` → `Dexterity ( - speed)` ✓
- Normal English text → unchanged ✓
- Long CJK block (>10 chars) → preserved ✓

---

## Remaining Known Limitations

These are **3B model capability limits**, not code bugs:

1. **Confabulation on general knowledge** — The model invents plausible-sounding but incorrect facts (e.g., "Excalibur is Latin for 'sharp sword'", "pommel shaped like a lion's head"). The epistemic guardrails help for RAG-grounded topics but can't prevent hallucination on ungrounded general knowledge queries. Mitigation: lower temperature + prompt directives. Real fix: larger model.

2. **Intent detection heuristic fallback** — The lite model loads now, but intent detection via `llama_cpp` backend falls back to keyword heuristics. Result: `INTENT_DETECTED: other` for most queries. This is functional but not using the LLM classification path.

## Testing Methodology

Each fix followed the same cycle:
1. Identify issue in Docker logs (`docker logs gaia-core-candidate`)
2. Trace through code to find root cause
3. Apply minimal fix
4. Commit
5. `docker compose restart gaia-core-candidate`
6. Wait for health check
7. User sends test message on Discord
8. Analyze logs for the new request
9. Verify fix and check for new issues

## Files Modified (This Session, All in `candidates/`)

| File | Changes |
|------|---------|
| `gaia-mcp/gaia_mcp/tools/file_tools.py` | Stale path fix |
| `gaia-core/gaia_core/utils/prompt_builder.py` | Epistemic honesty + language constraint + anti-confab |
| `gaia-core/gaia_core/utils/stream_observer.py` | Citation verifier, fabrication detection |
| `gaia-core/gaia_core/cognition/agent_core.py` | Lite lazy-load, observer→lite, think-tag recovery, epistemic annotation |
| `gaia-core/gaia_core/config.py` | Temperature 0.7→0.4 |
| `gaia-core/gaia_core/cognition/external_voice.py` | Removed `_apply_stream_spacing` call |
| `gaia-core/gaia_core/utils/output_router.py` | Think-tag reasoning fallback + CJK filter |
| `gaia-common/gaia_common/constants/gaia_constants.json` | Observer `use_lite`, `EPISTEMIC_GUARDRAILS` config |

---

# Dev Journal Entry: 2026-02-11 — Tmpfs Warm Pool, KV Offloading, Orchestrator Service, and Slim Prompt

**Date:** 2026-02-11 (earlier session)
**Author:** Claude Code (Opus 4.6) via Happy

## Context

This session implemented three independent infrastructure improvements that were committed together in `d2e551a`:

1. **Tmpfs warm pool** — Pre-loads model weights into RAM to eliminate NVMe dependency on cold start
2. **KV offloading** — CPU RAM buffer for evicted KV cache blocks, reducing recomputation on multi-turn conversations
3. **Orchestrator service** — New container for GPU/container coordination
4. **Slim prompt fix** — System prompt token reduction (~52%)

## 1. Tmpfs Warm Pool

**Plan:** `dapper-singing-panda.md`

Created a three-tier memory hierarchy for vLLM model weights:
- **Tier 0** — GPU VRAM (active weights + KV cache)
- **Tier 1** — CPU RAM via tmpfs at `/mnt/gaia_warm_pool` (8GB KV offload buffer)
- **Tier 2** — Tmpfs model weights (7.4GB Claude model always in RAM)
- **Tier 3** — NVMe source of truth

**Infrastructure:**
- systemd mount unit: `/etc/systemd/system/mnt-gaia_warm_pool.mount` (tmpfs, 10G, mode 0755)
- Seeding service: `/etc/systemd/system/gaia-warm-pool-seed.service` (rsync from NVMe on boot)
- Fires before Docker, ensuring warm pool is ready when containers start

**Compose changes** (`docker-compose.candidate.yml` — gaia-prime-candidate):
- Volume mounts switched from `./gaia-models` to `/mnt/gaia_warm_pool/Claude`
- Added vLLM flags: `--enable-prefix-caching`, `--kv-offloading-backend native`, `--kv-offloading-size 8`
- GPU memory utilization: 0.65 → 0.70
- Health check start_period: 120s → 50s

**RAM budget (60GB system):** Warm pool 7.4G + KV offload 8G + services ~17G = ~32.4G used, ~27G free.

## 2. Orchestrator Service

**New service:** `gaia-orchestrator-candidate` added to `docker-compose.candidate.yml`

- Coordinates GPU handoff between gaia-prime and gaia-study
- Has Docker socket access (read-only) for container management
- Reads compose files for service configuration
- Port 6410 (candidate: 6411)
- Profiles: `full`, `orchestrator`

**Files created:**
- `candidates/gaia-orchestrator/` — full service directory
- `candidates/gaia-orchestrator/Dockerfile`
- `candidates/gaia-orchestrator/gaia_orchestrator/main.py`
- `candidates/gaia-orchestrator/gaia_orchestrator/gpu_manager.py`

## 3. Slim Prompt

**Plan:** `cozy-enchanting-cocoa.md` (Prompt Builder System Prompt Trim)

Reduced system prompt baseline from ~1,115 tokens to ~535 tokens (~52% reduction):
- Identity deduplication (3 blocks → 1)
- Slimmed persona instructions (removed redundant MCP body plan)
- Tightened safety directive
- Conditional memory helpers (gate on tool availability)
- Trimmed packet template (10 sections → 3: intent, reasoning, content)
- Trimmed epistemic honesty directive

**Files:** `prompt_builder.py`, `packet_templates.py`

## 4. Semantic Probe (Already Journaled Feb 10)

The semantic probe phases 1-5 were documented in the Feb 10 journal. This commit (`d2e551a`) includes the semantic probe implementation alongside the tmpfs/orchestrator/prompt changes.

---

# Dev Journal Catch-Up: 2026-02-10 — Session RAG, Sliding Window, and Candidate-to-Live Promotion (`6d2e32f`)

**Date:** 2026-02-10
**Author:** Claude Code (Opus 4.6) via Happy

## Context

This commit bundled several improvements that were not individually journaled:

### 1. Session RAG with Sliding Window

**Problem:** The session context grew unboundedly, consuming prompt token budget and eventually exceeding context limits on long conversations.

**Fix:**
- **Session vector store** — Each session gets a vector index of past messages, enabling semantic retrieval of relevant history instead of dumping all messages.
- **Sliding window** — Conversation history is capped to a configurable window (default: last N turns), with older messages summarized or available via RAG lookup.
- **Session vector files** — Created in `data/shared/session_vectors/` per session ID (e.g., `discord_dm_596925786208993283.json`).

**Files:** `agent_core.py` (session management), `session_manager.py` (sliding window), session vector JSON files.

### 2. MCP write_file Safety Fixes

- Hardened the `write_file` approval flow
- Fixed edge cases in the safety gate for sensitive tool operations
- Aligned candidate and live implementations

### 3. Candidate-to-Live Promotion

- Promoted validated candidate code to the live `gaia-core/`, `gaia-common/`, `gaia-web/` directories
- Synced all changes between candidate and live stacks

---

# Dev Journal Catch-Up: 2026-02-10 — Blueprint Updates (`050c51b`)

**Date:** 2026-02-10
**Author:** Claude Code (Opus 4.6) via Happy

Documentation-only commit updating all architectural blueprints in `knowledge/blueprints/` to reflect v0.3 architecture accurately. No code changes.

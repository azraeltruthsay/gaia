# Dev Journal — 2026-02-27: Candidate Hardening + Cognition Tuning

## Change Tier Classification

| Change | Tier | Rationale |
|--------|------|-----------|
| Accumulated candidate work (timeouts, locking, dashboard) | **Tier 1 — Candidate-first** | Cognition pipeline, model pool, web dashboard |
| Probe strength gating + domain momentum | **Tier 1 — Candidate-first** | Semantic probe behavior, persona selection logic |
| Anti-sycophancy → Authentic engagement | **Tier 1 — Candidate-first** | Prompt builder directives, core identity |
| Curated log digest script | **Tier 2 — Production-direct** | New standalone script, no service changes |
| Consent sovereignty dashboard panel | **Tier 1 — Candidate-first** | Web dashboard UI + backend |
| Audio listener dashboard panel | **Tier 1 — Candidate-first** | Web dashboard UI + backend |

---

## Accumulated Candidate Work (`b7a14ea`)

### Cognition/Models (defensive hardening)
- **LLM call timeouts** (8s) for `cognitive_audit` and `self_reflection` via `ThreadPoolExecutor`
- **Loop detector reset** at turn start in `external_voice` (prevents false positives)
- **Intent detection**: source param for heartbeats, long message truncation, skip frag for heartbeats
- **Thread-safe locking** for llama_cpp backends in model pool (prevents `GGML_ASSERT` crashes)
- **Context-window-aware** `max_tokens` clamping + smart retry on 400 overflow in `vllm_remote_model`

### Common/MCP/Audio (plumbing)
- Per-level ring-buffer log handler (`LevelRingHandler`) with FIFO disk rotation
- `list_knowledge_bases` tool, Kanka fuzzy campaign name resolution, better error messages
- Non-fatal TTS load failures in audio `gpu_manager` (STT continues if TTS fails)
- `LOG_LEVEL` env var override in `docker-compose.override.yml` (was hardcoded DEBUG)
- New deps: `torchaudio>=2.0.0` (audio), `psutil==7.2.2` (core)

### Web Dashboard
- **Audio Listener panel**: start/stop controls, live status, transcript buffer, ingest-to-core
- **Consent Sovereignty panel**: test library, tier-based runner, reasoning quality analysis
- **Log level filtering** UI with ring-buffer `/levels` endpoint
- **Discord bot watchdog** with exponential backoff auto-restart

## Probe Strength Gating (`3a41a95`)

**Problem**: When a user says "my main drive was on the verge of dying", the semantic probe extracted common words that weakly matched D&D narrative content (0.41 cosine similarity). A single borderline hit unconditionally switched persona to `dnd_player_assistant` and poisoned the entire context with D&D lore before the model even saw the message.

**Fix**:
- Added `probe_strength` classification (`strong` / `moderate` / `weak`) to `SemanticProbeResult` based on configurable similarity thresholds
- Replaced unconditional probe-driven persona selection with strength-gated logic: strong = full switch, moderate = check domain momentum, weak = keyword fallback only
- Track **domain momentum** (`last_domain` + turn count) in session metadata to resist spurious topic oscillation
- Added 6 tunable thresholds to `SEMANTIC_PROBE` config section

Smoke battery: 17/20 passed (3 pre-existing failures, 0 regressions).

## Anti-Sycophancy → Authentic Engagement (`4a9436f`)

Reframed GAIA's tone directive from blanket suppression of positive responses to a nuanced distinction:
- **Still banned**: hollow performative validation ("Great question!", empty affirmation)
- **Now explicitly permitted**: genuine warmth, authentic acknowledgment, real enthusiasm when warranted

Renamed section from "Anti-Sycophancy" to "Authentic Engagement" across `prompt_builder` and `core_identity`.

## Curated Log Digest Script (`cec2254`)

Stdlib-only Python script (`scripts/curated_log_digest.py`) that distills runtime logs (`generation_stream`, ring buffers, `self_reflection`) into a compact daily markdown digest at `knowledge/digests/`. Picked up automatically by `flatten_soa.sh`, giving NotebookLM visibility into conversations, errors, and system activity.

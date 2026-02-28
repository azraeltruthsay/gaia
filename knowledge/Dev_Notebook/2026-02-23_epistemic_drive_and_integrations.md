# Dev Journal — 2026-02-23: Epistemic Drive + External Integrations

## Change Tier Classification

| Change | Tier | Rationale |
|--------|------|-----------|
| Epistemic drive (prompt directives, seed routing, confidence tiers) | **Tier 1 — Candidate-first** | Modifies cognition pipeline, prompt builder, vector indexer |
| Knowledge research sleep task | **Tier 1 — Candidate-first** | New sleep task that uses MCP tools autonomously |
| Kanka.io MCP tools | **Tier 1 — Candidate-first** | New MCP tools with external API calls + rate limiting |
| DM blocklist | **Tier 1 — Candidate-first** | New routes + Discord interface changes |
| Conversation examples, knowledge cleanup | **Tier 2 — Production-direct** | Documentation/knowledge files only |
| Smoke test expansion | **Tier 2 — Production-direct** | Test infrastructure |

---

## Epistemic Drive — Knowledge-Seeking Behavior (`a11eba9`)

4-phase implementation making GAIA epistemically grounded:

1. **Behavioral directives**: Tier 3.55 prompt directive with anti-sycophancy rules, calibrated confidence communication, and genuine curiosity. Updated thought seed directive to prioritize knowledge gaps. Replaced sycophantic conversation examples with epistemically grounded exemplars.

2. **Knowledge gap tracking**: `seed_type` field in thought seeds detects gap markers. Heartbeat auto-routes `knowledge_gap` seeds to ACT without LLM triage. `EPISTEMIC_DRIVE` config block in `gaia_constants.json`.

3. **Research-during-sleep**: `knowledge_research` sleep task uses MCP web tools to autonomously research gaps, save to `/knowledge/research/`, and index with `confidence_tier` tagging.

4. **Confidence tiers**: 6-tier taxonomy (`core` > `verified` > `experiential` > `curated` > `researched` > `training`) threaded through `vector_indexer`, `semantic_probe` ProbeHit, and prompt builder. Retrieved knowledge now shows `[Verified Knowledge]`, `[Curated Reference]`, etc. tags in semantic context.

## Kanka.io MCP Tools (`e1d911d`)

6 new MCP tools for structured access to Kanka.io world-building data:
- **Read**: `kanka_list_campaigns`, `kanka_search`, `kanka_list_entities`, `kanka_get_entity` (with `?related=1` support)
- **Write** (approval-gated): `kanka_create_entity`, `kanka_update_entity`

KankaClient with 25 req/min client-side rate limit (below 30 hard cap), TTL cache (2–10 min by operation type), and cache invalidation on writes. 29 unit tests passing. Live-verified against all 3 campaigns.

## Accumulated Infrastructure (`18c7d7b`)

- **DM blocklist**: `dm_blocklist.py` + `routes/discord.py` + data config for managing Discord DM block/allow lists
- **Smoke tests**: expanded cognitive smoke test battery
- **Discord interface**: extended `discord_interface.py` with DM handling, voice whitelist update
- **Knowledge**: condensed `conversation_examples.md`, removed `artificial_consciousness.md` research doc and stale vector index
- **Minor tweaks**: `packet_utils`, `sleep_cycle_loop`, config, `prompt_builder`, promote_pipeline script

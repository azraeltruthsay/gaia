# Blueprints — index

Design docs for GAIA. The cluster below is the **cognitive architecture** (the "developing mind");
other files in this directory are per-service or per-feature blueprints.

## The developing mind (read in this order)

1. **[COGNITIVE_ARCHITECTURE.md](COGNITIVE_ARCHITECTURE.md)** — *the umbrella premise.* GAIA as a mind
   that *develops*: every cognitive organ (reflexive/CFR/Samvega/affect/Observer/Council/ThoughtSeed/
   mempalace/temporal) mapped into one nervous system, the design principles (two gates, affect as
   capacity-not-content, developmental sequencing), and the **audited** fire/dormant status of each. Start here.
2. **[loop_audit.md](loop_audit.md)** — verified (live-probed) state of every organ: what fires, what's
   integrated vs. parallel, what's broken. The audit corrected the code-trace three times.
3. **[integration_plan.md](integration_plan.md)** — the sequenced wiring plan (Tracks A/B/C). Records
   what's done and what's next.

### Per-system

- **[cfr_conversation_virtual_memory.md](cfr_conversation_virtual_memory.md)** — gate 1
  (resident-for-reasoning): conversation context as virtual memory; relevance×decay foveation; the
  two-gate axis (§1a, §8). Phase 1 live.
- **[affect_appraisal_layer.md](affect_appraisal_layer.md)** — affect as *capacity, not content*: an
  appraiser feeds functional drives/foci from real subsystems; `feels` (the emotion-words) are hers,
  named at report time or self-coined in sleep. P0 built.
- **[affect_model.md](affect_model.md)** — the earlier affect-runtime data model (KG triples).
- **[sae_atlas_build_plan.md](sae_atlas_build_plan.md)** — the *measurement instrument*: SAE feature
  atlases for both models, both paths (safetensors GPU + GGUF CPU), to verify cognitive signals map to
  real features. Bridge + corpus built; training is GPU-gated.
- **[brain_region_map.md](brain_region_map.md)** — the 13-region brain map (to be re-derived from the
  new atlases).

## Toggles (cognitive systems, set in docker-compose gaia-core)

| Flag | Default | What |
|------|---------|------|
| `CFR_CONVERSATION_ENABLED` | 1 | relevance-scored working set (gate 1) |
| `CFR_BLUR_BREADCRUMB` | 0 | OFF — in-prompt blur breadcrumb backfires on Gemma4-E4B |
| `VOICE_GATE_ENABLED` | 1 | post-generation meta-commentary strip (worth-voicing) |
| `OBSERVER_ON_STREAM` | 1 | run the conscience on the `/process_packet` user path (fast mode) |
| `AFFECT_APPRAISAL_ENABLED` | 0→soak | feed the affect organ from real subsystems |
| `THOUGHT_SEED_MAX_PENDING` | 2000 | seed-backlog cap (drain the landfill) |

*Note: the other `*.md` here (GAIA_CORE, GAIA_PRIME, gaia-mcp, sleep cycle, etc.) are service/feature
blueprints, not part of the cognitive-mind cluster.*

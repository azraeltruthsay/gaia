# Decision: prime.md Over KV Cache Persistence

**Status:** Active
**Date:** 2026-02

## Context

When gaia-prime goes to sleep (GPU freed for gaia-study), the model's KV cache is lost. On wake, the model has no memory of prior conversations. We need a way to preserve cognitive context across sleep/wake cycles.

## Decision

**Write a natural-language checkpoint (`prime.md`) instead of serializing the KV cache.**

Before sleep, gaia-core asks the Lite model to introspect on the current cognitive state and write a 2-4 paragraph summary covering:

- What was being discussed
- Unresolved threads or pending tasks
- Emotional tone and relationship context
- What to do first on waking

This checkpoint is injected into the system prompt on wake, giving the model context to resume naturally.

## Rationale

1. **vLLM doesn't support KV cache serialization** across container restarts. The cache lives in GPU memory and is tied to the specific model instance.
2. **KV cache is opaque** — even if we could save it, we couldn't inspect, edit, or reason about its contents. A natural-language checkpoint is human-readable and debuggable.
3. **The checkpoint is composable** — it can be combined with session history, persona traits, and temporal context in the prompt builder. A KV cache is all-or-nothing.
4. **Sleep mode with CPU offload** (`--kv-offloading-backend native`) preserves some cache across short sleeps. `prime.md` handles the longer gaps where even offloaded cache is stale.

## Implementation

- `PrimeCheckpointManager` (`gaia_core/cognition/prime_checkpoint.py`)
- Written on: sleep entry, graceful shutdown, manual `POST /cognition/checkpoint`
- Rotated: `prime.md` → `prime_previous.md` → `prime_history/<timestamp>-sleep.md`
- Consumed: prompt builder injects once on wake, marks as consumed

## Consequences

- Cognitive continuity degrades gracefully — even a stale checkpoint is better than no context
- The checkpoint quality depends on the Lite model's introspection ability
- Falls back to a static template when no Lite model is available

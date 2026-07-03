# gaia-nano — The Reflex (DEPRECATED)

**Port:** 8090 (host) → 8080 (container) | **GPU:** No | **Image:** `alpine/socat`

> **Deprecated in the Sovereign Duality era.** The E2B Reflex tier was removed — Core
> handles all triage directly. The `gaia-nano` container now exists only as a **socat
> passthrough** that forwards `:8080` to Core's embedded engine at `gaia-core:8092`,
> preserving the `gaia-nano` DNS name so existing callers keep working. To restore a
> real Nano tier, replace the proxy with the original engine container (see the note in
> `docker-compose.yml`).

## Current Container

```yaml
gaia-nano:
  image: alpine/socat:latest
  command: TCP-LISTEN:8080,fork,reuseaddr TCP:gaia-core:8092
```

Requests to `gaia-nano:8080` (e.g. `/v1/chat/completions`, `/slots`, `/health`)
transparently hit Core's embedded GAIA Engine instance.

## Historical Role (pre-deprecation)

gaia-nano ran a tiny 0.8B Qwen model via GAIA Engine managed mode for sub-second triage
classification and transcript cleanup. It was the first stage of the old three-tier
cascade:

1. **Nano** (0.8B) — classified input as SIMPLE or COMPLEX in <100ms
2. **Core** — handled SIMPLE tasks, escalated COMPLEX to Prime
3. **Prime** (8B) — heavyweight reasoning, code, complex tasks

In Sovereign Duality this collapsed to two tiers: **Core** (Gemma4-E4B) does its own
triage; **Prime** (Qwen3-VL-8B) handles deep reasoning.

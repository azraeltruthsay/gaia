# Decision: tmpfs Warm Pool for Model Swap

**Status:** Active
**Date:** 2026-02

## Context

GAIA needs to load different models for different roles (prime, lite, personality adapters). Model files are large (2-8 GB). Loading from disk (even NVMe) takes 10-30 seconds. We need sub-second model availability for LoRA adapter swaps.

## Decision

**Keep model files on a tmpfs-backed warm pool (`/mnt/gaia_warm_pool`)** so they're always in RAM. vLLM loads from this pool, making initial load and adapter swaps near-instant.

## Rationale

1. **tmpfs is memory-backed** — reads are memory-speed, not disk-speed
2. **vLLM's `--enable-lora`** allows hot-swapping LoRA adapters without reloading the base model. The base model stays in GPU memory; only the adapter weights need to be transferred.
3. **The warm pool persists across container restarts** (Docker bind mount to host tmpfs). Only a host reboot clears it.

## Trade-offs

- **Memory cost:** The warm pool consumes host RAM proportional to model size (~4-8 GB). On a 128 GB system this is acceptable.
- **Volatility:** tmpfs is lost on host reboot. Models must be re-copied from persistent storage. An init script handles this.
- **Single GPU:** Only one model can be active in GPU memory at a time. The warm pool reduces the penalty for switching.

## Configuration

```yaml
# docker-compose.yml
gaia-prime:
  volumes:
    - /mnt/gaia_warm_pool:/models:ro
```

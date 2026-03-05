# Dev Journal: Workflow Lessons & System Foundations (MRI 2.0)
**Date:** 2026-03-04
**Era:** Sovereign Autonomy
**Topic:** Cognitive Save Point & Architecture Baseline

## Overview
This entry serves as the foundational baseline for the GAIA Speculative Pipeline and Immune System hardening. It documents the critical lessons learned during the recovery of the real-time streaming stack and provides the necessary context for future sessions to maintain systemic integrity.

## 1. Multi-Phase Streaming Architecture (The "Flush" Protocol)
We implemented a high-performance, asynchronous NDJSON streaming pipeline across the entire mesh.
- **Speculative Reflex (Nano)**: A 0.5B model provides a "reflex" response in < 0.25s for cold prompts.
- **Refinement Cycle (Prime)**: A deeper 8B model (GPU-accelerated) performs the full 20-stage reasoning in parallel.
- **The `flush` Event**: `AgentCore.run_turn` now yields a `{"type": "flush"}` event after each model completes its stream. This signals front-ends (like Discord) to send their accumulated token buffers immediately rather than waiting for the entire turn to finish.
- **Clean Discord UX**: `discord_interface.py` was refactored to treat Reflexes and Refinements as sequential messages, eliminating double-posting and word-by-word rate-limit noise.

## 2. Hardened Digital Immune System (MRI 2.0)
The "Doc" (gaia-doctor) and the internal `ImmuneSystem` are now "fatality-aware."
- **Severity-Aware Scoring**: Differentiates between minor linting noise (weight 0.5) and fatal defects like missing imports (`F821`) or syntax errors (`E999`) (weight 10-15).
- **Fast-Failure Audit**: The Doctor now executes a targeted `ruff` check *before* the slower `pytest` suite during code change audits.
- **Container Robustness**: `docker_restart` in the Doctor was upgraded to use `docker ps --filter` to find correct container IDs, handling non-standard names like `2a85f751fcd3_gaia-web`.
- **Dissonance Probe**: Module-level SHA-256 hash comparison is active to detect "split-brain" drift between Live and Candidate minds.

## 3. Workflow Lessons for Future Sessions
- **Steady-State Verification**: ALWAYS wait 5 minutes after a success to verify that background `StatReloads` or "auto-heals" haven't introduced silent regressions.
- **Context Pruning**: `flatten_soa.sh` must be configured to exclude `MagicMock` files and journals older than the current month to maintain a high-signal NotebookLM context.
- **HA Alignment**: Any change to a "Vital Organ" (`main.py`, `agent_core.py`, `mcp_client.py`, `tools.py`, `immune_system.py`) MUST be bit-for-bit identical across `live/` and `candidates/`.
- **Local Health Polling**: The Doctor should prefer `localhost:[PORT]` for health checks to avoid hostname resolution failures during container renames.

## 4. Current System State
- **Core**: `healthy`, eager-loading active, 16GB memory ceiling.
- **Web**: `healthy`, sequential streaming active.
- **Prime**: `healthy`, GPU acceleration restored.
- **Parity**: 100% verified.

This baseline ensures that GAIA remains fast, self-aware, and structurally synchronized across all future iterations.

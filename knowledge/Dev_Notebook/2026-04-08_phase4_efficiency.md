# 2026-04-08 — Design: Phase 4 — Cognitive Efficiency & Interactive Speed

## Context
Current `run_turn` overhead is ~60s, which is too slow for fluid interaction. High latency is caused by large redundant system prompts, repeated full-packet transmissions, and model pool staleness causing ReadTimeout errors.

## Objective
Reduce `run_turn` latency to < 10s for cached turns and eliminate timeout-induced overhead.

## The Strategy

### 1. Header Prefix Caching (Physical Layer)
*   **Mechanic**: Leverage the `llama.cpp` prefix cache in `gaia_cpp`. 
*   **Implementation**: Identify the static "Cognitive Header" (Identity + Rules + Tools). Ensure `AgentCore` sends this block first and the engine maintains the KV cache for this specific hash.
*   **Target**: Reduce time-to-first-token by 80% for repeated turns.

### 2. Differential Packets (v0.5 Schema)
*   **Mechanic**: Instead of sending the full `CognitionPacket` (which grows with history), implement a "Differential" mode.
*   **Implementation**: 
    *   `gaia-web` sends `PacketDelta` (New user input + session ID).
    *   `gaia-core` retrieves the full state from its local session cache.
    *   Only return the `Reasoning` and `Response` deltas to the UI.
*   **Target**: Reduce payload size by 90% for deep conversations.

### 3. Model Pool Hot-Refresh
*   **Mechanic**: Fix the `gpu_prime` ReadTimeout by pro-actively refreshing the model pool.
*   **Implementation**: 
    *   Add `/refresh_pool` endpoint to `gaia-core`.
    *   `ConsciousnessMatrix` triggers this endpoint immediately after a successful tier transition (e.g., FOCUSING -> AWAKE).
    *   `AgentCore` updates its internal `_model_clients` map to remove entries for tiers that moved from GPU to CPU.

### 4. Interactive Quality Gate (IQG)
*   **Mechanic**: Implement a "Fast Path" for simple intents.
*   **Implementation**: If intent is `greeting` or `identity`, skip the full multi-stage reflection loop and generate immediately using the `lite` tier.

## Implementation Tasks (Action for Claude)
1.  **Orchestrator Hook**: Implement the `/refresh_pool` call in `ConsciousnessMatrix._sync_lifecycle`.
2.  **Core Update**: Add the `refresh_pool` endpoint to `main.py` and the corresponding method in `ModelPool`.
3.  **Prefix Optimization**: Refactor `prompt_builder.py` to ensure the static header is at the absolute top of the prompt to maximize KV cache hits.

## Strategic Impact
Phase 4 transforms GAIA from a "batch processor" to a "responsive partner." By optimizing the physical and cognitive layers, we enable the high-frequency interaction required for the Penpal Podcast and deep research sessions.

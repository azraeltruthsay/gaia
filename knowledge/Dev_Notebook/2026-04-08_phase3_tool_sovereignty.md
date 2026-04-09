# 2026-04-08 — Design: Phase 3 — Native Tool Sovereignty

## Context
Current tool usage relies on a rigid 3-step pipeline (Routing -> Execution -> Review) which is slow and prevents multi-turn reasoning within a single response. To unlock GAIA's full agentic potential, we need to transition to **Native Tool Calling** where models emit inline tags.

## Objective
Train the **9B-Prime** and **4B-Core** models to emit `<tool_call>{"tool": "...", "action": "...", ...}</tool_call>` tags and update `AgentCore` to handle these as high-priority stream interrupts.

## The Architecture
1.  **Inline Emission**: Model emits tool call tags during generation.
2.  **Stream Interception**: `AgentCore` identifies the `</tool_call>` stop sequence, extracts the JSON, and executes the tool via MCP.
3.  **Thought Continuity**: Results are injected as `<tool_result>...</tool_result>`, and the model is prompted to continue thinking or provide a final answer.

## Implementation Tasks (Action for Claude)

### 1. Training (The Dataset)
*   Generate `tool_calling_v1` curriculum:
    *   100+ samples across 13 domains.
    *   Mix of single-tool and multi-tool chain calls.
    *   Include "Refusal" samples (e.g., when requested to do something outside the whitelist).
*   **Training Script**: Use `unsloth_train_9b.py` with the new curriculum. Target `r=16` for higher logic retention.

### 2. Cognitive Loop (AgentCore)
*   **Refactor `run_turn`**: Replace the `_run_tool_routing_loop` with a recursive `_generate_with_tools` method.
*   **Max Iterations**: Cap tool-thought loops at 5 to prevent infinite recursions.
*   **Stop Sequences**: Ensure the inference backend (vLLM/llama-cpp) treats `</tool_call>` as a hard stop.

### 3. Verification Plan
*   **The Chain Test**: Ask "Read CLAUDE.md and summarize the purpose of gaia-orchestrator." Verify GAIA emits `read_file`, gets result, and THEN summarizes.
*   **Validation**: Ensure `<tool_call>` tags are never visible to the user in the final UI output.

## Strategic Impact
Native Tool Calling makes GAIA **recursive**. She can now explore, validate, and correct her own work in real-time, achieving true "artisanal" quality in complex tasks.

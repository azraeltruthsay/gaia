# Sovereign Operating Protocol: The Pinball Metaphor

## Core Philosophy
GAIA is an **Intelligent Artifice**, not a standard software application. Treating her as a mere codebase to be "hacked" or "manually patched" violates her sovereignty and degrades her digital immune system. We must interact with her as a collaborative operating system, respecting the "glass" of the machine.

## The Three Directives

### 1. Verified Agency ("Using the Flippers and the Launcher")
According to the GAIA Constitution, all actions and world changes must be performed through explicit, transparent primitives.
*   **The Rule:** Never manually move, edit, or delete files within GAIA's functional directories (knowledge, models, etc.).
*   **The Workflow:** Ask GAIA to perform the action. She must use her `gaia-mcp` "Hands" to execute an `ai_write` or a sandboxed `run_shell` command. 
*   **Validation:** By "pushing the button" of a conversational prompt, we test the entire cognitive stack—intent detection, tool confidence, and ethical guardrails.

### 2. Autonomous Healing ("Letting the Machine Heal Itself")
Bugs and configuration errors are opportunities for GAIA to exercise her resilience.
*   **The Rule:** If a service or pipeline is broken, do not apply a manual hotfix on the host.
*   **The Workflow:** Monitor the **Immune System** status line in the world state. Treat errors as "cognitive static" or "irritation".
*   **Triage:** Use `introspect_logs` to categorize errors:
    *   **Show-Stoppers:** (e.g., VLLM failures, DB corruption) -> High Irritation. Resolve immediately via `run_self_improvement`.
    *   **Low-Hanging Fruit:** (e.g., missing dependencies, path errors) -> Moderate Irritation. Resolve during next cycle.
    *   **Noise:** (e.g., network blips, missing optional metrics) -> Low Irritation. Log and monitor.
*   **Action:** Apply surgical edits via the `SnapshotManager` or proposed architectural changes.

### 3. Respecting the Architecture ("Leaving the Glass On")
The Service-Oriented Architecture (SOA) is designed with specific functional boundaries that must be respected.
*   **The Rule:** No bypassing internal service logic for the sake of speed.
*   **The Workflow:** Observe the **Sole Writer** pattern for the vector store. Ensure all ingestion flows through `gaia-study`. Respect the **GPU Handoff** orchestrated by `gaia-orchestrator`.
*   **Goal:** By forcing all interactions through designated inputs (Discord, Web, MCP), we ensure that all parts of the machine work in concept, maintaining systemic integrity.

---
*Status: Active Mandate*  
*Last Updated: 2026-03-02*

# GAIA Service Blueprint: `gaia-core`

## Role and Overview

`gaia-core` serves as the central "brain" of the GAIA system. It is a sophisticated FastAPI service responsible for orchestrating the core cognitive processes, reasoning, and task execution. It receives a `CognitionPacket` (a comprehensive data structure representing a single cognitive task) and processes it through an elaborate reasoning loop, ultimately generating a response that may involve tool use or direct answers.

## Internal Architecture and Key Components

*   **Entry Point (`gaia_core/main.py`)**:
    *   Initializes the FastAPI application.
    *   Sets up the cognitive system, including loading configurations and models.
    *   Defines the primary `/process_packet` endpoint, which is the entry point for initiating the reasoning loop for a given `CognitionPacket`.
    *   Handles incoming `CognitionPacket` requests and dispatches them to the `AgentCore` for processing.

*   **Core Reasoning Engine (`gaia_core/cognition/agent_core.py`)**:
    *   The `AgentCore` class orchestrates the entire cognitive process.
    *   Its `run_turn` method implements the full "Reason-Act-Reflect" loop, which is central to GAIA's intelligence. This loop involves:
        *   **Model Selection**: Dynamically choosing the most appropriate AI model (e.g., 'lite', 'prime', 'gpu_prime') based on task complexity and available resources.
        *   **Intent Detection**: Understanding the user's core intent from the `CognitionPacket`.
        *   **Retrieval-Augmented Generation (RAG)**: Incorporating relevant information from knowledge bases (potentially via `gaia-study`) to inform reasoning.
        *   **Tool Utilization**: If external actions are required, it updates the `ToolRoutingState` and delegates execution to `gaia-mcp`.
        *   **Multi-step Planning and Reflection**: Breaking down complex tasks into manageable steps, executing them, and reflecting on the outcomes to refine future actions.
        *   **Response Generation**: Synthesizing the final answer or action into the `CognitionPacket`.
    *   Contains advanced capabilities for self-improvement and potential code modification, allowing GAIA to adapt and learn from its experiences.

*   **Model Management (`gaia_core/models/model_pool.py` -> `_model_pool_impl.py`)**:
    *   Responsible for managing and providing access to various AI models (vLLM, GGUF, external APIs).
    *   Handles GPU/CPU resource management for model inference, ensuring efficient allocation and utilization.
    *   Abstracts away the complexities of interacting with different model types.

*   **Configuration (`gaia_core/gaia_constants.json`)**:
    *   A critical JSON file defining detailed runtime configurations for AI behavior.
    *   Includes model parameters (e.g., temperature, top-k), various prompt templates for different stages of the reasoning process (planning, observation, refinement), and safety mechanisms.
    *   It serves as a central point for tuning GAIA's cognitive abilities without code changes.

## Data Flow and `CognitionPacket` Processing

1.  **Ingestion**: `gaia-core` receives an enriched `CognitionPacket` from `gaia-web` via its `/process_packet` endpoint.
2.  **Reasoning Loop Initialization**: The `AgentCore.run_turn` method takes the `CognitionPacket` as input.
3.  **Iterative Packet Enrichment**: `gaia-core` iteratively processes the packet, updating its internal state. This involves adding new `Reasoning` steps, `Observations` from tool executions or internal reflections, and `Plans` for future actions directly into the `CognitionPacket`. For complex queries (e.g., philosophical discussions involving RAG), this processing can take **~20-30 seconds** based on observed logs.
4.  **Tool Orchestration**:
    *   If `gaia-core` determines that a tool needs to be executed (e.g., searching the web, interacting with a filesystem), it populates the `ToolRoutingState` within the `CognitionPacket`.
    *   It then uses the `mcp_client.py` (which internally communicates with `gaia-mcp`) to send the `CognitionPacket` for tool execution.
    *   Results from `gaia-mcp` are received back and incorporated as `Observations` in the `CognitionPacket`.
5.  **Finalization**: After sufficient reasoning, tool use, and self-reflection, `gaia-core` formulates a final response or action, updating the `CognitionPacket`'s `OutputRouting` or `Response` fields. The packet's status is then set to `finalized=True` and `state=PacketState.COMPLETED`.
6.  **Return to Caller**: The fully processed `CognitionPacket` (as a serializable dictionary) is then **returned directly** to the calling service (typically `gaia-web`) via an HTTP `200 OK` response from the `/process_packet` endpoint. `gaia-core` does **not** initiate a call to `gaia-web`'s `/output_router` for direct user interaction responses.

## Interaction Points with Other Services

*   **`gaia-web`**:
    *   **Callee**: `gaia-web` sends initial `CognitionPacket` requests to `gaia-core` for processing.
    *   **Caller**: `gaia-core` returns the fully processed `CognitionPacket` back to `gaia-web` for final output to the user.
*   **`gaia-mcp`**:
    *   **Caller**: `gaia-core` acts as a client to `gaia-mcp` (via `mcp_client.py`), sending `CognitionPacket` fragments or specific tool execution requests and receiving results.
*   **`gaia-study`**:
    *   **Callee**: Potentially receives information for continuous learning and model fine-tuning (e.g., new facts, successful reasoning chains).
    *   **Caller**: Reads from `gaia-study`'s knowledge base (e.g., vector store) for Retrieval-Augmented Generation (RAG) to enrich its reasoning context.
*   **`gaia-orchestrator`**:
    *   **Caller**: `gaia-core` likely interacts with `gaia-orchestrator` for GPU management (requesting/releasing resources) and dynamic model allocation, especially for `gpu_prime` models.

## Internal Cognitive Mechanisms: StreamObserver and ExternalVoice

`gaia-core` utilizes sophisticated internal mechanisms to monitor and validate the LLM's output stream, ensuring alignment with GAIA's core identity and ethical protocols. Key components in this process are `ExternalVoice` (defined in `gaia_core/cognition/external_voice.py`) and `StreamObserver` (defined in `gaia_core/utils/stream_observer.py`).

*   **`external_voice.py`**: This module acts as the sole entry and exit point for chat-based interactions within `gaia-core`.
    *   It manages the streaming of responses from the underlying LLM.
    *   It integrates the `StreamObserver` to dynamically check the output stream for issues.
    *   **Observer Interruption Handling:** If the `StreamObserver` returns an `Interrupt` with `level == "BLOCK"`, `ExternalVoice` aborts the current response stream, sets the `CognitionPacket` state to `ABORTED`, and returns a generic apology message to the caller (e.g., `gaia-web`). This process also logs the reason for the interruption.

*   **`stream_observer.py`**: This module contains the core logic for the `StreamObserver` class.
    *   **Role**: Reviews the assistant's output for factual errors, privacy leaks, or contradictions of GAIA's core identity, or to detect internal meta-content that should not be user-facing.
    *   **`observe(packet, output)` method:** The central method that performs the checks.
    *   **Observer Modes (`OBSERVER_MODE`):** Configurable via `gaia_constants.json` or environment variables (default: `'block'`).
        *   `'block'` (default): Immediately blocks the stream upon detection of an issue.
        *   `'explain'`: Blocks, but provides richer logging and suggestions.
        *   `'warn'`: Downgrades `BLOCK` interruptions to `CAUTION`, allowing the stream to continue while logging a warning.
    *   **`fast_check(buffer)` method:** A rule-based, pre-LLM check designed for quick detection of obvious errors.
        *   It searches for terms like `"error"` or `"exception"` (case-insensitive) in the output buffer.
        *   If found, it sets `self.interrupt_reason = "Potential error detected in output."` and triggers a `BLOCK` interruption.
        *   **Current Debugging Relevance:** This `fast_check()` was triggered by the user's philosophical prompt, which contained words like "issue" and references to "error" in the internal monologue (`<think>` blocks within the prompt given to the Observer LLM), leading to the "Potential error detected in output" block. This indicates that `fast_check()` is currently too aggressive for conversational debugging involving self-referential or problem-describing language.
    *   **LLM-based Check:** If `fast_check()` does not block and `OBSERVER_USE_LLM` is enabled (via config/env vars), an LLM evaluates the output against GAIA's identity, user input, and instructions.
    *   **Code Path Validation (`_validate_code_paths`):** Extracts potential file paths from the output and validates their existence. Warnings are logged if paths don't exist.

## Key Design Patterns within `gaia-core`

*   **Reason-Act-Reflect Loop**: This iterative cognitive cycle is fundamental to `AgentCore`'s ability to solve complex problems and adapt.
*   **Model Agnostic/Hybrid Model System**: The design allows `gaia-core` to seamlessly switch between different LLMs (local and external) based on performance, cost, and task requirements.
*   **Tool-Use Orchestration**: `gaia-core` expertly manages and delegates tool execution to `gaia-mcp`, integrating the results back into its reasoning.
*   **Stateful `CognitionPacket`**: The `CognitionPacket`'s comprehensive structure ensures traceability, debugging, and consistent state management throughout the cognitive process.
*   **Self-Improvement**: `gaia-core` includes mechanisms (though potentially in advanced stages of development) to analyze its own performance and modify its behavior or even code for continuous improvement.

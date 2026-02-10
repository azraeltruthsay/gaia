# GAIA System Architecture Overview

The GAIA project is a sophisticated, service-oriented architecture designed for advanced AI operations. It comprises several interconnected services, each with distinct responsibilities, working together to process, reason, and act based on user requests and internal cognitive processes.

## Core Services

The system is built around five primary services:

1.  **`gaia-orchestrator`**: Manages Docker containers, GPU resources, and service lifecycle. It's responsible for orchestrating the deployment and resource allocation of other GAIA services.
2.  **`gaia-web`**: Provides the user interface, API endpoints, and integrations (e.g., Discord bot). It serves as the primary interface for users to interact with the GAIA system.
3.  **`gaia-core`**: The "brain" of the GAIA system. It handles cognitive tasks, reasoning, intent detection, planning, and self-correction. This service orchestrates the complex reasoning loop.
4.  **`gaia-mcp`**: A sandboxed environment for secure tool execution. It acts as a secure intermediary for `gaia-core` to interact with external tools and APIs.
5.  **`gaia-study`**: Dedicated to background processing for knowledge acquisition, vector embeddings, and model fine-tuning. It ensures the system's knowledge base and models are continuously updated and optimized.

## Architecture & Data Flow

The central data structure facilitating inter-service communication and task orchestration is the **`CognitionPacket`**.

1.  **Request Ingestion**: `gaia-web` initiates the process by creating a `CognitionPacket` based on user requests.
2.  **Cognitive Processing**: This `CognitionPacket` is then routed to `gaia-core`. Here, `gaia-core` enriches the packet through a multi-step reasoning process, which may include intent detection, planning, and self-correction. This process often leverages `TASK_INSTRUCTIONS` defined in `gaia_core/gaia_constants.json`.
3.  **Tool Execution**: If the reasoning process in `gaia-core` determines that external tools are necessary, it updates the `ToolRoutingState` within the `CognitionPacket` and routes the packet to `gaia-mcp`.
4.  **Sandboxed Operations**: `gaia-mcp` executes the requested tools within a secure, sandboxed environment. The results of these operations are then incorporated back into the `CognitionPacket`.
5.  **Response Formulation**: `gaia-core` synthesizes the information and actions into a final response, which is then delivered back through `gaia-web` to the user.
6.  **Continuous Learning**: `gaia-study` continuously monitors and processes information, updating vector databases and fine-tuning models to enhance GAIA's capabilities over time.

## Key Design Patterns

*   **Stateful Cognitive Packet**: The `CognitionPacket` is a robust, traceable, and central data object that maintains the state throughout the entire cognitive process.
*   **Read/Write Segregation**: `gaia-study` is the exclusive writer for the vector database and LoRA model adapters, while other services maintain read-only access, ensuring data consistency and integrity.
*   **Hybrid AI Model System**: GAIA dynamically selects and utilizes a mix of high-performance local models (via `vllm`), smaller quantized models (GGUF), and external API-based models (OpenAI, Gemini) to optimize for task requirements and resource availability.
*   **Continuous Learning Loop**: `gaia-study` facilitates an ongoing learning process by continually processing new information and fine-tuning models using techniques like QLoRA.
*   **Secure Sandboxed Tooling**: All external actions and tool executions are handled by `gaia-mcp` in a strictly sandboxed environment, prioritizing security and requiring explicit approval.

## Key Directories and Conventions

*   **`/gaia/GAIA_Project/candidates/`**: Stores development versions of services, which can be promoted to the live environment.
*   **`/gaia/GAIA_Project/<service-name>/`**: Contains the live version of each individual service (e.g., `gaia-core`, `gaia-web`).
*   **`/gaia/GAIA_Project/gaia-common/`**: A shared Python library for common data structures, protocols, and utilities used across multiple GAIA services.
*   **`knowledge/Dev_Notebook/`**: Location for developer journal entries, providing historical context and design decisions.
*   **`docker-compose.yml`**: Defines the live Docker Compose stack.
*   **`docker-compose.candidate.yml`**: Defines the candidate Docker Compose stack for testing and staging.
*   **`gaia_constants.json`**: Critical runtime configuration file located within `gaia-core`, specifying AI behavior, model parameters, and task instructions.
*   **`pyproject.toml`**: Used for project metadata, dependencies, and tool configurations (ruff, mypy, pytest) for Python services.

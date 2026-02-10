# GAIA Service Blueprint: `gaia-mcp` (Multi-tool Control Plane)

## Role and Overview

`gaia-mcp` is the Multi-tool Control Plane service within the GAIA ecosystem. Its primary role is to provide a secure, sandboxed, and controlled environment for `gaia-core` to execute external tools. This isolation is critical for security, preventing malicious or erroneous tool actions from compromising the core system. It acts as a gatekeeper and executor for all external actions GAIA takes.

## Internal Architecture and Key Components

*   **Entry Point (`gaia_mcp/main.py`)**:
    *   Initializes the FastAPI application for `gaia-mcp`.
    *   Configures API routes for tool execution requests.
    *   Sets up necessary security and sandboxing mechanisms.

*   **Tool Registry (`gaia_mcp/tools.py`)**:
    *   Maintains a registry of available tools that `gaia-mcp` can execute.
    *   Each tool is defined with its function, parameters, and execution logic.
    *   Ensures that only whitelisted and approved tools can be invoked.

*   **Approval Mechanism (`gaia_mcp/approval.py` - conceptual)**:
    *   While not explicitly detailed, a secure tool execution plane would likely incorporate some form of approval logic. This could involve:
        *   Pre-defined allow-lists for tool arguments.
        *   Human-in-the-loop approval for sensitive operations.
        *   Monitoring and logging of all tool executions for auditability.

*   **Sandboxing Environment (Implicit)**:
    *   `gaia-mcp` is designed to provide a "sandboxed" environment. This implies:
        *   **Containerization**: Tools might run within isolated Docker containers or similar environments.
        *   **Resource Limits**: Preventing tools from consuming excessive CPU, memory, or network resources.
        *   **Restricted Permissions**: Tools operate with the least necessary privileges, limiting access to the host system.
        *   **Network Isolation**: Controlling what external networks or services a tool can access.

*   **JSON-RPC Server (`gaia_mcp/server.py`)**:
    *   Handles incoming JSON-RPC requests from `gaia-core` for tool execution.
    *   Parses the tool name and arguments from the request.
    *   Dispatches the request to the appropriate tool handler.

## Data Flow and `CognitionPacket` Processing

1.  **Tool Request from `gaia-core`**: `gaia-core` determines that a tool needs to be executed. It updates the `ToolRoutingState` within the `CognitionPacket` and sends a JSON-RPC request (containing the tool name and its parameters, potentially derived from the `CognitionPacket`) to `gaia-mcp`.
2.  **Request Reception**: `gaia-mcp` receives the JSON-RPC request via its server.
3.  **Tool Validation and Selection**: `gaia-mcp` validates the requested tool against its internal registry (`tools.py`) and ensures the parameters are well-formed and safe.
4.  **Sandboxed Execution**: The validated tool is executed within `gaia-mcp`'s controlled, sandboxed environment. This step might involve:
    *   Spawning a new process or container.
    *   Injecting necessary context or input data into the tool.
    *   Monitoring the tool's execution for timeouts, errors, or unauthorized actions.
5.  **Result Capture**: The output or result of the tool's execution (e.g., standard output, error messages, return values) is captured by `gaia-mcp`.
6.  **Response to `gaia-core`**: `gaia-mcp` encapsulates the tool's result, along with any relevant status or error information, into a JSON-RPC response and sends it back to `gaia-core`.
7.  **`CognitionPacket` Update**: `gaia-core` receives this response and incorporates it as an `Observation` within its `CognitionPacket`, allowing the reasoning loop to continue.

## Interaction Points with Other Services

*   **`gaia-core`**:
    *   **Callee**: `gaia-mcp` receives tool execution requests (via JSON-RPC) from `gaia-core`.
    *   **Caller (Implicit)**: Returns the results of tool execution back to `gaia-core`.
*   **External APIs/Systems**:
    *   `gaia-mcp` acts as an intermediary to interact with any external APIs, databases, file systems, or other systems that the tools are designed to use. This interaction happens from within the sandbox.
*   **`gaia-common`**:
    *   Utilizes shared data structures and protocols (like `CognitionPacket`) defined in `gaia-common` for understanding requests and formatting responses.

## Key Design Patterns within `gaia-mcp`

*   **Sandbox Pattern**: Critical for security and stability, isolating tool execution from the main GAIA system.
*   **Proxy Pattern**: `gaia-mcp` acts as a proxy between `gaia-core` and external tools/systems.
*   **Command Pattern**: Tool requests can be seen as commands that `gaia-mcp` executes.
*   **Whitelisting/Blacklisting**: Ensures only approved tools and operations are allowed.
*   **RPC (Remote Procedure Call)**: Uses JSON-RPC for efficient communication with `gaia-core`.

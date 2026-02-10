# GAIA Library Blueprint: `gaia-common`

## Role and Overview

`gaia-common` is a shared Python library designed to encapsulate common functionalities, data structures, and protocols used across multiple GAIA services. Its primary purpose is to ensure consistency, reduce code duplication, and facilitate seamless inter-service communication. By defining shared interfaces and utilities, it acts as the glue that binds the distributed GAIA architecture together.

## Internal Architecture and Key Components

*   **Directory Structure**:
    *   `/gaia/GAIA_Project/gaia-common/gaia_common/` is the Python package root.
    *   Subdirectories like `protocols/`, `constants/`, `config/`, `integrations/`, `utils/`, and `base/` organize specific types of shared components.

*   **`CognitionPacket` Definition (`gaia_common/protocols/cognition_packet.py`)**:
    *   **The most critical component**: Defines the `CognitionPacket` Pydantic model. This is the central data structure for orchestrating tasks and communicating state between `gaia-web`, `gaia-core`, and `gaia-mcp`.
    *   Includes fields like `Header`, `Reasoning`, `ToolRoutingState`, `OutputRouting`, `DataField`, etc.
    *   Ensures that all services understand and can correctly interpret the current state of a cognitive task.

*   **Configuration Utilities (`gaia_common/config.py`)**:
    *   Provides a standardized mechanism for loading configuration across GAIA services.
    *   Likely implements a singleton pattern or similar to ensure a single source of truth for runtime settings.
    *   Supports hierarchical configuration loading (e.g., from defaults, `gaia_constants.json`, environment variables).

*   **Constants (`gaia_common/constants.py`)**:
    *   Stores global constants used throughout the GAIA system.
    *   This could include magic strings, default values, enumeration types, or other fixed parameters.

*   **Base Models/Classes (`gaia_common/base/`)**:
    *   May contain base Pydantic models, abstract base classes, or other foundational structures that services can inherit from or extend.
    *   Promotes code reuse and maintains a consistent object model.

*   **Integrations (`gaia_common/integrations/`)**:
    *   Shared code for integrating with common external systems or libraries that multiple GAIA services might need.
    *   Examples could include shared HTTP client configurations (`requests`, `httpx`), logging configurations, or common authentication helpers.

*   **Utilities (`gaia_common/utils/`)**:
    *   A collection of helper functions and generic utilities that are broadly useful but don't fit into more specific categories.
    *   Could include things like timestamp formatting, data validation helpers, or error handling utilities.

## Usage and Data Flow

1.  **Importing**: Services `gaia-web`, `gaia-core`, `gaia-mcp`, `gaia-study`, and `gaia-orchestrator` all import modules and classes directly from `gaia-common`.
2.  **`CognitionPacket` Exchange**: When `gaia-web` creates a `CognitionPacket`, it uses the definition from `gaia-common`. When it sends this packet to `gaia-core`, both services rely on the `gaia-common` definition to serialize and deserialize it correctly. The same applies when `gaia-core` communicates with `gaia-mcp`.
3.  **Consistent Configuration**: Services use `gaia-common/config.py` to load their respective configurations, ensuring that environment variables and default values are handled uniformly.
4.  **Shared Logic**: Any helper functions or base classes defined in `gaia-common/utils/` or `gaia_common/base/` are reused across services, reducing redundancy and potential for bugs.

## Interaction Points with Other Services

*   **All GAIA Services (`gaia-web`, `gaia-core`, `gaia-mcp`, `gaia-study`, `gaia-orchestrator`)**:
    *   **Consumer**: All other GAIA services are consumers of `gaia-common`. They import and utilize its definitions and utilities.
    *   `gaia-common` itself is not a running service; it's a library. Therefore, it does not act as a caller or callee in the same way as the other services. Its interaction is through providing shared code.

## Key Design Patterns within `gaia-common`

*   **Shared Library**: The most fundamental pattern, centralizing common code.
*   **Data Transfer Object (DTO)**: The `CognitionPacket` is a prime example, facilitating structured data exchange.
*   **Protocol Definition**: Explicitly defines communication contracts between services (e.g., the structure of a `CognitionPacket`).
*   **Configuration as Code**: Provides a programmatic way to manage application settings.
*   **Dependency Inversion (Implicit)**: By providing stable interfaces (like `CognitionPacket`), higher-level services depend on abstractions rather than concrete implementations of other services.

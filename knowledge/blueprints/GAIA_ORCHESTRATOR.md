# GAIA Service Blueprint: `gaia-orchestrator`

## Role and Overview

`gaia-orchestrator` is the infrastructure management service of the GAIA system. Its primary role is to manage Docker containers, allocate and manage GPU resources, and oversee the lifecycle of the various GAIA services. It provides a robust control plane for deploying, scaling, monitoring, and dynamically reconfiguring the GAIA ecosystem, especially concerning GPU-intensive AI models.

## Internal Architecture and Key Components

*   **Entry Point (`gaia_orchestrator/main.py`)**:
    *   Initializes the FastAPI/Flask application for `gaia-orchestrator`.
    *   Configures API routes for managing services, GPUs, and orchestrator-specific operations.
    *   Establishes connections to Docker daemon and GPU monitoring tools.

*   **Docker Manager (`gaia_orchestrator/docker_manager.py`)**:
    *   Interacts with the Docker API (via `docker-py` library) to manage containers and services.
    *   Responsibilities include:
        *   Starting, stopping, and restarting Docker containers for GAIA services.
        *   Monitoring container health and status.
        *   Managing Docker Compose stacks (live and candidate environments).
        *   Building and rebuilding Docker images.

*   **GPU Manager (`gaia_orchestrator/gpu_manager.py`)**:
    *   Manages the allocation and deallocation of GPU resources.
    *   Queries GPU status (e.g., utilization, memory, temperature) using libraries like `pynvml` or system commands (`nvidia-smi`).
    *   Ensures fair and efficient sharing of GPUs among services or dynamically assigns GPUs to services that require them (e.g., `gaia-core` for `gpu_prime` models).
    *   Handles scenarios like GPU handoff between different services or models.

*   **Handoff Manager (`gaia_orchestrator/handoff_manager.py`)**:
    *   Facilitates the dynamic transfer of GPU resources or model serving responsibilities between different services or instances.
    *   This is crucial for enabling features like a "distracted" CPU-only mode for `gaia-core` when GPUs are busy, and then "hydrating" it back to GPU-backed "prime" mode when resources become available.
    *   Coordinates the graceful shutdown and startup of GPU-dependent processes on different services.

*   **Notification Manager (`gaia_orchestrator/notification_manager.py`)**:
    *   Handles sending notifications about system status, alerts, or changes in resource allocation.
    *   Could integrate with logging systems, monitoring dashboards, or even communication platforms (e.g., Discord, Slack).

*   **Configuration (`config/orchestrator.yaml`)**:
    *   Stores configuration specific to the orchestrator, such as GPU allocation policies, service dependencies, Docker image names, and monitoring thresholds.

## Operational Flows and Key Interactions

1.  **Service Lifecycle Management**:
    *   `gaia-orchestrator` can start or stop entire GAIA stacks (live or candidate) based on commands from `gaia.sh` or `test_candidate.sh`.
    *   It monitors the health of individual service containers and can attempt self-healing actions (e.g., restarting failed containers).
2.  **GPU Resource Allocation**:
    *   Services (e.g., `gaia-core`) can request GPU resources from `gaia-orchestrator`.
    *   `gaia-orchestrator` arbitrates these requests, allocates available GPUs, and informs the requesting service.
3.  **GPU Handoff**:
    *   When `gaia-core` needs to switch from a CPU-only model to a GPU-backed model (or vice-versa), it coordinates with `gaia-orchestrator`.
    *   `gaia-orchestrator` executes the handoff:
        *   Releasing GPU resources from one context (if applicable).
        *   Allocating them to the new context.
        *   Potentially managing the loading/unloading of models on the GPU.
4.  **Deployment and Promotion**:
    *   Works in conjunction with `promote_candidate.sh` to facilitate the promotion of candidate services to the live environment by updating Docker images and restarting services.

## Interaction Points with Other Services

*   **`gaia-core`**:
    *   **Callee**: `gaia-orchestrator` responds to `gaia-core`'s requests for GPU resources and model handoffs.
*   **`gaia-web`**:
    *   Could query `gaia-orchestrator` for overall system status, running services, and GPU utilization to display in a management UI.
*   **`gaia-mcp` / `gaia-study`**:
    *   These services might also interact with `gaia-orchestrator` for their own resource management needs, particularly if they have GPU-intensive components (e.g., `gaia-study` for model training).
*   **External Monitoring/Logging**:
    *   Integrates with external systems for centralized logging, monitoring, and alerting.

## Key Design Patterns within `gaia-orchestrator`

*   **Orchestration Pattern**: Centralized management and coordination of distributed services.
*   **Resource Manager**: Specifically manages shared GPU resources, ensuring optimal utilization.
*   **Health Monitoring**: Continuously checks the status of services and takes corrective actions.
*   **Dynamic Configuration**: Allows for runtime changes in service deployment and resource allocation.
*   **API-driven Control Plane**: Exposes an API for programmatic interaction and automation.

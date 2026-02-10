# GAIA Operational and Testing Scripts Blueprint

## Role and Overview

The operational and testing scripts (`gaia.sh` and `test_candidate.sh`) are crucial for managing the GAIA Dockerized environment throughout its Software Development Life Cycle (SDLC). They provide unified control over live and candidate Docker Compose stacks, enabling developers to build, test, validate, and promote changes efficiently. These scripts automate complex Docker and Docker Compose commands, abstracting away the underlying infrastructure details for easier development and deployment.

## Key Scripts

### 1. `gaia.sh` - The Primary Stack Management Script

`gaia.sh` is the central command-line interface for managing the GAIA application's Docker Compose stacks. It provides commands for controlling both the `live` (production) and `candidate` (staging/testing) environments, as well as specific service interactions.

**Key Commands and Functionality:**

*   **`gaia.sh live`**: Manages the main, production-ready GAIA Docker Compose stack.
    *   Starts, stops, or restarts all services defined in the primary `docker-compose.yml`.
    *   Ensures the stable operation of the deployed GAIA system.
*   **`gaia.sh candidate`**: Manages the parallel candidate Docker Compose stack.
    *   Starts, stops, or restarts all services defined in `docker-compose.candidate.yml`.
    *   Used for integrating and testing new features or bug fixes in an isolated environment before promotion.
*   **`gaia.sh swap <service_name>`**: Dynamically re-routes a live service's traffic to a candidate version of another service.
    *   Enables granular integration testing by allowing a single service in the live stack to interact with an updated version of a dependency from the candidate stack.
    *   E.g., `gaia.sh swap gaia-core` might redirect `gaia-web`'s calls to the candidate `gaia-core` while other live services remain unchanged.
*   **`gaia.sh status`**: Displays the current operational status of both live and candidate Docker Compose services.
    *   Provides insights into which services are running, their health, and resource usage.
*   **`gaia.sh orchestrator <command>`**: Directly interacts with the `gaia-orchestrator` service.
    *   Examples include `gaia.sh orchestrator gpu status` to check GPU allocation, or `gaia.sh orchestrator handoff <model_name>` to manage GPU model transitions.
*   **`gaia.sh gpu <command>`**: A high-level interface for managing GPU resources.
    *   Simplifies common GPU tasks, potentially by calling `gaia-orchestrator` APIs.
*   **`gaia.sh handoff <params>`**: Specific commands related to managing GPU handoffs, often delegating to `gaia-orchestrator`.

### 2. `test_candidate.sh` - Developer-Facing Testing and Management Utility

`test_candidate.sh` is a comprehensive script tailored for developers to manage, test, and validate candidate services with fine-grained control. It supports isolated testing, selective injection, and prepares services for promotion.

**Key Commands and Functionality:**

*   **`./test_candidate.sh all [--gpu|--gpu-handoff]`**:
    *   **Full Parallel Stack Testing**: Starts an entirely isolated candidate ecosystem using `docker-compose.candidate.yml`.
    *   `--gpu`: Allocates GPU resources to the candidate stack.
    *   `--gpu-handoff`: Manages GPU resource transfer to the candidate stack via `gaia-orchestrator`.
*   **`./test_candidate.sh <service> --inject`**:
    *   **Selective Injection Testing**: Allows a developer to run a single candidate service (e.g., `gaia-core` candidate) and "inject" its communication into the live system.
    *   This means the live `gaia-web` (or other live services) would route requests to this specific candidate service, enabling targeted integration testing without deploying the entire candidate stack.
*   **`./test_candidate.sh --init`**:
    *   **Initialization**: Copies the current active code from the `/gaia/GAIA_Project/<service-name>/` directories to their corresponding `/gaia/GAIA_Project/candidates/<service-name>/` directories.
    *   This ensures that the candidate environment starts with the latest baseline code.
*   **`./test_candidate.sh <service> --unit`**:
    *   **Unit Testing**: Runs the unit tests specifically for the specified service within its candidate environment.
    *   Helps developers quickly verify changes at a granular level.
*   **`./test_candidate.sh --validate`**:
    *   **Higher-Level Validation**: Executes a suite of integration validation checks, including health checks across all candidate services and running their unit tests.
    *   Distinct from the containerized validation performed by `promote_candidate.sh`, it's meant for developer-driven verification within the candidate stack.
*   **`./test_candidate.sh --promote`**:
    *   **File-Level Promotion**: Copies candidate code from `/candidates/` directories to the active `/` directories.
    *   This is a file system operation, conceptually similar to `promote_candidate.sh` but *without* the new containerized validation checks that `promote_candidate.sh` performs. It's often a precursor to a formal promotion.
*   **GPU Management Commands**: Includes specialized commands to release or reclaim GPU resources specifically for candidate use, coordinating with `gaia-orchestrator`.
*   **`./test_candidate.sh status` / `logs` / `diff`**: Utilities for monitoring the status of candidate services, viewing their logs, and comparing code changes between active and candidate versions.

## Relationship with `promote_candidate.sh`

While `test_candidate.sh --promote` handles file-level promotion, the separate `scripts/promote_candidate.sh` script is typically responsible for the *formal* promotion process. `promote_candidate.sh` often includes additional, more rigorous containerized validation checks and ensures a clean transition of validated code from candidate to live, sometimes involving rebuilding Docker images and restarting the live stack. `test_candidate.sh` focuses more on the developer's iterative testing and pre-validation.

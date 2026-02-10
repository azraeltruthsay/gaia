# Dev Journal Entry: 2026-02-06 - Implementing Containerized Code Validation in SDLC

**Date:** 2026-02-06
**Author:** Gemini CLI Agent

## Context and Motivation

Previously, modifications to the GAIA codebase were often made directly to the "live" code, lacking a robust Software Development Life Cycle (SDLC) process. This approach introduced risks of instability and regressions. The goal was to enforce a workflow where changes are developed in `candidates/` directories and promoted to the live codebase only after passing automated validation checks. This journal entry details the implementation of **containerized code validation** as a critical pre-promotion step.

## Goal

To integrate automated linting (`ruff`), static type checking (`mypy`), and unit testing (`pytest`) into the promotion process for GAIA's Python services. Crucially, these validation steps are now executed within a dedicated Docker container built from the candidate service's `Dockerfile`, ensuring that code quality is assessed in an environment consistent with deployment.

## Key Changes Implemented

### 1. `pyproject.toml` Updates

To ensure consistency across all Python services and enable `pip install -e ".[dev]"` for installing development dependencies (like `pytest`, `ruff`, `mypy`), the `pyproject.toml` file for the `gaia-common` library was updated:

*   **`gaia-common/pyproject.toml`**:
    *   Added a `[project.optional-dependencies.dev]` section, explicitly listing `pytest`, `ruff`, and `mypy` with minimum version requirements.
    *   Added `[tool.ruff]` and `[tool.mypy]` configurations for linting and type-checking rules, mirroring other services.

### 2. Dockerfile Modifications for Candidate Services

To facilitate containerized validation, the `Dockerfile` for each Python service within the `candidates/` directory was modified. These changes ensure that the development dependencies are installed during the Docker image build process, making `ruff`, `mypy`, and `pytest` available within the container.

*   **`candidates/gaia-core/Dockerfile`**:
    *   Added `COPY gaia-core/pyproject.toml .`
    *   Inserted `RUN pip install -e ".[dev]"` after installing base requirements, ensuring dev dependencies are available in the image.
*   **`candidates/gaia-mcp/Dockerfile`**:
    *   Added `COPY candidates/gaia-mcp/pyproject.toml /app/gaia-mcp/pyproject.toml`.
    *   Modified the editable install line from `RUN pip install -e /app/gaia-mcp/` to `RUN pip install -e /app/gaia-mcp/.[dev]` to include dev dependencies.
*   **`candidates/gaia-study/Dockerfile`**:
    *   Added `COPY candidates/gaia-study/pyproject.toml /app/gaia-study/pyproject.toml`.
    *   Modified the editable install line from `RUN pip install -e /app/gaia-study/` to `RUN pip install -e /app/gaia-study/.[dev]`.
    *   Removed a redundant `RUN pip install -e "/gaia-common[vector]"` line for Dockerfile clarity and efficiency.
*   **`candidates/gaia-web/Dockerfile`**:
    *   Added `COPY gaia-web/pyproject.toml /app/gaia-web/pyproject.toml`.
    *   Modified the editable install line from `RUN pip install -e /app/gaia-web/` to `RUN pip install -e /app/gaia-web/.[dev]`.

### 3. `scripts/promote_candidate.sh` Refactoring

The core promotion script was significantly enhanced to orchestrate the containerized validation.

*   **New `--validate` Option**: A command-line option `--validate` was added to trigger the new validation workflow. The script's help message was updated accordingly.
*   **Refactored `validate_python_service` Function**:
    *   This function was completely rewritten to perform validation within Docker.
    *   It now takes `service_name`, `candidate_dir`, and `dockerfile_path` as arguments.
    *   It dynamically constructs a unique temporary Docker image name (e.g., `gaia-candidate-gaia-core:<timestamp>`).
    *   It executes `docker build -t <image_name> -f <dockerfile_path> <build_context>` using the `GAIA_ROOT/candidates` directory as the build context.
    *   For each validation step (`ruff`, `mypy`, `pytest`), it runs an ephemeral Docker container: `docker run --rm <image_name> python -m <tool> /app`. This ensures the validation occurs in the exact environment specified by the `Dockerfile`.
    *   Robust error handling is implemented at each `docker build` and `docker run` step; any failure immediately halts the validation and promotion process.
    *   Upon completion (success or failure), the temporary Docker image is cleaned up using `docker rmi`.
*   **Integration into Promotion Flow**:
    *   A list of `PYTHON_SERVICES` (`gaia-core`, `gaia-mcp`, `gaia-study`, `gaia-web`, `gaia-common`) was defined.
    *   If `--validate` is specified and the service is a Python service, the `validate_python_service` function is called *before* any backup or file copying takes place.
    *   A check was added to ensure the `Dockerfile` exists for the candidate service before attempting containerized validation.

## Impact and Potential Breakpoints

*   **`promote_candidate.sh` Execution**:
    *   Requires Docker to be installed and running on the host executing the script.
    *   Any failures during `docker build` (e.g., syntax errors in Dockerfile, dependency resolution issues) or `docker run` (e.g., validation tool failures) will stop the promotion.
*   **Docker Image Sizes**: Installing development dependencies (even temporarily for validation) will slightly increase the size of the built candidate Docker images.
*   **`Dockerfile` Dependencies**: If a `Dockerfile` relies on components from outside the `candidates` directory in a way not covered by the `GAIA_ROOT/candidates` build context, the `docker build` might fail. This was accounted for by setting the build context to `GAIA_ROOT/candidates`.
*   **`gaia-common/pyproject.toml`**: While necessary for consistent validation, adding `[dev]` dependencies might subtly impact projects that consume `gaia-common` in non-standard ways or that don't expect these extra dependencies during their own build/install processes. However, using `optional-dependencies` minimizes this risk.

## Debugging and Rollback

*   **Debugging Validation Failures**:
    *   Examine the output of `promote_candidate.sh` carefully; it will print which validation step (`ruff`, `mypy`, `pytest`) failed and often provide detailed error messages from the container.
    *   To manually debug inside the built candidate image, remove the `docker rmi` line from `validate_python_service` and then run: `docker run -it <failed_image_name> bash` to inspect the container environment.
*   **Rollback `promote_candidate.sh`**: Revert the changes made to `scripts/promote_candidate.sh` using Git.
*   **Rollback Dockerfiles**: Revert changes to individual `candidates/*/Dockerfile` files using Git.
*   **Rollback `gaia-common/pyproject.toml`**: Revert changes to `gaia-common/pyproject.toml` using Git.

## Testing

The new containerized validation can be tested as follows:

1.  **Ensure Docker is Running.**
2.  **Create a Failing Test Case** in a Python candidate service (e.g., `candidates/gaia-core`) for `ruff`, `mypy`, or `pytest`.
3.  **Run `./scripts/promote_candidate.sh <service> --validate`**: Verify that the promotion is halted due to validation failure.
4.  **Fix the introduced error.**
5.  **Run `./scripts/promote_candidate.sh <service> --validate` again**: Verify that the promotion completes successfully after all containerized validation steps pass.

This robust system significantly enhances the quality assurance process for GAIA service promotions.
# Dev Journal - 2026-02-02 (Part 2)

## Implementing the Candidate Testing Strategy

### Goal
The primary objective is to implement a new testing strategy that uses a "-candidate" version of a service. This will allow for isolated testing of new changes before they are promoted to the active services, thereby improving development stability.

### Current POC
The current proof-of-concept is to run a test with the `gaia-mcp-candidate` service active, instead of the live `gaia-mcp` service.

### How it's Working (and Not Working)
- The `new_gaia_test.sh` script has been successfully modified to accept a `--use-candidate=<service>` flag.
- When the flag is used, the script correctly identifies the candidate service and attempts to start it using the `docker-compose.candidate.yml` file.
- The active services (`gaia-core`, `gaia-web`, `gaia-study`) start up successfully.
- The `gaia-mcp-candidate` service, however, fails to start.

### Where We're Stuck
The project is currently blocked by a persistent `ModuleNotFoundError` in the `gaia-mcp` and `gaia-mcp-candidate` services.

**The Error:**
`ModuleNotFoundError: No module named 'gaia_core'`

This error indicates that the `gaia_core` module is not available in the Python path of the `gaia-mcp` containers. This is happening despite `gaia_core` being installed as a dependency.

**Troubleshooting Steps Taken:**
I have made numerous attempts to resolve this issue, including:
1.  **Creating `setup.py` files:** I created `setup.py` files for `gaia-core`, `gaia-web`, and `gaia-mcp` to define them as proper Python packages.
2.  **Editable Installs:** I modified the Dockerfiles for all services to use `pip install -e .` to install the packages in editable mode. This should have made the packages discoverable on the `PYTHONPATH`.
3.  **Dockerfile Modifications:** I have tried various combinations of `COPY`, `WORKDIR`, and `ENV PYTHONPATH` in the Dockerfiles to ensure that the source code is correctly placed and accessible within the containers.
4.  **Docker Pruning:** I have performed a comprehensive prune of the Docker system (`docker system prune -a --volumes`) to eliminate any potential caching issues.

Despite these efforts, the `ModuleNotFoundError` persists. This suggests a fundamental issue with the Docker build process, the Python environment within the containers, or the interaction between `pip`, `uvicorn`, and the way the services are structured, which I have been unable to resolve.

The immediate next step is to seek guidance on the correct way to structure and build these services to ensure that the dependencies are correctly resolved at runtime.

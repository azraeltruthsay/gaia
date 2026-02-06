**Date:** 2026-02-02
**Title:** Debugging `ModuleNotFoundError` in Decoupled GAIA Services

**Problem:**

After refactoring the GAIA codebase to decouple services and move shared logic to a `gaia-common` library, the `gaia-mcp-candidate` and `gaia-web-candidate` services are failing to start. The error is a `ModuleNotFoundError: No module named 'gaia_common.config'`, which indicates that the Python interpreter cannot find the `gaia-common` package.

**Context:**

The goal of the refactoring is to remove the direct dependency of `gaia-mcp` on `gaia-core`, as outlined in `knowledge/Dev_Notebook/SOA-decoupled-proposal.md`. The refactoring was performed in the `candidates/` directory to allow for isolated testing.

**Attempted Solutions:**

1.  **Verified File Locations:** Confirmed that the `gaia-common` package exists and that the `config.py` file is in the correct location (`candidates/gaia-common/gaia_common/config.py`).
2.  **Corrected Dockerfile Paths:** Updated `docker-compose.candidate.yml` to point to the correct Dockerfiles in the `candidates/` subdirectories.
3.  **Set `PYTHONPATH` in `docker-compose.yml`:** Added the `PYTHONPATH` environment variable to the `gaia-mcp-candidate`, `gaia-core-candidate`, `gaia-web-candidate`, and `gaia-study-candidate` services to explicitly include `/app` and `/gaia-common`. This did not resolve the issue.
4.  **Mounted `gaia-common` directly into `/app`:** Modified the `volumes` in `docker-compose.candidate.yml` to mount `gaia-common` to `/app/gaia-common`. This also did not resolve the issue.
5.  **Added `__init__.py` files:** Created `__init__.py` files in `gaia-common/` and `candidates/gaia-common/` to ensure they are treated as packages. This did not resolve the issue.

**Current Status:**

The `gaia-core-candidate` and `gaia-study-candidate` services are starting correctly, but `gaia-mcp-candidate` and `gaia-web-candidate` are still failing with the `ModuleNotFoundError`.

**Next Steps:**

The issue seems to be related to how the Python path is being resolved within the containers, especially in the context of `pip install -e` and volume mounts.

My next step will be to simplify the Dockerfiles even further and try to build the containers in a way that avoids these conflicts. I will also try to run the services with a shell command inside the container to manually inspect the `PYTHONPATH` and the file system.

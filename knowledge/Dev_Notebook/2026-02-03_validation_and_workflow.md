**Date:** 2026-02-03
**Title:** Full-Stack Validation of Live/Candidate Container Architecture

## Summary

Following the recent decoupling of services, a full-stack validation was performed to ensure the integrity of the live, candidate, and mixed-mode container architectures. The initial test of the live stack failed due to a misconfiguration in the `gaia-mcp` service. After a diagnosis and fix, all validation tests passed successfully, confirming the robustness of the new container management system.

## Initial Failure and Debugging

The validation process began with an attempt to start the full live stack (`./gaia.sh live start`). This failed immediately, with the `gaia-mcp` container reporting as unhealthy.

**Investigation revealed two root causes:**

1.  **Stale `gaia-common` Dependency:** The `gaia-mcp` service depended on code in `gaia-common` that had been updated in the candidate environment but not yet promoted to live. Specifically, `gaia-mcp` was trying to import `gaia_common.config`, which did not exist in the live `gaia-common` directory.
2.  **Incorrect Dockerfile Promotion:** The `gaia-mcp/Dockerfile` itself contained hardcoded `COPY` instructions pointing to the `candidates/` directory. When the `mcp` candidate was promoted previously, this incorrect Dockerfile was copied into the live service directory, causing the build process to continue using stale candidate code even after a promotion.

## The Fix

The issues were resolved with a two-part fix:

1.  **Promoted `gaia-common`:** The `candidates/gaia-common` directory was manually promoted to `gaia-common` using `rsync` to ensure the live services had access to the latest shared code.
2.  **Corrected `gaia-mcp` Dockerfile:** The `gaia-mcp/Dockerfile` was modified to remove the hardcoded `candidates/` paths, changing the `COPY` instructions to source from the correct `gaia-mcp/` and `gaia-common/` live directories.

After applying these fixes and forcing a Docker rebuild (`docker compose up --build`), the `gaia-mcp` container started successfully.

## Validation Test Results

With the `gaia-mcp` service fixed, the full validation plan was executed.

| Test Scenario | Command(s) | Result |
| :--- | :--- | :--- |
| **1. Live Stack** | `./gaia.sh live start` | ✅ **PASS** |
| **2. Candidate Stack** | `./gaia.sh candidate start` | ✅ **PASS** |
| **3. Mixed Stack** | `./gaia.sh live start`<br>`./gaia.sh swap mcp candidate` | ✅ **PASS** |

All services reported as healthy in all three configurations. The tests confirm that the service-oriented architecture is functioning as designed, and the `gaia.sh` script can correctly manage the lifecycle and interaction of live and candidate containers.

## Cleanup

All live and candidate containers were successfully shut down after the validation was complete.

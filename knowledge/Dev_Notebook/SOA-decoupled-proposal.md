Technical Proposal: Decoupling GAIA-MCP via Shared Common Library
Date: February 2, 2026 Priority: High (Blocker for Candidate Testing Infrastructure) Target System: gaia-mcp (The Hands) and gaia-core (The Brain)
1. Executive Summary
Currently, the gaia-mcp service fails to start in candidate environments because it attempts to import Python modules (gaia_core.*) that are not persistently available in its container. To resolve this, gaia-mcp currently relies on copying the entire gaia-core codebase into itself during the build—a fragile practice that breaks when development volumes are mounted.
This proposal outlines the refactoring of shared logic (Configuration, Safe Execution, and World State) into the existing gaia-common library. This will allow gaia-mcp to operate as a standalone service that communicates with Core exclusively via the JSON-RPC network layer, eliminating the need to physically copy Core source code.
2. Problem Analysis
• Current State: gaia-mcp imports gaia_core.config, gaia_core.utils.gaia_rescue_helper, and gaia_core.utils.world_state to function.
• The Conflict: The Dockerfile copies gaia-core/ to /app/gaia-core to satisfy these imports. However, docker-compose.yml mounts the host directory to /app, obscuring the copied gaia-core folder at runtime, causing ModuleNotFoundError.
• Architectural Violation: "The Hands" (MCP) should not contain the source code of "The Brain" (Core). They should share a language (gaia-common) and communicate over a network (JSON-RPC).
3. Implementation Plan
Phase 1: Promote Shared Logic to gaia-common
We must move the logic that both services need into the shared library.
1. Move Configuration & Constants
• Action: Move gaia_constants.json and the Config loading logic from gaia-core to gaia-common.
• Source: gaia_core/config.py and gaia_core/gaia_constants.json.
• Destination: gaia_common/config.py and gaia_common/constants/gaia_constants.json.
• Why: Both the Brain (Core) and the Hands (MCP) need to know the system defaults (e.g., SAFE_EXECUTE_FUNCTIONS, MCP_LITE_ENDPOINT).
2. Refactor GAIARescueHelper (Split Dispatcher vs. Executor)
• Analysis: Currently, GAIARescueHelper mixes dispatching actions (calling mcp_client) with executing actions (running subprocess).
• Action: Extract the execution primitives into gaia-common.
    ◦ Create gaia_common.utils.safe_execution.py.
    ◦ Move run_shell_safe logic (whitelist checking and subprocess calls) and file I/O safeguards here.
• Impact: gaia-mcp imports this to execute tools locally. gaia-core keeps GAIARescueHelper but uses it solely to dispatch requests to the MCP.
3. Generalize World State
• Action: Move gaia_core.utils.world_state to gaia_common.utils.world_state.
• Refactor: The world_state module currently imports tools_registry (already in common) and checks environment variables. It is generic enough to live in gaia-common.
Phase 2: Refactor gaia-mcp (The Hands)
Update the MCP service to rely solely on gaia-common.
1. Update Imports in server.py and tools.py
• Change from gaia_core.config import Config to from gaia_common.config import Config.
• Change from gaia_core.utils.gaia_rescue_helper import GAIARescueHelper to from gaia_common.utils.safe_execution import SafeExecutor.
• Change from gaia_core.utils.world_state import world_state_detail to from gaia_common.utils.world_state import world_state_detail.
2. Clean the Dockerfile
• Remove: COPY gaia-core/ /app/gaia-core/ and RUN pip install -e /app/gaia-core/.
• Verify: Ensure gaia-common is installed (already present).
Phase 3: Refactor gaia-core (The Brain)
Update the Core to consume the new Common locations.
• Update gaia_core/config.py to inherit or utilize gaia_common.config.
• Update agent_core.py and gaia_rescue.py to import constants and helper logic from the new Common paths.
4. Why This Approach?
1. Eliminates Circular Dependencies: gaia-mcp will no longer crash because it cannot find gaia-core. It will self-contain all logic needed to execute tools via gaia-common.
2. Fixes Volume Mount Issue: Since gaia-common is installed in a separate path (/gaia-common or /libs/gaia-common), mounting development code into /app will no longer delete the dependencies.
3. Correct Abstraction: This enforces the "Hands vs. Brain" separation. The Hands (MCP) provide capabilities (defined in Common); the Brain (Core) provides intent (sent via Network).
5. Success Criteria
• gaia-mcp-candidate builds and starts without ModuleNotFoundError.
• gaia-mcp can execute run_shell and world_state without importing gaia-core.
• gaia-core can successfully send a JSON-RPC request to gaia-mcp to trigger a tool.
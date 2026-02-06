"""
GAIA MCP-Lite Server - Entry Point

Exposes GAIA's tools and primitives over a local JSON-RPC 2.0 interface.
This provides a secure and audited boundary between cognition and action.

Usage:
    uvicorn gaia_mcp.main:app --host 0.0.0.0 --port 8765
"""

import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from gaia_common.utils import setup_logging, get_logger, install_health_check_filter

from .server import create_app
from .approval import ApprovalStore

# Initialize logging
log_dir = os.getenv("GAIA_LOG_DIR", "/var/log/gaia")
log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
setup_logging(log_dir=log_dir, level=log_level, service_name="gaia-mcp")
install_health_check_filter()

logger = get_logger(__name__)

# Global approval store for pending sensitive actions
approval_store = ApprovalStore(
    ttl_seconds=int(os.getenv("MCP_APPROVAL_TTL", "900"))
)

# Create the FastAPI application instance
# We'll pass the approval_store to create_app if it needs it directly.
# For now, it's globally available for server.py to access if needed (less ideal).
# Let's modify create_app to accept approval_store
app = create_app()

# Now update the endpoints in `app` to use the global `approval_store`
# This requires a more complex interaction, as FastAPI endpoints usually capture
# dependencies on definition.

# Re-define endpoints or inject approval_store differently
# For now, I will manually inject the approval_store into the app's state,
# and ensure server.py uses `request.app.state.approval_store`
app.state.approval_store = approval_store

# Set MCP_BYPASS flag in app state
app.state.mcp_bypass = os.getenv("GAIA_MCP_BYPASS", "false").lower() in ("1", "true", "yes")
app.state.sensitive_tools = {"ai_write", "write_file", "run_shell", "memory_rebuild_index"}


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    logger.info("gaia-mcp starting up...")
    logger.info(f"Sandbox root: {os.getenv('SANDBOX_ROOT', '/sandbox')}")
    logger.info(f"Approval TTL: {app.state.approval_store._ttl}s")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("gaia-mcp shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "gaia_mcp.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8765")),
        reload=os.getenv("GAIA_ENV", "production") == "development",
    )

"""
GAIA MCP-Lite Server

Exposes GAIA's tools and primitives over a local JSON-RPC 2.0 interface.
This provides a secure and audited boundary between cognition and action.
"""

import logging
import traceback
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from gaia_common.config import Config
from gaia_common.utils.gaia_rescue_helper import GAIARescueHelper
from .approval import ApprovalStore
from .notebooklm_tools import _close_client as _close_notebooklm_client
from .tools import execute_limb, TOOLS, SENSITIVE_TOOLS
from gaia_common.utils.domain_tools import DOMAIN_TOOLS

import os

# --- Configuration & Initialization ---

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GAIA.MCPServer")

try:
    from gaia_common.utils.error_logging import log_gaia_error
    from gaia_common.errors import lookup as _lookup_error
except ImportError:
    def log_gaia_error(lgr, code, detail="", **kw):
        lgr.error("[%s] %s", code, detail)
    def _lookup_error(code):
        return None

config = Config()
gaia_helper = GAIARescueHelper(config)
approval_store = ApprovalStore()

# Opt-in bypass for approval flow (useful for local testing)
# Time-limited: bypass expires after 1 hour if set via env var.
# This prevents forgotten bypass flags from persisting indefinitely.
_BYPASS_ENV = os.getenv("GAIA_MCP_BYPASS", "false").lower() in ("1", "true", "yes")
_BYPASS_EXPIRES = None
if _BYPASS_ENV:
    import time as _bypass_time
    _BYPASS_EXPIRES = _bypass_time.time() + 3600  # 1 hour
    logger.warning("MCP_BYPASS active — will auto-expire in 1 hour")

def _is_bypass_active() -> bool:
    if not _BYPASS_ENV:
        return False
    if _BYPASS_EXPIRES and _bypass_time.time() > _BYPASS_EXPIRES:
        logger.warning("MCP_BYPASS has EXPIRED (1 hour limit reached)")
        return False
    return True

MCP_BYPASS = _is_bypass_active  # Now a callable, not a static bool

app = FastAPI(
    title="GAIA MCP-Lite Server",
    description="Provides a secure JSON-RPC interface to GAIA's tools.",
    version="0.1.0"
)

# Inter-service HMAC authentication
try:
    from gaia_common.utils.service_auth import AuthMiddleware
    if AuthMiddleware:
        app.add_middleware(AuthMiddleware)
except ImportError:
    pass  # gaia_common not available — skip auth


@app.on_event("shutdown")
async def _shutdown():
    """Clean up async clients on shutdown."""
    await _close_notebooklm_client()


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy", "service": "gaia-mcp"}


def create_app() -> FastAPI:
    """Factory function to return the configured FastAPI application."""
    return app


# --- JSON-RPC Endpoint ---

@app.post("/jsonrpc")
async def jsonrpc_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None}, status_code=400)

    request_id = body.get("id")

    # Basic JSON-RPC validation
    if body.get("jsonrpc") != "2.0" or "method" not in body:
        return JSONResponse(content={"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}, "id": request_id}, status_code=400)

    method = body["method"]
    params = body.get("params", {})

    # Enforce approval for sensitive legacy tools unless bypass is enabled.
    # Domain tools (file, shell, etc.) handle sensitivity per-action inside execute_limb.
    if (not MCP_BYPASS()) and method in SENSITIVE_TOOLS and method not in DOMAIN_TOOLS:
        return JSONResponse(content={"jsonrpc": "2.0", "error": {"code": -32001, "message": f"'{method}' requires approval. Use /request_approval first."}, "id": request_id}, status_code=403)

    # Validate params against the tool's schema (skip for domain tools — they validate internally)
    if method in TOOLS and method not in DOMAIN_TOOLS:
        from jsonschema import validate, ValidationError
        try:
            validate(instance=params, schema=TOOLS[method]["params"])
        except ValidationError as e:
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "error": {"code": -32602, "message": f"Invalid params: {e.message}"},
                "id": request_id
            }, status_code=400)

    # Dispatch the tool call via the authoritative tools module
    try:
        # Note: server.py handles its own security checks (sensitive tools & blast shield) 
        # but we also pass the approval_store to execute_limb for consistent internal checks.
        result = await execute_limb(
            method=method, 
            params=params, 
            approval_store=approval_store,
            pre_approved=MCP_BYPASS() # Bypass checks if MCP_BYPASS is active
        )
        
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "result": result,
            "id": request_id
        })
    except PermissionError as e:
        gaia_code = "GAIA-MCP-010" if "blast shield" in str(e).lower() else "GAIA-MCP-015"
        defn = _lookup_error(gaia_code)
        log_gaia_error(logger, gaia_code, str(e))
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "error": {
                "code": -32002,
                "message": str(e),
                "data": {
                    "gaia_code": gaia_code,
                    "hint": defn.hint if defn else "",
                    "errorCategory": defn.category.value if defn else "safety",
                    "isRetryable": defn.is_retryable if defn else False,
                },
            },
            "id": request_id
        }, status_code=403)
    except Exception as e:
        log_gaia_error(logger, "GAIA-MCP-020", f"Tool '{method}': {e}", exc_info=True)
        defn = _lookup_error("GAIA-MCP-020")
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "error": {
                "code": -32603,
                "message": f"Internal error: {e}",
                "data": {
                    "gaia_code": "GAIA-MCP-020",
                    "hint": defn.hint if defn else "",
                    "errorCategory": defn.category.value if defn else "internal",
                    "isRetryable": defn.is_retryable if defn else False,
                    "traceback": traceback.format_exc(),
                },
            },
            "id": request_id
        }, status_code=500)


@app.post("/request_approval")
async def request_approval_endpoint(request: Request):
    """
    Endpoint to request human approval for a sensitive tool call.
    Returns an action_id and a challenge.
    """
    try:
        body = await request.json()
        method = body.get("method")
        params = body.get("params", {})
        
        if not method:
            raise HTTPException(status_code=400, detail="method is required")
            
        # 1. Create a pending action
        action_id, challenge = approval_store.create_request(method, params)
        
        # 2. Log for the user
        logger.warning(f"🚨 APPROVAL REQUIRED: '{method}' (ID: {action_id})")
        logger.warning(f"   Params: {params}")
        logger.warning(f"   Challenge: {challenge}")
        
        return {
            "ok": True,
            "action_id": action_id,
            "challenge": challenge,
            "message": f"Approval required for '{method}'. Please approve via /approve_action."
        }
    except Exception as e:
        logger.error(f"Error creating approval request: {e}")
        return JSONResponse(content={"ok": False, "error": str(e)}, status_code=500)


@app.post("/approve_action")
async def approve_action_endpoint(request: Request):
    """
    Endpoint to approve a pending action using the challenge response.
    If approved, the tool is executed and results returned.
    """
    try:
        body = await request.json()
        action_id = body.get("action_id")
        approval_response = body.get("approval")
        
        if not action_id or not approval_response:
            raise HTTPException(status_code=400, detail="action_id and approval are required")
            
        # 1. Validate the approval
        pending = approval_store.validate_approval(action_id, approval_response)
        if not pending:
            return JSONResponse(content={"ok": False, "error": "Invalid action_id or incorrect challenge response."}, status_code=403)
            
        # 2. Execute the tool (since it's now approved)
        method = pending["method"]
        params = pending["params"]
        
        logger.info(f"✅ ACTION APPROVED: '{method}' (ID: {action_id})")
        
        result = await execute_limb(
            method=method, 
            params=params, 
            approval_store=approval_store,
            pre_approved=True # Explicitly approved
        )
        
        # 3. Mark as completed
        approval_store.clear_request(action_id)
        
        return {
            "ok": True,
            "method": method,
            "result": result
        }
    except Exception as e:
        logger.error(f"Error approving action: {e}", exc_info=True)
        return JSONResponse(content={"ok": False, "error": str(e)}, status_code=500)


@app.get("/")
def read_root():
    return {"message": "GAIA MCP-Lite Server is running."}

# app/utils/mcp_client.py
"""
GAIA MCP-Lite Client

This module is responsible for dispatching sidecar actions from a CognitionPacket
to the MCP-lite server.
"""

import logging
import os
import requests
from datetime import datetime, timezone
from typing import List, Dict, Any

from gaia_common.protocols.cognition_packet import CognitionPacket
from gaia_core.config import Config

logger = logging.getLogger("GAIA.MCPClient")

# Low-level JSON-RPC call
def _normalize_endpoint(ep: str) -> str:
    """Ensure the base endpoint includes /jsonrpc for RPC calls."""
    if not ep:
        return ep
    if ep.endswith("/jsonrpc"):
        return ep
    # If it already ends with a path (e.g., /approve_action), leave as-is for non-RPC callers.
    if ep.endswith("/approve_action") or ep.endswith("/request_approval") or ep.endswith("/pending_approvals"):
        return ep
    if ep.endswith("/"):
        return ep + "jsonrpc"
    return ep + "/jsonrpc"


def call_jsonrpc(method: str, params: Dict, endpoint: str = None, timeout: int = 20) -> Dict:
    ep_raw = endpoint or os.getenv("MCP_LITE_ENDPOINT") or Config().constants.get("MCP_LITE_ENDPOINT")
    ep = _normalize_endpoint(ep_raw)
    if not ep:
        return {"ok": False, "error": "MCP endpoint not configured"}
    payload = {"jsonrpc": "2.0", "method": method, "params": params or {}, "id": datetime.now(timezone.utc).isoformat()}
    try:
        r = requests.post(ep, json=payload, timeout=timeout)
        if r.status_code == 403:
            # Sensitive tool — route through approval flow with auto-pending
            logger.info(f"call_jsonrpc: '{method}' requires approval (403). Requesting auto-approval.")
            approval_result = request_approval_via_mcp(
                method=method,
                params={**(params or {}), "_allow_pending": True}
            )
            if approval_result.get("ok"):
                # Auto-approved (MCP_BYPASS=true) — result is in the response
                return {"ok": True, "response": approval_result.get("result", approval_result)}
            elif approval_result.get("action_id"):
                # Pending approval — return info so caller can handle it
                return {
                    "ok": False,
                    "pending_approval": True,
                    "action_id": approval_result.get("action_id"),
                    "challenge": approval_result.get("challenge"),
                    "error": f"'{method}' requires approval. Challenge: {approval_result.get('challenge')}"
                }
            else:
                return {"ok": False, "error": f"Approval request failed: {approval_result.get('error', 'unknown')}"}
        r.raise_for_status()
        return {"ok": True, "response": r.json()}
    except Exception as e:
        logger.error(f"[{datetime.now(timezone.utc).isoformat()}] call_jsonrpc failed: {e}")
        try:
            logger.error(f"[{datetime.now(timezone.utc).isoformat()}] full error response: {r.json()}")
        except Exception:
            pass
        return {"ok": False, "error": str(e)}

def dispatch_sidecar_actions(packet: CognitionPacket, config: Config) -> List[Dict]:
    """
    Dispatches all sidecar actions in a packet to the MCP-lite server.

    Args:
        packet: The CognitionPacket containing the actions to dispatch.
        config: The application configuration.

    Returns:
        A list of results from the server for each action.
    """
    if not config.constants.get("MCP_LITE_ENABLED") or not packet.response.sidecar_actions:
        return []

    endpoint = config.constants.get("MCP_LITE_ENDPOINT")
    if not endpoint:
        logger.error("MCP_LITE_ENDPOINT is not configured. Cannot dispatch actions.")
        return []

    results = []
    for i, action in enumerate(packet.response.sidecar_actions):
        request_id = f"{packet.header.packet_id}-{i}"

        # Defensive: ensure params is a dict
        params: Dict[str, Any] = action.params or {}
        if not isinstance(params, dict):
            logger.warning(f"[{datetime.now(timezone.utc).isoformat()}] action.params is not a dict, coercing to {{}} for request {request_id}")
            params = {}

        # Allow optional model/token options to be passed without breaking older callers.
        # Accept either top-level keys 'token_options' or 'model_options' inside params.
        token_opts = None
        if "token_options" in params:
            token_opts = params.get("token_options")
        elif "model_options" in params:
            token_opts = params.get("model_options")

        # Build RPC params while keeping original params safe
        rpc_params = dict(params)
        if token_opts is not None:
            # Only forward if it's a mapping; otherwise ignore to avoid sending invalid data
            if isinstance(token_opts, dict):
                rpc_params["token_options"] = token_opts
            else:
                logger.warning(f"[{datetime.now(timezone.utc).isoformat()}] Ignoring non-dict token_options for request {request_id}")

        payload = {
            "jsonrpc": "2.0",
            "method": action.action_type,
            "params": rpc_params,
            "id": request_id
        }

        ts = datetime.now(timezone.utc).isoformat()
        logger.info(f"[{ts}] Dispatching action to MCP server: method={action.action_type} id={request_id} params_keys={list(rpc_params.keys())}")

        try:
            response = requests.post(endpoint, json=payload, timeout=20)
            if response.status_code == 403:
                # Sensitive tool — route through approval flow with auto-pending
                logger.info(f"[{ts}] Action '{action.action_type}' requires approval (403). Routing to approval flow.")
                approval_result = request_approval_via_mcp(
                    method=action.action_type,
                    params={**rpc_params, "_allow_pending": True}
                )
                results.append({
                    "id": request_id,
                    "dispatched_at": ts,
                    "pending_approval": True,
                    "action_id": approval_result.get("action_id"),
                    "challenge": approval_result.get("challenge"),
                    "proposal": approval_result.get("proposal"),
                })
                continue
            response.raise_for_status()
            try:
                data = response.json()
            except Exception:
                data = {"raw": response.text}
            results.append({"id": request_id, "dispatched_at": ts, "response": data})
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to dispatch action '{action.action_type}' to MCP server: {e}"
            logger.error(f"[{datetime.now(timezone.utc).isoformat()}] {error_msg}")
            results.append({
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": error_msg},
                "id": request_id,
                "dispatched_at": ts,
            })
    
    return results


## --- High-level MCP primitives used by the codebase ---------------------
def ai_read(path: str) -> Dict:
    """Read a file via the MCP abstraction. Returns a dict with fields (ok, content, error)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
        ts = datetime.now(timezone.utc).isoformat()
        logger.info(f"[{ts}] MCP.ai_read: read {path} ({len(content)} bytes)")
        return {"ok": True, "op": "ai.read", "path": path, "content": content, "read_at": ts}
    except Exception as e:
        logger.error(f"[{datetime.now(timezone.utc).isoformat()}] MCP.ai_read failed for {path}: {e}")
        return {"ok": False, "op": "ai.read", "path": path, "error": str(e)}


_WRITABLE_DIRS = ("/sandbox/", "/shared/", "/knowledge/", "/tmp/", "/logs/")

def ai_write(path: str, content: str) -> Dict:
    """Write a file via the MCP abstraction. Returns a dict with ok and metadata.

    Security: Writes are restricted to allowed directories to prevent
    arbitrary filesystem modification.
    """
    resolved = os.path.realpath(path)
    if not any(resolved.startswith(d) for d in _WRITABLE_DIRS):
        logger.warning(f"MCP.ai_write blocked: {path} resolves to {resolved} (outside allowed dirs)")
        return {"ok": False, "op": "ai.write", "path": path, "error": f"Write blocked: path outside allowed directories"}
    try:
        dirname = os.path.dirname(resolved)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as fh:
            fh.write(content)
        ts = datetime.now(timezone.utc).isoformat()
        logger.info(f"[{ts}] MCP.ai_write: wrote {resolved} ({len(content)} bytes)")
        return {"ok": True, "op": "ai.write", "path": resolved, "bytes": len(content), "written_at": ts}
    except Exception as e:
        logger.error(f"[{datetime.now(timezone.utc).isoformat()}] MCP.ai_write failed for {resolved}: {e}")
        return {"ok": False, "op": "ai.write", "path": resolved, "error": str(e)}


def ai_execute(command: str, timeout: int = 30, shell: bool = False, dry_run: bool = False) -> Dict:
    """Execute a shell command via MCP. Returns dict with stdout/stderr/returncode.

    Note: Use with caution. The MCP layer can enforce dry_run or safety checks at a later stage.
    """
    import subprocess
    import shlex

    if dry_run:
        logger.warning(f"[{datetime.now(timezone.utc).isoformat()}] MCP.ai_execute (dry_run): {command}")
        return {"ok": True, "op": "ai.execute", "command": command, "dry_run": True, "stdout": "", "stderr": ""}

    try:
        cmd = command if shell else shlex.split(command)
        res = subprocess.run(cmd, shell=shell, check=False, capture_output=True, text=True, timeout=timeout)
        ts = datetime.now(timezone.utc).isoformat()
        logger.info(f"[{ts}] MCP.ai_execute: ran command (rc={res.returncode}) command={command}")
        return {"ok": True, "op": "ai.execute", "command": command, "returncode": res.returncode, "stdout": res.stdout, "stderr": res.stderr, "executed_at": ts}
    except subprocess.TimeoutExpired as e:
        logger.error(f"[{datetime.now(timezone.utc).isoformat()}] MCP.ai_execute timeout: {e}")
        return {"ok": False, "op": "ai.execute", "command": command, "error": "timeout", "detail": str(e)}
    except Exception as e:
        logger.error(f"[{datetime.now(timezone.utc).isoformat()}] MCP.ai_execute failed: {e}")
        return {"ok": False, "op": "ai.execute", "command": command, "error": str(e)}


def embedding_query(query: str, top_k: int = 5, knowledge_base_name: str = "system") -> Dict:
    """Proxy to the vector indexer for embeddings/nearest neighbor queries.

    Returns a dict with `ok` and `results` list. If vector_indexer is unavailable, returns error.
    """
    try:
        from gaia_common.utils.vector_indexer import VectorIndexer
        vi = VectorIndexer.instance(knowledge_base_name)
        results = vi.query(query, top_k=top_k)
        ts = datetime.now(timezone.utc).isoformat()
        logger.info(f"[{ts}] MCP.embedding_query: got {len(results)} results for query")
        return {"ok": True, "op": "embedding.query", "query": query, "results": results, "queried_at": ts}
    except Exception as e:
        logger.error(f"[{datetime.now(timezone.utc).isoformat()}] MCP.embedding_query failed: {e}")
        return {"ok": False, "op": "embedding.query", "query": query, "error": str(e)}


## --- Approval helpers (client-side) -------------------------------------
def request_approval_via_mcp(method: str, params: Dict) -> Dict:
    """Ask the MCP server to create a pending action requiring human approval.

    Returns: {"ok": True, "action_id": str, "challenge": str} or error dict
    """
    # Prefer explicit environment override so containers can target the mcp sidecar by name
    endpoint = _normalize_endpoint(os.getenv("MCP_LITE_ENDPOINT") or Config().constants.get("MCP_LITE_ENDPOINT"))
    if not endpoint:
        logger.error("MCP endpoint not configured for approval request")
        return {"ok": False, "error": "no endpoint"}

    url = endpoint.replace('/jsonrpc', '/request_approval')
    try:
        # Allow callers to pass a special _allow_pending param inside params to opt-in
        allow_pending = False
        if isinstance(params, dict) and params.get("_allow_pending") is True:
            allow_pending = True
            # Remove internal flag when sending to server body top-level for clarity
            params = dict(params)
            params.pop("_allow_pending", None)

        payload = {"method": method, "params": params}
        if allow_pending:
            payload["allow_pending"] = True

        logger.info(f"Requesting MCP approval: url={url} method={method} allow_pending={allow_pending}")
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        # Include any human-friendly proposal text and timestamps if provided by server
        return {
            "ok": True,
            "action_id": data.get("action_id"),
            "challenge": data.get("challenge"),
            "proposal": data.get("proposal"),
            "created_at": data.get("created_at"),
            "expiry": data.get("expiry"),
        }
    except Exception as e:
        logger.error(f"Failed to request approval via MCP: {e}")
        return {"ok": False, "error": str(e)}


def approve_action_via_mcp(action_id: str, approval: str) -> Dict:
    """Submit approval string to the MCP server to execute a pending action.

    Returns the execution result dict on success.
    """
    endpoint = _normalize_endpoint(os.getenv("MCP_LITE_ENDPOINT") or Config().constants.get("MCP_LITE_ENDPOINT"))
    if not endpoint:
        logger.error("MCP endpoint not configured for approval submit")
        return {"ok": False, "error": "no endpoint"}

    url = endpoint.replace('/jsonrpc', '/approve_action')
    try:
        logger.info(f"Submitting MCP approval: url={url} action_id={action_id}")
        r = requests.post(url, json={"action_id": action_id, "approval": approval}, timeout=10)
        r.raise_for_status()
        return {"ok": True, "result": r.json()}
    except Exception as e:
        logger.error(f"Failed to submit approval via MCP: {e}")
        return {"ok": False, "error": str(e)}


def get_pending_action(action_id: str) -> Dict:
    """Fetch pending approvals list from MCP and return the entry matching action_id, or None."""
    endpoint = _normalize_endpoint(os.getenv("MCP_LITE_ENDPOINT") or Config().constants.get("MCP_LITE_ENDPOINT"))
    if not endpoint:
        logger.error("MCP endpoint not configured for getting pending approvals")
        return {"ok": False, "error": "no endpoint"}

    url = endpoint.replace('/jsonrpc', '/pending_approvals')
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
        pending = data.get("pending") or []
        for p in pending:
            if p.get("action_id") == action_id:
                return {"ok": True, "entry": p}
        return {"ok": False, "error": "not found"}
    except Exception as e:
        logger.error(f"Failed to fetch pending approvals: {e}")
        return {"ok": False, "error": str(e)}


def discover(endpoint: str = None, timeout: int = 3) -> Dict:
    """Structured discovery of MCP capabilities.

    Tries (best-effort) several discovery approaches and returns a concise
    dict: {ok: bool, methods: [name,...], raw: <short snippet>, error: <msg>}
    """
    try:
        if endpoint is None:
            endpoint = os.getenv("MCP_LITE_ENDPOINT") or Config().constants.get("MCP_LITE_ENDPOINT")
        if not endpoint:
            return {"ok": False, "error": "no endpoint configured"}

        # 1) Try JSON-RPC methods that the MCP server exposes. The MCP server
        # implemented in this repo exposes a 'list_tools' JSON-RPC method which
        # returns an array of tool names. Try a few common variants and
        # normalize the response from either a top-level list or a {'result': ...}
        try:
            jsonrpc_candidates = ["list_tools", "rpc.discover", "system.listMethods", "list_methods"]
            for jmethod in jsonrpc_candidates:
                payload = {"jsonrpc": "2.0", "method": jmethod, "params": {}, "id": f"discover-{jmethod}"}
                try:
                    r = requests.post(endpoint, json=payload, timeout=timeout)
                except Exception:
                    r = None
                if not r:
                    continue
                if not r.ok:
                    # continue trying other candidate methods
                    continue
                try:
                    data = r.json()
                except Exception:
                    # not JSON, skip
                    continue

                # Normalize common JSON-RPC shapes: {'result': [...] } or top-level list/object
                result = None
                if isinstance(data, dict) and 'result' in data:
                    result = data['result']
                else:
                    result = data

                methods = []
                if isinstance(result, list):
                    methods = [str(x) for x in result]
                elif isinstance(result, dict):
                    # look for a 'methods' key or fall back to dict keys
                    if 'methods' in result and isinstance(result['methods'], list):
                        methods = [m.get('name') if isinstance(m, dict) else str(m) for m in result['methods']]
                    else:
                        # maybe it's a mapping of tool->metadata; return the keys
                        methods = list(result.keys())

                return {"ok": True, "methods": methods, "raw": str(data)[:400], "endpoint": endpoint, "jsonrpc_method": jmethod}
        except Exception:
            pass

        # If initial attempts failed, try common service hostnames / port fallbacks.
        try_hosts = []
        try:
            # Normalize endpoint to base URL
            base = endpoint
            if base.endswith('/jsonrpc'):
                base = base[:-8]
            # Try service name used in docker-compose
            try_hosts.append(base)
            # Common replacements
            if 'localhost' in base or '127.0.0.1' in base:
                try_hosts.append(base.replace('localhost', 'gaia-mcp-lite'))
                try_hosts.append(base.replace('127.0.0.1', 'gaia-mcp-lite'))
            # try common docker-compose service name directly
            try_hosts.append('http://gaia-mcp-lite:4141')
            # try with explicit /jsonrpc appended
            try_hosts = list(dict.fromkeys(try_hosts))
        except Exception:
            try_hosts = []

        for host in try_hosts:
            for suffix in ['/jsonrpc', '/capabilities', '/methods', '/request_approval']:
                url = host + suffix
                try:
                    r = requests.post(url, json={"jsonrpc": "2.0", "method": "rpc.discover", "params": {}, "id": "discover"}, timeout=timeout) if suffix == '/jsonrpc' else requests.get(url, timeout=timeout)
                    if r.ok:
                        try:
                            payload = r.json()
                            if isinstance(payload, dict) and 'methods' in payload and isinstance(payload['methods'], list):
                                methods = [m.get('name') if isinstance(m, dict) else str(m) for m in payload['methods']]
                                return {"ok": True, "methods": methods, "raw": str(payload)[:400], "endpoint": url}
                            elif isinstance(payload, dict):
                                return {"ok": True, "methods": list(payload.keys()), "raw": str(payload)[:400], "endpoint": url}
                            elif isinstance(payload, list):
                                return {"ok": True, "methods": [str(x) for x in payload], "raw": str(payload)[:400], "endpoint": url}
                        except Exception:
                            return {"ok": True, "methods": [], "raw": r.text[:400], "endpoint": url}
                except Exception:
                    continue

        # 2) Try REST-like discovery endpoints
        for p in ["/capabilities", "/methods", "/jsonrpc"]:
            try:
                url = endpoint.replace('/jsonrpc', p)
                r = requests.get(url, timeout=timeout)
                if r.ok:
                    try:
                        payload = r.json()
                        if isinstance(payload, dict):
                            if 'methods' in payload and isinstance(payload['methods'], list):
                                methods = [m.get('name') if isinstance(m, dict) else str(m) for m in payload['methods']]
                                return {"ok": True, "methods": methods, "raw": str(payload)[:400]}
                            else:
                                return {"ok": True, "methods": list(payload.keys()), "raw": str(payload)[:400]}
                        elif isinstance(payload, list):
                            return {"ok": True, "methods": [str(x) for x in payload], "raw": str(payload)[:400]}
                    except Exception:
                        return {"ok": True, "methods": [], "raw": r.text[:400]}
            except Exception:
                continue

        return {"ok": False, "error": "no discovery info"}
    except Exception as e:
        logger.error(f"MCP discovery failed: {e}")
        return {"ok": False, "error": str(e)}
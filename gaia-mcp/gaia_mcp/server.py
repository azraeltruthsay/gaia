"""
GAIA MCP-Lite Server

Exposes GAIA's tools and primitives over a local JSON-RPC 2.0 interface.
This provides a secure and audited boundary between cognition and action.
"""

import logging
import uuid
import random
import string
import threading
import time
import traceback
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from gaia_common.config import Config
from gaia_common.utils.gaia_rescue_helper import GAIARescueHelper
from gaia_common.utils.safe_execution import run_shell_safe
from gaia_common.utils.tools_registry import TOOLS
from gaia_common.utils.vector_indexer import VectorIndexer
from gaia_common.utils.world_state import world_state_detail
from gaia_common.utils.service_client import get_study_client
from gaia_common.integrations.discord import DiscordConfig, DiscordWebhookSender
from .approval import ApprovalStore
from .web_tools import web_search, web_fetch
import json
from pathlib import Path
import os
import asyncio
import subprocess

# Study service client for gateway pattern
_study_client = None


def get_study_service():
    """Get the study service client (gateway to gaia-study)."""
    global _study_client
    if _study_client is None:
        _study_client = get_study_client()
    return _study_client

# --- Configuration & Initialization ---

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GAIA.MCPServer")

config = Config()
gaia_helper = GAIARescueHelper(config)




approval_store = ApprovalStore()

# Opt-in bypass for approval flow (useful for local testing)
MCP_BYPASS = os.getenv("GAIA_MCP_BYPASS", "false").lower() in ("1", "true", "yes")

app = FastAPI(
    title="GAIA MCP-Lite Server",
    description="Provides a secure JSON-RPC interface to GAIA's tools.",
    version="0.1.0"
)


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy", "service": "gaia-mcp"}


def create_app() -> FastAPI:
    """Factory function to return the configured FastAPI application.

    This returns the existing app instance which has all routes registered.
    """
    return app

# Tools that must go through approval (unless MCP_BYPASS is set)
SENSITIVE_TOOLS = {"ai_write", "write_file", "run_shell", "memory_rebuild_index"}

# --- Tool Dispatcher ---

async def dispatch_tool(tool_name: str, params: dict) -> any:
    """Maps a tool name to its implementation and executes it."""
    try:
        logger.info("Dispatching tool '%s'", tool_name)
        logger.debug("[DEBUG] Dispatching tool '%s' param_keys=%s", tool_name, sorted(list((params or {}).keys())))
    except Exception:
        logger.debug("[DEBUG] Dispatching tool '%s'", tool_name)

    if tool_name == "list_tools":
        return list(TOOLS.keys())
    
    if tool_name == "describe_tool":
        return TOOLS.get(params.get("tool_name"), {"error": "Tool not found"})

    # Map tool names to GAIARescueHelper methods
    # This is where the core logic from the helper is exposed.
    tool_map = {
        "run_shell": lambda p: run_shell_safe(p.get("command"), set(config.SAFE_EXECUTE_FUNCTIONS)),
        "read_file": lambda p: _read_file_impl(p),
        "write_file": lambda p: _write_file_impl(p),
        "ai_write": lambda p: _ai_write_impl(p),
        "list_dir": lambda p: _list_dir_impl(p),
        "list_files": lambda p: _list_files_impl(p),
        "list_tree": lambda p: _list_tree_impl(p),
        "world_state": lambda p: world_state_detail(),
        "memory_status": lambda p: _memory_status_impl(p),
        "memory_query": lambda p: _memory_query_impl(p),
        "memory_rebuild_index": lambda p: _memory_rebuild_index_impl(p),
        "find_files": lambda p: _find_files_impl(p),
        "find_relevant_documents": lambda p: _find_relevant_documents(p),
        # Response fragmentation tools
        "fragment_write": lambda p: _fragment_write_impl(p),
        "fragment_read": lambda p: _fragment_read_impl(p),
        "fragment_assemble": lambda p: _fragment_assemble_impl(p),
        "fragment_list_pending": lambda p: _fragment_list_pending_impl(p),
        "fragment_clear": lambda p: _fragment_clear_impl(p),
        # Knowledge base tools
        "embed_documents": lambda p: VectorIndexer.instance(p.get("knowledge_base_name")).add_document(p.get("file_path")) if p.get("file_path") else VectorIndexer.instance(p.get("knowledge_base_name")).build_index_from_docs(),
        "query_knowledge": lambda p: VectorIndexer.instance(p.get("knowledge_base_name")).query(p.get("query"), top_k=p.get("top_k", 5)),
        "add_document": lambda p: VectorIndexer.instance(p.get("knowledge_base_name")).add_document(p.get("file_path")),
        "send_discord_message": lambda p: _send_discord_message_impl(p),
        # Web research tools
        "web_search": lambda p: web_search(p),
        "web_fetch": lambda p: web_fetch(p),
        # Self-introspection tools
        "introspect_logs": lambda p: _introspect_logs_impl(p),
    }

    # Study mode / LoRA adapter tools are async (gateway calls to gaia-study)
    async_tools = {
        "study_start": _study_start_impl,
        "study_status": _study_status_impl,
        "study_cancel": _study_cancel_impl,
        "adapter_list": _adapter_list_impl,
        "adapter_load": _adapter_load_impl,
        "adapter_unload": _adapter_unload_impl,
        "adapter_delete": _adapter_delete_impl,
        "adapter_info": _adapter_info_impl,
    }

    if tool_name in async_tools:
        return await async_tools[tool_name](params)
    # Validate tool exists and execute
    if tool_name not in tool_map:
        raise ValueError(f"Tool '{tool_name}' is not a valid, implemented tool.")

    # Execute the mapped function (may be sync or async; handle sync results)
    try:
        result = tool_map[tool_name](params)
        return result
    except Exception as e:
        logger.error(f"Error executing tool '{tool_name}': {e}")
        logger.error(traceback.format_exc())
        raise


def _send_discord_message_impl(params: dict) -> dict:
    """Send a message to Discord via webhook."""
    content = params.get("content")
    if not content:
        raise ValueError("content is required")
    
    config = DiscordConfig.from_env()
    if not config.webhook_url:
        raise ValueError("DISCORD_WEBHOOK_URL is not configured")

    sender = DiscordWebhookSender(config.webhook_url, config.bot_name, config.avatar_url)
    success = sender.send(content)
    return {"ok": success}


def _ai_write_impl(params: dict) -> dict:
    """Perform a safe ai_write: write params['content'] to params['path'] and return metadata."""
    try:
        path = params.get("path")
        content = params.get("content", "")
        base_cwd = params.get("base_cwd")
        if not path:
            raise ValueError("missing path")
        p = Path(path)
        # If caller provides a base_cwd, use it to resolve relative paths. This
        # allows requesters (or tests) to assert where files should be written.
        if base_cwd:
            try:
                base = Path(base_cwd)
                if not base.is_absolute():
                    # Normalize non-absolute base to cwd-relative absolute
                    base = (Path.cwd() / base).resolve()
                if not p.is_absolute():
                    p = base / p
            except Exception:
                # Fallback to cwd if base_cwd is invalid
                if not p.is_absolute():
                    p = Path.cwd() / p
        else:
            # If path is relative, make it explicit relative to current working dir
            if not p.is_absolute():
                p = Path.cwd() / p
        # Diagnostic: log current working dir and resolved target path to help debugging
        try:
            cwd = Path.cwd()
            logger.info(f"ai_write: cwd={cwd} target={path} resolved={p.resolve()}")
        except Exception:
            logger.info(f"ai_write: cwd unknown target={p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return {"ok": True, "path": str(p), "bytes": len(content)}
    except Exception as e:
        logger.error(f"ai_write failed for {params.get('path')}: {e}")
        raise


def _list_dir_impl(params: dict):
    """List a directory (shallow, non-recursive)."""
    path = params.get("path") or "/knowledge"
    p = Path(path).resolve()
    if not p.is_dir():
        raise ValueError(f"{p} is not a directory")
    entries = sorted([e.name for e in p.iterdir()])
    return {"ok": True, "path": str(p), "entries": entries[:200]}


def _list_files_impl(params: dict):
    """Recursively list files under a given path."""
    root = params.get("path") or "/knowledge"
    max_depth = int(params.get("max_depth", 3))
    max_entries = int(params.get("max_entries", 1000))
    max_depth = max(1, min(max_depth, 8))
    max_entries = max(1, min(max_entries, 5000))

    root_path = Path(root).resolve()
    allow_roots = [Path("/knowledge").resolve(), Path("/sandbox").resolve(), Path("/models").resolve()]
    if not any(str(root_path).startswith(str(a)) for a in allow_roots):
        raise ValueError("Path not allowed")
    if not root_path.exists() or not root_path.is_dir():
        raise ValueError(f"{root_path} is not a directory")

    results = []
    exclude_dirs = {".git", ".pytest_cache", "__pycache__", ".cache", ".venv", "node_modules", "archive"}
    def walk(path: Path, depth: int):
        if depth > max_depth or len(results) >= max_entries:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda x: x.name)
        except Exception:
            return
        for entry in entries:
            if len(results) >= max_entries:
                return
            try:
                if entry.name.startswith(".") or entry.name in exclude_dirs:
                    continue
                if entry.is_file():
                    results.append(str(entry.resolve()))
                if entry.is_dir():
                    walk(entry, depth + 1)
            except Exception:
                continue
    walk(root_path, 0)
    truncated = len(results) >= max_entries
    return {"ok": True, "path": str(root_path), "max_depth": max_depth, "files": results, "truncated": truncated}

def _list_tree_impl(params: dict):
    """Produce a bounded directory tree with depth/entry limits."""
    root = params.get("path") or "/knowledge"
    max_depth = int(params.get("max_depth", 3))
    max_entries = int(params.get("max_entries", 200))
    max_depth = max(1, min(max_depth, 6))
    max_entries = max(10, min(max_entries, 1000))

    root_path = Path(root).resolve()
    allow_roots = [Path("/knowledge").resolve(), Path("/sandbox").resolve(), Path("/models").resolve()]
    if not any(str(root_path).startswith(str(a)) for a in allow_roots):
        raise ValueError("Path not allowed")
    if not root_path.exists() or not root_path.is_dir():
        raise ValueError(f"{root_path} is not a directory")

    lines = []
    count = 0
    prefix_map = {}

    def add_line(depth, name):
        nonlocal count
        if count >= max_entries:
            return False
        indent = "    " * depth
        lines.append(f"{indent}{name}")
        count += 1
        return True

    def walk(path: Path, depth: int):
        if depth > max_depth:
            return
        entries = sorted(path.iterdir(), key=lambda x: x.name)
        for entry in entries:
            if count >= max_entries:
                return
            # Skip noisy/hidden folders and bulky archives
            if entry.name.startswith(".") or entry.name in ("__pycache__", ".git", ".pytest_cache", "node_modules", "archive"):
                continue
            name = entry.name + ("/" if entry.is_dir() else "")
            if not add_line(depth, name):
                return
            if entry.is_dir():
                walk(entry, depth + 1)

    add_line(0, str(root_path))
    walk(root_path, 1)
    truncated = count >= max_entries
    return {"ok": True, "path": str(root_path), "max_depth": max_depth, "max_entries": max_entries, "truncated": truncated, "tree": "\n".join(lines)}

def _read_file_impl(params: dict):
    """Read a file with an allowlist and size guard."""
    path = params.get("path")
    if not path:
        raise ValueError("path is required")
    p = Path(path).resolve()
    allow_roots = [
        Path("/knowledge").resolve(),
        Path("/sandbox").resolve(),
        Path("/models").resolve(),
    ]
    if not any(str(p).startswith(str(a)) for a in allow_roots):
        raise ValueError("Path not allowed")
    if not p.is_file():
        raise ValueError(f"{p} is not a file")
    data = p.read_bytes()
    max_bytes = 512 * 1024
    if len(data) > max_bytes:
        raise ValueError(f"File too large to read safely ({len(data)} bytes > {max_bytes})")
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        text = data.decode("latin-1", errors="replace")
    return {"ok": True, "path": str(p), "bytes": len(data), "content": text}

def _write_file_impl(params: dict) -> dict:
    """Write content to a file, restricted to writable data volumes."""
    path = params.get("path")
    content = params.get("content", "")
    if not path:
        raise ValueError("path is required")

    # Resolve to absolute path
    p = Path(path)
    if not p.is_absolute():
        p = Path("/sandbox") / p
    p = p.resolve()

    # Allowlist: only the writable data volumes from docker-compose.yml
    # Excludes /app (source code) and /gaia-common, /models (read-only)
    allow_roots = [Path("/knowledge").resolve(), Path("/sandbox").resolve()]
    if not any(str(p).startswith(str(a) + "/") or str(p) == str(a) for a in allow_roots):
        raise ValueError(f"Path not allowed — write_file is restricted to: {[str(a) for a in allow_roots]}")

    # Re-check after resolve to prevent symlink traversal
    real = p.resolve()
    if not any(str(real).startswith(str(a) + "/") or str(real) == str(a) for a in allow_roots):
        raise ValueError("Path not allowed after symlink resolution")

    # Create parent directories
    real.parent.mkdir(parents=True, exist_ok=True)

    # Write file
    with open(real, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"write_file: wrote {len(content)} bytes to {real}")
    return {"ok": True, "path": str(real), "bytes": len(content)}


VECTOR_INDEX_PATH = Path("/knowledge/vector_store/index.json")

def _memory_status_impl(params: dict):
    """Summarize vector index status."""
    vi = VectorIndexer.instance()
    vi.refresh_index()
    docs = vi.index.get("docs") or []
    embeddings = vi.index.get("embeddings") or []
    return {
        "ok": True,
        "doc_count": len(docs),
        "embedding_count": len(embeddings),
        "index_path": str(VECTOR_INDEX_PATH),
        "model_path": vi.model_path,
    }

def _memory_query_impl(params: dict):
    query = params.get("query")
    top_k = int(params.get("top_k", 5))
    if not query:
        raise ValueError("query is required")
    vi = VectorIndexer.instance()
    results = vi.query(query, top_k=top_k)
    return {"ok": True, "results": results}

def _memory_rebuild_index_impl(params: dict):
    doc_dir = params.get("doc_dir") or "./knowledge/system_reference/GAIA_Function_Map"
    vi = VectorIndexer.instance()
    ok = vi.build_index_from_docs(doc_dir=doc_dir)
    vi.refresh_index()
    return {
        "ok": bool(ok),
        "doc_count": len(vi.index.get("docs") or []),
        "model_path": vi.model_path,
        "indexed_dir": str(Path(doc_dir).resolve()),
    }

def _find_files_impl(params: dict):
    """Search for files by substring under an allowlisted root, bounded by depth and result count."""
    query = (params.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    root = Path(params.get("root") or "/knowledge").resolve()
    max_depth = int(params.get("max_depth", 5))
    max_results = int(params.get("max_results", 50))
    max_depth = max(1, min(max_depth, 8))
    max_results = max(1, min(max_results, 200))

    allow_roots = [Path("/knowledge").resolve(), Path("/sandbox").resolve(), Path("/models").resolve()]
    if not any(str(root).startswith(str(a)) for a in allow_roots):
        raise ValueError("Root not allowed")
    if not root.exists() or not root.is_dir():
        raise ValueError(f"{root} is not a directory")

    results = []
    exclude_dirs = {".git", ".pytest_cache", "__pycache__", ".cache", ".venv", "node_modules"}
    def walk(path: Path, depth: int):
        if depth > max_depth or len(results) >= max_results:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda x: x.name)
        except Exception:
            return
        for entry in entries:
            if len(results) >= max_results:
                return
            try:
                if entry.name.startswith(".") or entry.name in exclude_dirs:
                    continue
                if entry.is_file() and query.lower() in entry.name.lower():
                    results.append(str(entry.resolve()))
                if entry.is_dir():
                    walk(entry, depth + 1)
            except Exception:
                continue
    walk(root, 0)
    return {"ok": True, "query": query, "root": str(root), "max_depth": max_depth, "results": results, "truncated": len(results) >= max_results}


def _find_relevant_documents(params: dict):
    """Find documents relevant to a query within a knowledge base."""
    query = params.get("query")
    knowledge_base_name = params.get("knowledge_base_name")

    if not query or not knowledge_base_name:
        raise ValueError("query and knowledge_base_name are required")

    # Use the already-imported Config from gaia_common
    conf = Config()

    # Access the KNOWLEDGE_BASES dictionary from constants
    knowledge_bases = conf.constants.get("KNOWLEDGE_BASES", {})
    kb_config = knowledge_bases.get(knowledge_base_name)

    if not kb_config:
        # Fallback: check if the user passed a direct path or if it's in a different config structure
        logger.warning(f"Knowledge base '{knowledge_base_name}' not found in KNOWLEDGE_BASES config.")
        return {"files": []}

    doc_dir = kb_config.get("doc_dir")
    
    # Safety Check: Ensure directory exists
    if not os.path.exists(doc_dir):
        logger.warning(f"Doc directory not found: {doc_dir}") 
        return {"files": []}

    # Prepare grep command
    keywords = query.split()
    safe_keywords = [k for k in keywords if k.isalnum()] 
    
    if not safe_keywords:
        return {"files": []}

    # Construct grep command: grep -r -i -l -E "term1|term2" /path/to/docs
    pattern = "|".join(safe_keywords)
    cmd = ["grep", "-r", "-i", "-l", "-E", pattern, doc_dir]

    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        files = result.stdout.strip().splitlines()
        return {"files": files}

    except subprocess.CalledProcessError as e:
        # Grep returns exit code 1 if NO matches are found.
        if e.returncode == 1:
            return {"files": []}
        
        logger.error(f"Grep failed with error: {e.stderr}")
        raise RuntimeError(f"Search command failed: {e.stderr}")

# --- Self-Introspection Tools ---

# Map service names to their log file paths inside the container
_LOG_FILE_MAP = {
    "gaia-core": "/logs/gaia-core.log",
    "gaia-web": "/logs/gaia-web.log",
    "gaia-mcp": "/logs/gaia-mcp.log",
    "gaia-study": "/logs/gaia-study.log",
    "discord": "/logs/discord_bot.log",
}

# Severity levels in ascending order
_LEVEL_ORDER = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}


def _introspect_logs_impl(params: dict) -> dict:
    """View recent service logs for self-diagnosis."""
    service = params.get("service")
    if not service:
        raise ValueError("service is required")

    lines_requested = min(int(params.get("lines", 50)), 200)
    search = (params.get("search") or "").strip().lower()
    level = (params.get("level") or "").strip().upper()

    log_path = _LOG_FILE_MAP.get(service)
    if not log_path:
        return {"ok": False, "error": f"Unknown service: {service}. Valid: {list(_LOG_FILE_MAP.keys())}"}

    p = Path(log_path)
    if not p.is_file():
        return {"ok": False, "error": f"Log file not found: {log_path}. Service may not have started yet or logging is not configured."}

    # Read efficiently: for files > 2MB, only read the tail
    max_read = 2 * 1024 * 1024  # 2MB
    file_size = p.stat().st_size

    try:
        if file_size > max_read:
            with open(p, "rb") as f:
                f.seek(-max_read, 2)
                raw = f.read().decode("utf-8", errors="replace")
            # Drop the first (likely partial) line
            all_lines = raw.split("\n")[1:]
        else:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.read().split("\n")
    except Exception as e:
        return {"ok": False, "error": f"Failed to read log file: {e}"}

    # Remove trailing empty line from split
    if all_lines and all_lines[-1] == "":
        all_lines = all_lines[:-1]

    total_in_file = len(all_lines)

    # Apply severity filter
    if level and level in _LEVEL_ORDER:
        min_level = _LEVEL_ORDER[level]
        filtered = []
        for line in all_lines:
            for lv, lv_num in _LEVEL_ORDER.items():
                if lv_num >= min_level and (f" {lv}:" in line or f" {lv} " in line):
                    filtered.append(line)
                    break
        all_lines = filtered

    # Apply search filter
    if search:
        all_lines = [line for line in all_lines if search in line.lower()]

    # Take the last N lines
    result_lines = all_lines[-lines_requested:]

    return {
        "ok": True,
        "service": service,
        "log_file": log_path,
        "file_size_bytes": file_size,
        "total_lines_in_file": total_in_file,
        "lines_after_filter": len(all_lines),
        "lines_returned": len(result_lines),
        "content": "\n".join(result_lines),
    }


# --- Response Fragmentation Tools ---

def _fragment_write_impl(params: dict) -> dict:
    """Store a response fragment for later assembly."""
    parent_request_id = params.get("parent_request_id")
    if not parent_request_id:
        raise ValueError("parent_request_id is required")

    sequence = int(params.get("sequence", 0))
    content = params.get("content", "")
    continuation_hint = params.get("continuation_hint", "")
    is_complete = bool(params.get("is_complete", False))
    token_count = int(params.get("token_count", 0))

    return gaia_helper.fragment_write(
        parent_request_id=parent_request_id,
        sequence=sequence,
        content=content,
        continuation_hint=continuation_hint,
        is_complete=is_complete,
        token_count=token_count
    )


def _fragment_read_impl(params: dict) -> dict:
    """Retrieve all fragments for a given request."""
    parent_request_id = params.get("parent_request_id")
    if not parent_request_id:
        raise ValueError("parent_request_id is required")

    fragments = gaia_helper.fragment_read(parent_request_id)
    return {"ok": True, "fragments": fragments, "count": len(fragments)}


def _fragment_assemble_impl(params: dict) -> dict:
    """Assemble fragments into a complete response."""
    parent_request_id = params.get("parent_request_id")
    if not parent_request_id:
        raise ValueError("parent_request_id is required")

    seam_overlap_check = bool(params.get("seam_overlap_check", True))
    return gaia_helper.fragment_assemble(parent_request_id, seam_overlap_check=seam_overlap_check)


def _fragment_list_pending_impl(params: dict) -> dict:
    """List all pending (incomplete) fragment requests."""
    pending = gaia_helper.fragment_list_pending()
    return {"ok": True, "pending": pending, "count": len(pending)}


def _fragment_clear_impl(params: dict) -> dict:
    """Clear fragments for a specific request or all fragments."""
    parent_request_id = params.get("parent_request_id")  # Optional
    result = gaia_helper.fragment_clear(parent_request_id)
    return {"ok": True, "message": result}


# --- Study Mode / LoRA Adapter Tools (Gateway to gaia-study service) ---

async def _study_start_impl(params: dict) -> dict:
    """Start a study session to learn from documents (via gaia-study gateway)."""
    adapter_name = params.get("adapter_name")
    documents = params.get("documents", [])

    if not adapter_name:
        raise ValueError("adapter_name is required")
    if not documents:
        raise ValueError("documents list is required and cannot be empty")

    # Call gaia-study service via gateway
    client = get_study_service()
    try:
        result = await client.post("/study/start", {
            "adapter_name": adapter_name,
            "documents": documents,
            "tier": int(params.get("tier", 3)),
            "pillar": params.get("pillar", "general"),
            "description": params.get("description", ""),
            "max_steps": int(params.get("max_steps", 100)),
            "activation_triggers": params.get("activation_triggers", []),
            "tags": params.get("tags", []),
        })
        return result
    except Exception as e:
        logger.error(f"Failed to start study via gateway: {e}")
        return {"ok": False, "error": str(e)}


async def _study_status_impl(params: dict) -> dict:
    """Get current study mode status (via gaia-study gateway)."""
    client = get_study_service()
    try:
        return await client.get("/study/status")
    except Exception as e:
        logger.error(f"Failed to get study status via gateway: {e}")
        return {"state": "error", "message": str(e)}


async def _study_cancel_impl(params: dict) -> dict:
    """Cancel an in-progress training session (via gaia-study gateway)."""
    client = get_study_service()
    try:
        return await client.post("/study/cancel")
    except Exception as e:
        logger.error(f"Failed to cancel study via gateway: {e}")
        return {"ok": False, "message": str(e)}


async def _adapter_list_impl(params: dict) -> dict:
    """List available LoRA adapters (via gaia-study gateway)."""
    client = get_study_service()
    tier = params.get("tier")
    try:
        query_params = {"tier": tier} if tier is not None else None
        return await client.get("/adapters", params=query_params)
    except Exception as e:
        logger.error(f"Failed to list adapters via gateway: {e}")
        return {"ok": False, "adapters": [], "error": str(e)}


async def _adapter_load_impl(params: dict) -> dict:
    """Load a LoRA adapter for use in generation (via gaia-study gateway)."""
    adapter_name = params.get("adapter_name")
    tier = int(params.get("tier", 3))

    if not adapter_name:
        raise ValueError("adapter_name is required")

    client = get_study_service()
    try:
        return await client.post("/adapters/load", {
            "adapter_name": adapter_name,
            "tier": tier
        })
    except Exception as e:
        logger.error(f"Failed to load adapter via gateway: {e}")
        return {"ok": False, "error": str(e)}


async def _adapter_unload_impl(params: dict) -> dict:
    """Unload a LoRA adapter (via gaia-study gateway)."""
    adapter_name = params.get("adapter_name")

    if not adapter_name:
        raise ValueError("adapter_name is required")

    client = get_study_service()
    try:
        return await client.post("/adapters/unload", {
            "adapter_name": adapter_name,
            "tier": int(params.get("tier", 3))
        })
    except Exception as e:
        logger.error(f"Failed to unload adapter via gateway: {e}")
        return {"ok": False, "error": str(e)}


async def _adapter_delete_impl(params: dict) -> dict:
    """Delete a LoRA adapter (via gaia-study gateway)."""
    adapter_name = params.get("adapter_name")
    tier = int(params.get("tier", 3))

    if not adapter_name:
        raise ValueError("adapter_name is required")

    client = get_study_service()
    try:
        return await client.delete(f"/adapters/{adapter_name}", params={"tier": tier})
    except Exception as e:
        logger.error(f"Failed to delete adapter via gateway: {e}")
        return {"ok": False, "error": str(e)}


async def _adapter_info_impl(params: dict) -> dict:
    """Get detailed info about a specific adapter (via gaia-study gateway)."""
    adapter_name = params.get("adapter_name")
    tier = int(params.get("tier", 3))

    if not adapter_name:
        raise ValueError("adapter_name is required")

    client = get_study_service()
    try:
        return await client.get(f"/adapters/{adapter_name}", params={"tier": tier})
    except Exception as e:
        logger.error(f"Failed to get adapter info via gateway: {e}")
        return {"ok": False, "error": str(e)}


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

    # Enforce approval for sensitive tools unless bypass is explicitly enabled
    if (not MCP_BYPASS) and method in SENSITIVE_TOOLS:
        return JSONResponse(content={"jsonrpc": "2.0", "error": {"code": -32001, "message": f"'{method}' requires approval. Use /request_approval first."}, "id": request_id}, status_code=403)

    # Validate params against the tool's schema
    if method in TOOLS:
        from jsonschema import validate, ValidationError
        try:
            validate(instance=params, schema=TOOLS[method]["params"])
        except ValidationError as e:
            return JSONResponse(content={
                "jsonrpc": "2.0", 
                "error": {"code": -32602, "message": f"Invalid params: {e.message}"},
                "id": request_id
            }, status_code=400)

    # Dispatch the tool call
    try:
        result = await dispatch_tool(method, params)
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "result": result,
            "id": request_id
        })
    except Exception as e:
        logger.error(f"Error dispatching tool '{method}': {e}", exc_info=True)
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "error": {"code": -32603, "message": f"Internal error: {e}", "data": traceback.format_exc()},
            "id": request_id
        }, status_code=500)

@app.get("/")
def read_root():
    return {"message": "GAIA MCP-Lite Server is running."}


@app.post("/request_approval")
async def request_approval(request: Request):
    """Create a pending action that requires human approval.

    Body should be JSON: {"method": "tool_name", "params": {...}}
    Returns: {"action_id": str, "challenge": str}
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"error": "invalid json"}, status_code=400)

    method = body.get("method")
    params = body.get("params", {})
    if not method:
        return JSONResponse(content={"error": "missing method"}, status_code=400)

    # Validate method exists in TOOLS
    if method not in TOOLS:
        return JSONResponse(content={"error": "unknown tool"}, status_code=400)

    # If bypass mode is enabled, immediately dispatch the tool and return the result
    if MCP_BYPASS:
        try:
            result = await dispatch_tool(method, params)
            return {"ok": True, "bypassed": True, "result": result}
        except Exception as e:
            return JSONResponse(content={"error": f"execution failed in bypass mode: {e}"}, status_code=500)

    # Determine whether the caller explicitly allows the action to remain pending.
    # Support two ways: top-level field 'allow_pending' or params['_allow_pending'] for backwards compat.
    allow_pending = False
    if isinstance(body.get("allow_pending"), bool):
        allow_pending = body.get("allow_pending")
    elif isinstance(params, dict) and params.get("_allow_pending") is True:
        allow_pending = True

    action_id, challenge, created_at, expiry = approval_store.create_pending(method, params, allow_pending=allow_pending)
    # Retrieve proposal text (if any) from the store for client display
    try:
        proposal = approval_store._store.get(action_id, {}).get("proposal")
    except Exception:
        proposal = None

    # Return challenge, timestamps and a human-friendly proposal (ISO8601) so callers can correlate
    return {
        "action_id": action_id,
        "challenge": challenge,
        "proposal": proposal,
        "created_at": datetime.utcfromtimestamp(created_at).isoformat(),
        "expiry": datetime.utcfromtimestamp(expiry).isoformat()
    }


@app.get("/pending_approvals")
async def pending_approvals():
    """Return a list of pending approvals (no secrets)."""
    return {"pending": approval_store.list_pending()}


@app.post("/approve_action")
async def approve_action(request: Request):
    """Approve a pending action by providing the reversed 5-char challenge.

    Body: {"action_id": str, "approval": str}
    If approved, dispatches the underlying tool via dispatch_tool and returns result.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"error": "invalid json"}, status_code=400)

    action_id = body.get("action_id")
    approval = body.get("approval")
    if not action_id or not approval:
        return JSONResponse(content={"error": "missing fields"}, status_code=400)

    try:
        payload = approval_store.approve(action_id, approval)
    except KeyError:
        return JSONResponse(content={"error": "action not found or expired"}, status_code=404)
    except ValueError:
        return JSONResponse(content={"error": "invalid approval string"}, status_code=403)

    # Dispatch the tool now that it's approved
    try:
        result = await dispatch_tool(payload["method"], payload.get("params", {}))
        approved_at = datetime.utcfromtimestamp(time.time()).isoformat()
        # Append audit entry (without secrets) to audit log
        try:
            audit_dir = Path("knowledge/system_reference")
            audit_dir.mkdir(parents=True, exist_ok=True)
            audit_path = audit_dir / "dev_matrix_audit.log"
            audit_entry = {
                "action_id": action_id,
                "method": payload.get("method"),
                "created_at": datetime.utcfromtimestamp(payload.get("created_at", 0)).isoformat(),
                "approved_at": approved_at,
                "params_preview": {k: (str(v)[:200] + "..." if isinstance(v, str) and len(v) > 200 else v) for k, v in (payload.get("params") or {}).items()}
            }
            with open(audit_path, "a", encoding="utf-8") as af:
                af.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")
        except Exception:
            logging.getLogger("GAIA.MCPServer").exception("Failed to write audit entry")

        return {"ok": True, "result": result, "approved_at": approved_at}
    except ValueError as e:
        # Tool-level validation errors (path not allowed, missing params, etc.)
        # Return as structured error, not 500 — the tool was approved but the
        # params were invalid. This lets the caller distinguish between
        # approval failures (403/404) and tool execution failures.
        return {"ok": False, "error": str(e), "approved_at": datetime.utcfromtimestamp(time.time()).isoformat()}
    except Exception as e:
        logging.getLogger("GAIA.MCPServer").error(f"Error executing approved action: {e}", exc_info=True)
        return JSONResponse(content={"error": f"execution failed: {e}"}, status_code=500)

# To run this server:
# uvicorn app.mcp_lite_server:app --host 0.0.0.0 --port 4141 --reload

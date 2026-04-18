"""
GAIA MCP Tools - Tool Dispatcher and Implementations.
"""

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

from gaia_common.utils import get_logger
try:
    from gaia_common.utils.error_logging import log_gaia_error
except ImportError:
    def log_gaia_error(lgr, code, detail="", **kw):
        lgr.error("[%s] %s", code, detail)
from gaia_common.config import Config
from gaia_common.utils.safe_execution import run_shell_safe
from gaia_common.utils.gaia_rescue_helper import GAIARescueHelper
from gaia_common.utils.vector_indexer import VectorIndexer
from gaia_common.utils.cfr_manager import CFRManager
from gaia_common.utils.world_state import world_state_detail
from gaia_common.utils.service_client import get_study_client

from .approval import ApprovalStore
from .web_tools import web_search, web_fetch
from .browser_tools import browser_tool as _browser_tool
from .kanka_tools import (
    kanka_list_campaigns, kanka_search, kanka_list_entities,
    kanka_get_entity, kanka_create_entity, kanka_update_entity,
)
from .notebooklm_tools import (
    notebooklm_list_notebooks, notebooklm_get_notebook,
    notebooklm_list_sources, notebooklm_list_notes,
    notebooklm_list_artifacts, notebooklm_chat,
    notebooklm_download_audio, notebooklm_generate_audio,
    notebooklm_create_note,
)
from .listener_tools import (
    audio_listen_start, audio_listen_stop, audio_listen_status,
)
from .inbox_tools import (
    audio_inbox_status, audio_inbox_list, audio_inbox_review,
    audio_inbox_process,
)
from .fabric_tools import fabric_schemas, execute_fabric_tool

logger = get_logger(__name__)

# Initialize GAIARescueHelper (needs config). This needs to be instantiated once.
# For now, create a global instance. A better pattern would be dependency injection.
_config_instance = Config()
_gaia_helper = GAIARescueHelper(_config_instance)
_cfr_manager = CFRManager()

# Default timeout for LLM-dependent tool calls (seconds).
# CPU/GGUF inference at ~7 tok/s can take 60-120s; 120s gives headroom.
LLM_TOOL_TIMEOUT = int(os.getenv("MCP_LLM_TOOL_TIMEOUT", "120"))


async def _run_blocking_with_timeout(func, *args, timeout: int = LLM_TOOL_TIMEOUT, **kwargs):
    """Run a blocking (synchronous) function in a thread pool with a timeout.

    Prevents LLM-dependent tools from blocking the asyncio event loop and
    makes the MCP server stay responsive to health checks and other requests.

    Returns the function result, or a structured error dict on timeout.
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(func, *args, **kwargs),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "LLM tool call timed out after %ds: %s", timeout, func.__name__
        )
        return {
            "ok": False,
            "error": f"LLM inference timeout ({timeout}s). CPU inference may be too slow for this operation.",
            "timeout": True,
            "timeout_seconds": timeout,
        }


# Placeholder for TOOLS registry (will be populated from gaia-common.utils.tools_registry)
from gaia_common.utils.tools_registry import TOOLS
from gaia_common.utils.domain_tools import (
    DOMAIN_TOOLS, validate_domain_call, is_sensitive,
)

# Merge dynamically-loaded Fabric pattern tools into the TOOLS registry.
# This runs at import time within gaia-mcp's process only — other services
# importing TOOLS from gaia-common won't see fabric tools.
TOOLS.update(fabric_schemas)

# Tools that require explicit human approval before execution
SENSITIVE_TOOLS = {
    "ai_write", "write_file", "run_shell", "memory_rebuild_index",
    "promotion_create_request", "kanka_create_entity", "kanka_update_entity",
    "notebooklm_create_note", "audio_listen_start"
}

def list_tools() -> list:
    """List available domain tools (13 public-facing interfaces).

    Returns domain names only.  Legacy tool names (70+) are kept
    internally for routing but are NOT exposed to clients.
    Use ``describe_tool(domain)`` for full schema with actions.
    """
    from gaia_common.utils.domain_tools import DOMAIN_TOOLS
    return list(DOMAIN_TOOLS.keys())


def list_tools_full() -> dict:
    """Full domain tool schemas with actions and sensitivity markers."""
    from gaia_common.utils.domain_tools import build_domain_schemas
    return build_domain_schemas()


def describe_tool(tool_name: str) -> Dict[str, Any]:
    """Get schema and description for a specific tool.

    Accepts both domain names (file, web, knowledge) and legacy
    names (read_file, web_search) for backward compatibility.
    """
    # Try domain schema first
    try:
        from gaia_common.utils.domain_tools import build_domain_schemas
        domains = build_domain_schemas()
        if tool_name in domains:
            return domains[tool_name]
    except Exception:
        pass
    # Fallback to legacy
    return TOOLS.get(tool_name, {"error": f"Tool '{tool_name}' not found. Available domains: file, web, knowledge, shell, audio, study, introspect, worldbuild, notebook, context, browser, manage, fabric"})


async def execute_limb(method: str, params: Dict, approval_store: ApprovalStore, pre_approved: bool = False) -> Any:
    """
    Executes a tool method with the given parameters.
    Handles sensitive tools requiring approval and blast shield validation.
    Supports both legacy tool names AND domain tool names (file, shell, web, etc.).
    """
    # ── Meta-verb routing (Unified Skill Architecture) ───────────────
    # 5 meta-verbs that cover all tool functionality. Routes to the
    # SkillGateway which dispatches to existing infrastructure.
    _META_VERBS = {"search", "do", "learn", "remember", "ask"}
    if method in _META_VERBS:
        from gaia_mcp.skill_gateway import get_gateway
        gateway = get_gateway()
        return await gateway.route(method, params or {})

    # ── Domain tool routing ────────────────────────────────────────────
    # If method is a domain name (e.g., "file"), pop "action" from params,
    # resolve to the legacy tool name, and delegate to the same function.
    if method in DOMAIN_TOOLS and not DOMAIN_TOOLS[method].get("dynamic"):
        action = (params or {}).pop("action", None)
        if not action:
            raise ValueError(f"Domain tool '{method}' requires an 'action' parameter. "
                             f"Available: {list(DOMAIN_TOOLS[method]['actions'].keys())}")
        legacy_name = validate_domain_call(method, action)
        logger.info("Domain route: %s(action=%s) → %s", method, action, legacy_name)
        # Check domain-level sensitivity before delegating
        if not pre_approved and is_sensitive(method, action):
            raise PermissionError(f"Tool '{method}(action={action})' requires explicit approval.")
        return await execute_limb(legacy_name, params, approval_store, pre_approved=True)

    # Fabric domain: pattern param selects which fabric tool to run
    if method == "fabric":
        pattern = (params or {}).pop("pattern", (params or {}).pop("action", None))
        if not pattern:
            raise ValueError("fabric tool requires a 'pattern' parameter")
        legacy_name = f"fabric_{pattern}"
        logger.info("Domain route: fabric(pattern=%s) → %s", pattern, legacy_name)
        return await execute_limb(legacy_name, params, approval_store, pre_approved)

    # ── Legacy tool execution ──────────────────────────────────────────
    logger.info("Executing tool '%s'", method)
    logger.debug("[DEBUG] Executing tool '%s' with params_keys=%s", method, sorted(list((params or {}).keys())))

    # 1. Enforce sensitive tool policy unless pre-approved (bypass or already approved)
    if not pre_approved and method in SENSITIVE_TOOLS:
        raise PermissionError(f"Tool '{method}' requires explicit approval.")

    # 2. Blast Shield validation (proactive safety check)
    if approval_store:
        try:
            approval_store.validate_against_blast_shield(method, params)
        except ValueError as e:
            log_gaia_error(logger, "GAIA-MCP-010", f"Blocked '{method}': {e}")
            raise PermissionError(f"Blast Shield block: {e}")

    # 3. Handle built-in discoveries — expose domain tools, not legacy
    if method == "list_tools" or method == "rpc.discover":
        return list_tools()

    if method == "list_tools_full":
        return list_tools_full()

    if method == "describe_tool":
        return describe_tool(params.get("tool_name", ""))

    # 4. Map tool implementations
    tool_map = {
        "run_shell": lambda p: run_shell_safe(p.get("command"), set(_config_instance.SAFE_EXECUTE_FUNCTIONS)),
        "read_file": lambda p: _read_file_impl(p),
        "write_file": lambda p: _write_file_impl(p),
        "ai_write": lambda p: _ai_write_impl(p, _gaia_helper),
        "list_dir": lambda p: _list_dir_impl(p),
        "list_files": lambda p: _list_files_impl(p),
        "list_tree": lambda p: _list_tree_impl(p),
        "count_chars": lambda p: _count_chars_impl(p),
        "world_state": lambda p: world_state_detail(),
        "gpu_status": lambda p: world_state_detail(),
        "recall_events": lambda p: _recall_events_impl(p),
        "memory_status": lambda p: _memory_status_impl(p),
        "memory_query": lambda p: _memory_query_impl(p),
        "memory_rebuild_index": lambda p: _memory_rebuild_index_impl(p),
        "find_files": lambda p: _find_files_impl(p),
        "find_relevant_documents": lambda p: _find_relevant_documents(p),
        "list_knowledge_bases": lambda p: _list_knowledge_bases_impl(p),
        # Response fragmentation tools
        "fragment_write": lambda p: _gaia_helper.fragment_write(
            parent_request_id=p.get("parent_request_id"),
            sequence=int(p.get("sequence", 0)),
            content=p.get("content", ""),
            continuation_hint=p.get("continuation_hint", ""),
            is_complete=bool(p.get("is_complete", False)),
            token_count=int(p.get("token_count", 0))
        ),
        "fragment_read": lambda p: _gaia_helper.fragment_read(p.get("parent_request_id")),
        "fragment_assemble": lambda p: _gaia_helper.fragment_assemble(p.get("parent_request_id"), seam_overlap_check=bool(p.get("seam_overlap_check", True))),
        "fragment_list_pending": lambda p: _gaia_helper.fragment_list_pending(),
        "fragment_clear": lambda p: _gaia_helper.fragment_clear(p.get("parent_request_id")),
        # Cognitive Focus and Resolution (CFR) tools
        "cfr_ingest": lambda p: _cfr_manager.ingest(file_path=p.get("file_path"), doc_id=p.get("doc_id"), chunk_target=int(p.get("chunk_target", 3500))),
        "cfr_focus": lambda p: _cfr_manager.focus(doc_id=p["doc_id"], section_index=int(p["section_index"])),
        "cfr_expand": lambda p: _cfr_manager.expand(doc_id=p["doc_id"], section_index=int(p["section_index"])),
        "cfr_status": lambda p: _cfr_manager.status(doc_id=p.get("doc_id", "")),
        # Browser tools (OpenClaw methodology, GAIA security)
        "browser_browse": lambda p: _browser_tool({"action": "browse", **p}),
        "browser_snapshot": lambda p: _browser_tool({"action": "snapshot", **p}),
        "browser_links": lambda p: _browser_tool({"action": "links", **p}),
        "browser_forms": lambda p: _browser_tool({"action": "forms", **p}),
        "browser_click": lambda p: _browser_tool({"action": "click", "full_browser": True, **p}),
        "browser_type": lambda p: _browser_tool({"action": "type", "full_browser": True, **p}),
        "browser_screenshot": lambda p: _browser_tool({"action": "screenshot", "full_browser": True, **p}),
        "cfr_review_conversation": lambda p: _cfr_review_conversation(p),
        # Knowledge base tools (VectorIndexer from gaia_common)
        "embed_documents": lambda p: VectorIndexer.instance(p.get("knowledge_base_name")).add_document(p.get("file_path")) if p.get("file_path") else VectorIndexer.instance(p.get("knowledge_base_name")).build_index_from_docs(),
        "query_knowledge": lambda p: VectorIndexer.instance(p.get("knowledge_base_name")).query(p.get("query"), top_k=p.get("top_k", 5)),
        "add_document": lambda p: VectorIndexer.instance(p.get("knowledge_base_name")).add_document(p.get("file_path")),
        "index_document": lambda p: VectorIndexer.instance(p.get("knowledge_base_name")).add_document(p.get("file_path")),
        # Knowledge Graph (temporal triple store)
        "kg_query": lambda p: _kg_query_impl(p),
        "kg_add": lambda p: _kg_add_impl(p),
        "kg_invalidate": lambda p: _kg_invalidate_impl(p),
        "kg_timeline": lambda p: _kg_timeline_impl(p),
        "kg_stats": lambda p: _kg_stats_impl(p),
        # Web research tools
        "web_search": lambda p: web_search(p),
        "web_fetch": lambda p: web_fetch(p),
        # Kanka.io world-building tools
        "kanka_list_campaigns": lambda p: kanka_list_campaigns(p),
        "kanka_search": lambda p: kanka_search(p),
        "kanka_list_entities": lambda p: kanka_list_entities(p),
        "kanka_get_entity": lambda p: kanka_get_entity(p),
        "kanka_create_entity": lambda p: kanka_create_entity(p),
        "kanka_update_entity": lambda p: kanka_update_entity(p),
        # Promotion & blueprint tools
        "generate_blueprint": lambda p: _generate_blueprint_impl(p),
        "assess_promotion": lambda p: _assess_promotion_impl(p),
        # Promotion lifecycle tools
        "promotion_create_request": lambda p: _promotion_create_request_impl(p),
        "promotion_list_requests": lambda p: _promotion_list_requests_impl(p),
        "promotion_request_status": lambda p: _promotion_request_status_impl(p),
        # Audio inbox tools (sync — file-based)
        "audio_inbox_status": lambda p: audio_inbox_status(p),
        "audio_inbox_list": lambda p: audio_inbox_list(p),
        "audio_inbox_review": lambda p: audio_inbox_review(p),
        "audio_inbox_process": lambda p: audio_inbox_process(p),
        # Audio listener tools (sync — file-based control)
        "audio_listen_start": lambda p: audio_listen_start(p),
        "audio_listen_stop": lambda p: audio_listen_stop(p),
        "audio_listen_status": lambda p: audio_listen_status(p),
        # MemPalace tools
        "palace_store": lambda p: _palace_store_impl(p),
        "palace_recall": lambda p: _palace_recall_impl(p),
        "palace_navigate": lambda p: _palace_navigate_impl(p),
        "palace_status": lambda p: _palace_status_impl(p),
        # Self-introspection tools
        "introspect_logs": lambda p: _introspect_logs_impl(p),
        "replace": lambda p: _replace_impl(p),
        # Discord integration
        "send_discord_message": lambda p: _send_discord_message_impl(p),
    }

    async_tool_map = {
        # CFR tools that call LLM inference (blocking HTTP → run in thread pool with timeout)
        "cfr_compress": lambda p: _run_blocking_with_timeout(
            _cfr_manager.compress, doc_id=p["doc_id"], section_index=int(p["section_index"]),
        ),
        "cfr_synthesize": lambda p: _run_blocking_with_timeout(
            _cfr_manager.synthesize, doc_id=p["doc_id"],
        ),
        "cfr_rolling_context": lambda p: _run_blocking_with_timeout(
            _cfr_manager.rolling_context, doc_id=p["doc_id"], target_section=int(p["target_section"]),
        ),
        # NotebookLM tools (async httpx client)
        "notebooklm_list_notebooks": notebooklm_list_notebooks,
        "notebooklm_get_notebook": notebooklm_get_notebook,
        "notebooklm_list_sources": notebooklm_list_sources,
        "notebooklm_list_notes": notebooklm_list_notes,
        "notebooklm_list_artifacts": notebooklm_list_artifacts,
        "notebooklm_chat": notebooklm_chat,
        "notebooklm_download_audio": notebooklm_download_audio,
        "notebooklm_generate_audio": notebooklm_generate_audio,
        "notebooklm_create_note": notebooklm_create_note,
        # Study mode / LoRA adapter tools (gateway calls to gaia-study)
        "study_start": _study_start_impl,
        "study_status": _study_status_impl,
        "study_cancel": _study_cancel_impl,
        "adapter_list": _adapter_list_impl,
        "adapter_load": _adapter_load_impl,
        "adapter_unload": _adapter_unload_impl,
        "adapter_delete": _adapter_delete_impl,
        "adapter_info": _adapter_info_impl,
    }

    # Fabric pattern tools (all async — call gaia-core over HTTP)
    for _fn in fabric_schemas:
        if _fn not in async_tool_map:
            async_tool_map[_fn] = lambda p, _name=_fn: execute_fabric_tool(_name, p)

    if method not in tool_map and method not in async_tool_map:
        log_gaia_error(logger, "GAIA-MCP-001", f"Tool '{method}' not found")
        raise ValueError(f"Tool '{method}' is not a valid, implemented tool.")

    # Execute the tool
    if method in tool_map:
        return tool_map[method](params)
    elif method in async_tool_map:
        return await async_tool_map[method](params)
    
    raise ValueError(f"Tool '{method}' failed to dispatch.")


# Tool implementations (extracted from mcp_lite_server.py)
# =========================================================

def _validate_python_content(path: str, content: str):
    """
    If the path ends in .py, attempt to compile the content.
    Raises ValueError if compilation fails.
    """
    if path.endswith(".py"):
        try:
            import py_compile
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=True) as tmp:
                tmp.write(content)
                tmp.flush()
                py_compile.compile(tmp.name, doraise=True)
        except Exception as e:
            log_gaia_error(logger, "GAIA-MCP-025", f"Compilation failed for {path}: {e}")
            raise ValueError(f"Sovereign Shield: Cannot save {path} because it contains syntax errors: {e}")


def _ai_write_impl(params: dict, gaia_helper: GAIARescueHelper) -> dict:
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

        # [Security] Path allowlist — same restrictions as write_file
        _resolved = str(p.resolve())
        _allowed_prefixes = ["/knowledge", "/sandbox", "/gaia/GAIA_Project",
                             "/shared", "/tmp", "/logs"]
        _blocked_prefixes = ["/etc", "/boot", "/root", "/.ssh", "/run/secrets",
                             "/proc", "/sys", "/dev"]
        if any(_resolved.startswith(bp) for bp in _blocked_prefixes):
            return {"ok": False, "error": f"Path blocked by security policy: {_resolved}"}
        if not any(_resolved.startswith(ap) for ap in _allowed_prefixes):
            return {"ok": False, "error": f"Path outside allowed directories: {_resolved}"}

        # [Sovereign Shield] Pre-compile check
        _validate_python_content(str(p), content)

        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return {"ok": True, "path": str(p), "bytes": len(content)}
    except Exception as e:
        logger.error(f"ai_write failed for {params.get('path')}: {e}")
        raise


def _write_file_impl(params: dict) -> dict:
    """Write content to a file, restricted to writable data volumes and enforces Production Lock."""
    path = params.get("path")
    content = params.get("content", "")
    if not path:
        raise ValueError("path is required")

    # Resolve to absolute path
    p = Path(path)
    if not p.is_absolute():
        p = Path("/sandbox") / p
    p = p.resolve()

    # --- PRODUCTION LOCK (Sovereign Shield) ---
    # Block direct writes to live service directories.
    # Forces usage of /candidates/ path for development.
    path_str = str(p)
    is_live_code = any(segment in path_str for segment in ["/gaia-core/", "/gaia-web/", "/gaia-mcp/", "/gaia-common/"])
    is_candidate = "/candidates/" in path_str
    
    if is_live_code and not is_candidate:
        if os.getenv("BREAKGLASS_EMERGENCY") != "1":
            log_gaia_error(logger, "GAIA-MCP-030", f"Attempted write to live code path: {p}")
            raise PermissionError(
                "PRODUCTION LOCK ACTIVE: Direct writes to live services are forbidden. "
                "Modify code in /candidates/ and use the promotion pipeline instead."
            )

    # Allowlist: only the writable data volumes from docker-compose.yml
    allow_roots = [
        Path("/knowledge").resolve(), 
        Path("/sandbox").resolve(),
        Path("/gaia/GAIA_Project").resolve()
    ]
    if not any(str(p).startswith(str(a) + "/") or str(p) == str(a) for a in allow_roots):
        raise ValueError(f"Path not allowed — write_file is restricted to: {[str(a) for a in allow_roots]}")

    # Re-check after resolve to prevent symlink traversal
    real = p.resolve()
    if not any(str(real).startswith(str(a) + "/") or str(real) == str(a) for a in allow_roots):
        raise ValueError("Path not allowed after symlink resolution")

    # Create parent directories
    real.parent.mkdir(parents=True, exist_ok=True)

    # [Sovereign Shield] Pre-compile check
    _validate_python_content(str(real), content)

    # Write file
    with open(real, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"write_file: wrote {len(content)} bytes to {real}")
    return {"ok": True, "path": str(real), "bytes": len(content)}


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
    allow_roots = [Path("/knowledge").resolve(), Path("/gaia-common").resolve(), Path("/sandbox").resolve()]
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


# ── Knowledge Graph (temporal triple store) ──────────────────────────────

_kg_instance = None

def _get_kg():
    """Lazy-load the KnowledgeGraph singleton."""
    global _kg_instance
    if _kg_instance is not None:
        return _kg_instance
    from gaia_common.utils.knowledge_graph import KnowledgeGraph
    _kg_instance = KnowledgeGraph()
    return _kg_instance


def _kg_query_impl(params: dict) -> dict:
    """Query the knowledge graph for an entity's relationships."""
    kg = _get_kg()
    entity = params.get("entity") or params.get("subject", "")
    as_of = params.get("as_of")
    direction = params.get("direction", "both")
    if not entity:
        return {"ok": False, "error": "entity parameter required"}
    results = kg.query_entity(entity, as_of=as_of, direction=direction)
    return {"ok": True, "entity": entity, "facts": results, "count": len(results)}


def _kg_add_impl(params: dict) -> dict:
    """Add a fact (triple) to the knowledge graph."""
    kg = _get_kg()
    subject = params.get("subject", "")
    predicate = params.get("predicate", "")
    obj = params.get("object", "")
    if not (subject and predicate and obj):
        return {"ok": False, "error": "subject, predicate, and object are all required"}
    triple_id = kg.add_triple(
        subject=subject,
        predicate=predicate,
        obj=obj,
        valid_from=params.get("valid_from"),
        valid_to=params.get("valid_to"),
        confidence=float(params.get("confidence", 1.0)),
        source=params.get("source"),
    )
    return {"ok": True, "triple_id": triple_id}


def _kg_invalidate_impl(params: dict) -> dict:
    """Mark a fact as no longer valid."""
    kg = _get_kg()
    subject = params.get("subject", "")
    predicate = params.get("predicate", "")
    obj = params.get("object", "")
    ended = params.get("ended")
    if not (subject and predicate and obj):
        return {"ok": False, "error": "subject, predicate, and object are all required"}
    rows = kg.invalidate(subject, predicate, obj, ended=ended)
    return {"ok": True, "invalidated": rows}


def _kg_timeline_impl(params: dict) -> dict:
    """Get chronological facts, optionally filtered by entity."""
    kg = _get_kg()
    entity = params.get("entity")
    limit = int(params.get("limit", 100))
    facts = kg.timeline(entity_name=entity, limit=limit)
    return {"ok": True, "facts": facts, "count": len(facts)}


def _kg_stats_impl(params: dict) -> dict:
    """Get knowledge graph statistics."""
    kg = _get_kg()
    return {"ok": True, **kg.stats()}


# ── MemPalace (structured memory architecture) ────────────────────────────

_palace_instance = None

def _get_palace():
    """Lazy-load the MemPalace singleton."""
    global _palace_instance
    if _palace_instance is not None:
        return _palace_instance
    from gaia_common.utils.mempalace import MemPalace
    conf = Config()
    palace_config = conf.constants.get("MEMPALACE", {})
    _palace_instance = MemPalace(palace_config)
    return _palace_instance


def _palace_store_impl(params: dict) -> dict:
    """Store a memory in the palace."""
    palace = _get_palace()
    text = params.get("text", "")
    if not text:
        return {"ok": False, "error": "text parameter required"}
    source = params.get("source", "conversation")
    date_str = params.get("date")
    return palace.store(text, source=source, date_str=date_str)


def _palace_recall_impl(params: dict) -> dict:
    """Recall memories by text search with KG enrichment."""
    palace = _get_palace()
    query = params.get("query", "")
    if not query:
        return {"ok": False, "error": "query parameter required"}
    top_k = int(params.get("top_k", 5))
    return palace.recall(query, top_k=top_k)


def _palace_navigate_impl(params: dict) -> dict:
    """Browse the palace spatially."""
    palace = _get_palace()
    wing = params.get("wing")
    room = params.get("room")
    return palace.navigate(wing=wing, room=room)


def _palace_status_impl(params: dict) -> dict:
    """Get palace-wide stats."""
    palace = _get_palace()
    return palace.status()


def _list_knowledge_bases_impl(params: dict) -> dict:
    """Return all configured knowledge bases and their doc directories."""
    conf = Config()
    kbs = conf.constants.get("KNOWLEDGE_BASES", {})
    return {
        "ok": True,
        "knowledge_bases": {
            name: {"doc_dir": cfg.get("doc_dir"), "description": cfg.get("description", "")}
            for name, cfg in kbs.items()
        }
    }

def _list_tree_impl(params: dict):
    """Produce a bounded directory tree with depth/entry limits."""
    root = params.get("path") or "/knowledge"
    max_depth = int(params.get("max_depth", 3))
    max_entries = int(params.get("max_entries", 200))
    max_depth = max(1, min(max_depth, 6))
    max_entries = max(10, min(max_entries, 1000))

    root_path = Path(root).resolve()
    allow_roots = [Path("/knowledge").resolve(), Path("/gaia-common").resolve(), Path("/sandbox").resolve()]
    if not any(str(root_path).startswith(str(a)) for a in allow_roots):
        raise ValueError("Path not allowed")
    if not root_path.exists() or not root_path.is_dir():
        raise ValueError(f"{root_path} is not a directory")

    lines = []
    count = 0

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
        Path("/gaia-common").resolve(), 
        Path("/sandbox").resolve(),
        Path("/gaia/GAIA_Project").resolve()
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


# VECTOR_INDEX_PATH (from mcp_lite_server.py) is missing here. It was a global constant.
# Need to decide how to provide it. Perhaps through config.
VECTOR_INDEX_PATH = Path("./knowledge/vector_store/index.json") # Placeholder for now.


def _count_chars_impl(params: dict):
    """Count character occurrences — compensates for tokenization blindness."""
    text = params.get("text", "")
    char = params.get("char", "")
    if not text or not char:
        return {"error": "text and char required"}
    char = char[0].lower()  # Single character
    count = text.lower().count(char)
    positions = [i + 1 for i, c in enumerate(text.lower()) if c == char]
    return {
        "text": text,
        "char": char,
        "count": count,
        "positions": positions,
        "spelled_out": "-".join(text.lower()),
    }


def _recall_events_impl(params: dict):
    """Recall recent system events from episodic memory."""
    try:
        from gaia_common.event_buffer import EventBuffer
        buf = EventBuffer.instance()

        hours = float(params.get("hours", 6))
        limit = int(params.get("limit", 20))
        use_cfr = bool(params.get("cfr", False))

        if use_cfr:
            # Full detailed log for CFR analysis
            text = buf.full_formatted(hours=hours)
            events = buf.full(hours=hours)
            return {
                "ok": True,
                "mode": "cfr",
                "hours": hours,
                "event_count": len(events),
                "timeline": text,
            }
        else:
            # Concise recent summary
            events = buf.recent(n=limit)
            text = buf.recent_formatted(n=limit)
            return {
                "ok": True,
                "mode": "recent",
                "event_count": len(events),
                "timeline": text,
            }
    except ImportError:
        return {"ok": False, "error": "Event buffer not available"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _memory_status_impl(params: dict):
    """Summarize vector index status."""
    VECTOR_INDEX_PATH = Path("/knowledge/vector_store/index.json")
    vi = VectorIndexer.instance()
    vi.refresh_index()
    docs = vi.index.get("docs") or []
    embeddings = vi.index.get("embeddings") or []
    return {
        "ok": True,
        "doc_count": len(docs),
        "embedding_count": len(embeddings),
        "index_path": str(VECTOR_INDEX_PATH), # Uses VECTOR_INDEX_PATH
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
    Path("/knowledge/vector_store/index.json") # Define here for now
    vi = VectorIndexer.instance()
    ok = vi.build_index_from_docs()
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
    root = Path(params.get("root") or "/sandbox").resolve() # Default to sandbox for file operations
    max_depth = int(params.get("max_depth", 5))
    max_results = int(params.get("max_results", 50))
    max_depth = max(1, min(max_depth, 8))
    max_results = max(1, min(max_results, 200))

    FIND_FILES_ALLOW_ROOTS = [
        Path("/knowledge").resolve(),
        Path("/gaia-common").resolve(),
        Path("/sandbox").resolve(),
    ]

    if not any(str(root).startswith(str(a)) for a in FIND_FILES_ALLOW_ROOTS):
        allowed_str = ", ".join(str(a) for a in FIND_FILES_ALLOW_ROOTS)
        raise ValueError(
            f"Root not allowed: '{root}'. "
            f"find_files is restricted to: {allowed_str}"
        )
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

    # FIX: Use Config directly instead of the missing load_knowledge_bases import
    conf = Config() # Instantiate Config here
    
    # Access the KNOWLEDGE_BASES dictionary from constants
    knowledge_bases = conf.constants.get("KNOWLEDGE_BASES", {})
    kb_config = knowledge_bases.get(knowledge_base_name)

    if not kb_config:
        # Fallback: check if the user passed a direct path or if it's in a different config structure
        available = list(knowledge_bases.keys())
        logger.warning(f"Knowledge base '{knowledge_base_name}' not found. Available: {available}")
        return {
            "files": [],
            "error": f"Knowledge base '{knowledge_base_name}' not found.",
            "available_knowledge_bases": available  # ← tell the model what exists
        }

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

    return _gaia_helper.fragment_write(
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

    fragments = _gaia_helper.fragment_read(parent_request_id)
    return {"ok": True, "fragments": fragments, "count": len(fragments)}


def _fragment_assemble_impl(params: dict) -> dict:
    """Assemble fragments into a complete response."""
    parent_request_id = params.get("parent_request_id")
    if not parent_request_id:
        raise ValueError("parent_request_id is required")

    seam_overlap_check = bool(params.get("seam_overlap_check", True))
    return _gaia_helper.fragment_assemble(parent_request_id, seam_overlap_check=seam_overlap_check)


def _fragment_list_pending_impl(params: dict) -> dict:
    """List all pending (incomplete) fragment requests."""
    pending = _gaia_helper.fragment_list_pending()
    return {"ok": True, "pending": pending, "count": len(pending)}


def _fragment_clear_impl(params: dict) -> dict:
    """Clear fragments for a specific request or all fragments."""
    parent_request_id = params.get("parent_request_id")  # Optional
    result = _gaia_helper.fragment_clear(parent_request_id)
    return {"ok": True, "message": result}


# --- Study Mode / LoRA Adapter Tools (Gateway to gaia-study service) ---

_study_client = None


def _get_study_client():
    """Get the study service client (gateway to gaia-study)."""
    global _study_client
    if _study_client is None:
        _study_client = get_study_client()
    return _study_client


async def _study_start_impl(params: dict) -> dict:
    """Start a study session to learn from documents (via gaia-study gateway)."""
    adapter_name = params.get("adapter_name")
    documents = params.get("documents", [])

    if not adapter_name:
        raise ValueError("adapter_name is required")
    if not documents:
        raise ValueError("documents list is required and cannot be empty")

    client = _get_study_client()
    try:
        return await client.post("/study/start", {
            "adapter_name": adapter_name,
            "documents": documents,
            "tier": int(params.get("tier", 3)),
            "pillar": params.get("pillar", "general"),
            "description": params.get("description", ""),
            "max_steps": int(params.get("max_steps", 100)),
            "activation_triggers": params.get("activation_triggers", []),
            "tags": params.get("tags", []),
        })
    except Exception as e:
        logger.error(f"Failed to start study via gateway: {e}")
        return {"ok": False, "error": str(e), "errorCategory": "network", "isRetryable": True}


async def _study_status_impl(params: dict) -> dict:
    """Get current study mode status (via gaia-study gateway)."""
    client = _get_study_client()
    try:
        return await client.get("/study/status")
    except Exception as e:
        logger.error(f"Failed to get study status via gateway: {e}")
        return {"state": "error", "message": str(e)}


async def _study_cancel_impl(params: dict) -> dict:
    """Cancel an in-progress training session (via gaia-study gateway)."""
    client = _get_study_client()
    try:
        return await client.post("/study/cancel")
    except Exception as e:
        logger.error(f"Failed to cancel study via gateway: {e}")
        return {"ok": False, "message": str(e)}


async def _adapter_list_impl(params: dict) -> dict:
    """List available LoRA adapters (via gaia-study gateway)."""
    client = _get_study_client()
    tier = params.get("tier")
    try:
        query_params = {"tier": tier} if tier is not None else None
        return await client.get("/adapters", params=query_params)
    except Exception as e:
        logger.error(f"Failed to list adapters via gateway: {e}")
        return {"ok": False, "adapters": [], "error": str(e), "errorCategory": "network", "isRetryable": True}


async def _adapter_load_impl(params: dict) -> dict:
    """Load a LoRA adapter for use in generation (via gaia-study gateway)."""
    adapter_name = params.get("adapter_name")
    tier = int(params.get("tier", 3))

    if not adapter_name:
        raise ValueError("adapter_name is required")

    client = _get_study_client()
    try:
        return await client.post("/adapters/load", {
            "adapter_name": adapter_name,
            "tier": tier
        })
    except Exception as e:
        logger.error(f"Failed to load adapter via gateway: {e}")
        return {"ok": False, "error": str(e), "errorCategory": "network", "isRetryable": True}


async def _adapter_unload_impl(params: dict) -> dict:
    """Unload a LoRA adapter (via gaia-study gateway)."""
    adapter_name = params.get("adapter_name")

    if not adapter_name:
        raise ValueError("adapter_name is required")

    client = _get_study_client()
    try:
        return await client.post("/adapters/unload", {
            "adapter_name": adapter_name,
            "tier": int(params.get("tier", 3))
        })
    except Exception as e:
        logger.error(f"Failed to unload adapter via gateway: {e}")
        return {"ok": False, "error": str(e), "errorCategory": "network", "isRetryable": True}


async def _adapter_delete_impl(params: dict) -> dict:
    """Delete a LoRA adapter (via gaia-study gateway)."""
    adapter_name = params.get("adapter_name")
    tier = int(params.get("tier", 3))

    if not adapter_name:
        raise ValueError("adapter_name is required")

    client = _get_study_client()
    try:
        return await client.delete(f"/adapters/{adapter_name}", params={"tier": tier})
    except Exception as e:
        logger.error(f"Failed to delete adapter via gateway: {e}")
        return {"ok": False, "error": str(e), "errorCategory": "network", "isRetryable": True}


async def _adapter_info_impl(params: dict) -> dict:
    """Get detailed info about a specific adapter (via gaia-study gateway)."""
    adapter_name = params.get("adapter_name")
    tier = int(params.get("tier", 3))

    if not adapter_name:
        raise ValueError("adapter_name is required")

    client = _get_study_client()
    try:
        return await client.get(f"/adapters/{adapter_name}", params={"tier": tier})
    except Exception as e:
        logger.error(f"Failed to get adapter info via gateway: {e}")
        return {"ok": False, "error": str(e), "errorCategory": "network", "isRetryable": True}


# ── Self-Introspection Tools ────────────────────────────────────────────────

# Map service names to their log file paths inside the container
_LOG_FILE_MAP = {
    "gaia-core": "/logs/gaia-core.log",
    "gaia-web": "/logs/gaia-web.log",
    "gaia-mcp": "/logs/gaia-mcp.log",
    "gaia-study": "/logs/gaia-study.log",
    "discord": "/logs/discord_bot.log",
}

# ---------------------------------------------------------------------------
# CFR Conversation Review — ingest conversation history for deep review
# ---------------------------------------------------------------------------

def _cfr_review_conversation(params: dict) -> dict:
    """Export current session's conversation history to a temp file and CFR-ingest it.

    This allows GAIA to use CFR focus/compress/synthesize on her own conversation
    history — reviewing earlier exchanges at variable resolution.

    Params:
        session_id (str, optional): Session to review. Fetched from gaia-core.

    Returns:
        CFR ingest result with doc_id and section count.
    """
    import json as _json
    import tempfile
    from urllib.request import Request, urlopen

    session_id = params.get("session_id", "")

    # Step 1: Fetch conversation history from gaia-core
    try:
        core_url = os.environ.get("GAIA_CORE_ENDPOINT", "http://gaia-core:6415")

        # If no session_id, get the most recent active session
        if not session_id:
            req = Request(f"{core_url}/session/list", method="GET")
            with urlopen(req, timeout=10) as resp:
                sessions = _json.loads(resp.read())
                if isinstance(sessions, list) and sessions:
                    session_id = sessions[0].get("session_id", "")
                elif isinstance(sessions, dict):
                    # Might be a dict of session_id → data
                    session_id = next(iter(sessions), "")

        if not session_id:
            return {"ok": False, "error": "No active session found"}

        # Fetch history for this session
        req = Request(f"{core_url}/session/{session_id}/history", method="GET")
        with urlopen(req, timeout=10) as resp:
            history = _json.loads(resp.read())

        if not history:
            return {"ok": False, "error": f"No history for session {session_id}"}

    except Exception as e:
        # Fallback: try reading from the markdown chat log directly
        log_path = f"/logs/chat_history/{session_id}.md"
        if os.path.exists(log_path):
            with open(log_path) as f:
                history_text = f.read()
            if history_text.strip():
                # Write directly and ingest
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".md", prefix=f"conv_{session_id}_",
                    dir="/shared/gaia_state/cfr", delete=False,
                )
                tmp.write(history_text)
                tmp.close()

                doc_id = f"conversation_{session_id[:16]}"
                result = _cfr_manager.ingest(file_path=tmp.name, doc_id=doc_id)
                result["session_id"] = session_id
                result["source"] = "chat_log_file"
                return result

        return {"ok": False, "error": f"Could not fetch session history: {e}"}

    # Step 2: Format history as readable markdown
    lines = [f"# Conversation History: {session_id}\n"]
    for msg in history:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        ts = msg.get("timestamp", "")
        lines.append(f"### {role}" + (f" ({ts})" if ts else ""))
        lines.append(content)
        lines.append("")

    history_text = "\n".join(lines)

    # Step 3: Write to temp file
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", prefix=f"conv_{session_id[:16]}_",
        dir="/shared/gaia_state/cfr", delete=False,
    )
    tmp.write(history_text)
    tmp.close()

    # Step 4: CFR ingest
    doc_id = f"conversation_{session_id[:16]}"
    result = _cfr_manager.ingest(file_path=tmp.name, doc_id=doc_id)
    result["session_id"] = session_id
    result["source"] = "session_manager"
    result["message_count"] = len(history)

    logger.info("CFR conversation review: session=%s, %d messages, %d sections",
                session_id, len(history), result.get("section_count", 0))

    return result


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
        return {"ok": False, "error": f"Failed to read log file: {e}", "errorCategory": "resource", "isRetryable": True}

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


# ── Promotion & Blueprint Tools ─────────────────────────────────────────────


def _generate_blueprint_impl(params: Dict) -> Dict:
    """Generate a candidate blueprint for a service from source analysis."""
    service_id = params.get("service_id")
    source_dir = params.get("source_dir")
    role_hint = params.get("role_hint", "")

    if not service_id:
        raise ValueError("service_id is required")

    # Auto-detect source_dir if not provided
    if not source_dir:
        pkg_name = service_id.replace("-", "_")
        candidates = [
            f"/gaia/GAIA_Project/candidates/{service_id}/{pkg_name}",
            f"/gaia/GAIA_Project/{service_id}/{pkg_name}",
        ]
        source_dir = next((p for p in candidates if Path(p).exists()), None)
        if not source_dir:
            return {"ok": False, "error": f"Could not find source directory for {service_id}"}

    try:
        from gaia_common.utils.blueprint_generator import generate_candidate_blueprint
        from gaia_common.utils.blueprint_io import save_blueprint

        bp = generate_candidate_blueprint(
            service_id=service_id,
            source_dir=source_dir,
            role_hint=role_hint,
        )
        path = save_blueprint(bp, candidate=True)

        return {
            "ok": True,
            "service_id": service_id,
            "path": str(path),
            "interfaces": len(bp.interfaces),
            "inbound": len(bp.inbound_interfaces()),
            "outbound": len(bp.outbound_interfaces()),
            "dependencies": len(bp.dependencies.services),
            "failure_modes": len(bp.failure_modes),
            "source_files": len(bp.source_files),
            "intent": bp.intent.purpose if bp.intent else None,
        }
    except Exception as e:
        logger.error("Blueprint generation failed for %s: %s", service_id, e, exc_info=True)
        return {"ok": False, "error": str(e)}


def _assess_promotion_impl(params: Dict) -> Dict:
    """Run promotion readiness assessment for a candidate service."""
    service_id = params.get("service_id")
    if not service_id:
        raise ValueError("service_id is required")

    try:
        from gaia_common.utils.promotion_readiness import assess_promotion_readiness
        report = assess_promotion_readiness(service_id)
        return {
            "ok": True,
            **report.to_dict(),
        }
    except Exception as e:
        logger.error("Promotion assessment failed for %s: %s", service_id, e, exc_info=True)
        return {"ok": False, "error": str(e)}


# ── Promotion Lifecycle Tools ────────────────────────────────────────────────


def _promotion_create_request_impl(params: Dict) -> Dict:
    """Create a promotion request after readiness assessment."""
    from gaia_common.utils.promotion_request import (
        create_promotion_request, load_pending_request,
    )

    service_id = params.get("service_id")
    verdict = params.get("verdict")
    recommendation = params.get("recommendation")
    pipeline_cmd = params.get("pipeline_cmd")
    check_summary = params.get("check_summary")

    if not all([service_id, verdict, recommendation, pipeline_cmd, check_summary]):
        raise ValueError("All fields required: service_id, verdict, recommendation, pipeline_cmd, check_summary")

    # Reject if assessment verdict is not_ready
    if verdict == "not_ready":
        return {
            "ok": False,
            "error": "Cannot create promotion request with verdict 'not_ready'. "
                     "Fix issues and re-assess first.",
        }

    # Check for existing active request
    existing = load_pending_request(service_id)
    if existing:
        return {
            "ok": False,
            "error": f"Active request already exists: {existing.request_id} (status={existing.status}). "
                     "Resolve it before creating a new one.",
        }

    req = create_promotion_request(
        service_id=service_id,
        verdict=verdict,
        recommendation=recommendation,
        pipeline_cmd=pipeline_cmd,
        check_summary=check_summary,
    )
    return {
        "ok": True,
        "request_id": req.request_id,
        "status": req.status,
        "message": "Promotion request created. Awaiting human approval (Gate 1).",
    }


def _promotion_list_requests_impl(params: Dict) -> Dict:
    """List promotion requests with optional filters."""
    from gaia_common.utils.promotion_request import list_requests

    service_id = params.get("service_id")
    status_filter = params.get("status_filter")

    requests = list_requests(service_id=service_id, status_filter=status_filter)
    return {
        "ok": True,
        "count": len(requests),
        "requests": [
            {
                "request_id": r.request_id,
                "service_id": r.service_id,
                "status": r.status,
                "verdict": r.verdict,
                "requested_at": r.requested_at,
            }
            for r in requests
        ],
    }


def _promotion_request_status_impl(params: Dict) -> Dict:
    """Get full details of a specific promotion request."""
    from gaia_common.utils.promotion_request import load_request

    request_id = params.get("request_id")
    if not request_id:
        raise ValueError("request_id is required")

    req = load_request(request_id)
    if req is None:
        return {"ok": False, "error": f"Request not found: {request_id}"}

    return {
        "ok": True,
        **req.to_dict(),
    }


def _send_discord_message_impl(params: dict) -> dict:
    """Send a message to Discord via webhook."""
    from gaia_common.integrations.discord import DiscordConfig, DiscordWebhookSender
    
    content = params.get("content")
    if not content:
        raise ValueError("content is required")
    
    config = DiscordConfig.from_env()
    if not config.webhook_url:
        raise ValueError("DISCORD_WEBHOOK_URL is not configured")

    sender = DiscordWebhookSender(config.webhook_url, config.bot_name, config.avatar_url)
    success = sender.send(content)
    return {"ok": success}


def _replace_impl(params: dict) -> dict:
    """Surgically replace text in a file with allowlist and safety checks."""
    path = params.get("file_path")
    old_string = params.get("old_string")
    new_string = params.get("new_string")
    allow_multiple = bool(params.get("allow_multiple", False))

    if not all([path, old_string is not None, new_string is not None]):
        raise ValueError("file_path, old_string, and new_string are required")

    # 1. Resolve and validate path
    p = Path(path).resolve()
    
    # --- PRODUCTION LOCK (Sovereign Shield) ---
    path_str = str(p)
    is_live_code = any(segment in path_str for segment in ["/gaia-core/", "/gaia-web/", "/gaia-mcp/", "/gaia-common/"])
    is_candidate = "/candidates/" in path_str
    
    if is_live_code and not is_candidate:
        if os.getenv("BREAKGLASS_EMERGENCY") != "1":
            logger.critical(f"🛡️ PRODUCTION LOCK: Attempted replace in live code path: {p}")
            raise PermissionError(
                "PRODUCTION LOCK ACTIVE: Direct modifications to live services are forbidden. "
                "Modify code in /candidates/ and use the promotion pipeline instead."
            )

    allow_roots = [
        Path("/knowledge").resolve(), 
        Path("/sandbox").resolve(),
        Path("/gaia/GAIA_Project").resolve()
    ]
    if not any(str(p).startswith(str(a) + "/") or str(p) == str(a) for a in allow_roots):
        raise ValueError("Path not allowed for replace")

    if not p.is_file():
        raise ValueError(f"{path} is not a file")

    # 2. Perform the replacement
    content = p.read_text(encoding="utf-8")
    
    count = content.count(old_string)
    if count == 0:
        raise ValueError(f"Could not find exact match for: {old_string}")
    if count > 1 and not allow_multiple:
        raise ValueError(f"Found {count} occurrences of old_string. Set allow_multiple=True to replace all.")

    new_content = content.replace(old_string, new_string)

    # 3. [Sovereign Shield] Pre-compile check
    _validate_python_content(str(p), new_content)

    # 4. Save the file
    p.write_text(new_content, encoding="utf-8")
    
    logger.info(f"replace: modified {p} ({count} occurrences replaced)")
    return {"ok": True, "path": str(p), "count": count}
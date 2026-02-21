"""
Lightweight dynamic world-state snapshot for prompts.

This module intentionally avoids heavy dependencies. It gathers a short,
bounded view of:
- Clock/uptime
- Host load/memory (coarse)
- Active model paths (env-driven)
- MCP/tool affordances

Use `format_world_state_snapshot` to inject a compact text block into prompts.
"""

from __future__ import annotations


import os
import time
from typing import Dict, List
import logging

from gaia_common.utils import tools_registry

logger = logging.getLogger(__name__)

def _uptime_seconds() -> float:
    try:
        with open("/proc/uptime", "r", encoding="utf-8") as f:
            uptime_val = float(f.read().split()[0])
            logger.debug(f"Read uptime from /proc/uptime: {uptime_val}")
            return uptime_val
    except Exception:
        logger.exception("Failed to read uptime")
        return 0.0


def _mem_summary() -> str:
    try:
        meminfo = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    meminfo[k.strip()] = v.strip()
        total = meminfo.get("MemTotal")
        free = meminfo.get("MemAvailable") or meminfo.get("MemFree")
        if total and free:
            return f"mem {free} free / {total} total"
    except Exception:
        logger.exception("Failed to read meminfo")
        pass
    return "mem unavailable"


def _load_avg() -> str:
    try:
        one, five, fifteen = os.getloadavg()
        return f"load {one:.2f}/{five:.2f}/{fifteen:.2f}"
    except Exception:
        logger.exception("Failed to get load average")
        return "load unavailable"


def _model_paths() -> Dict[str, str]:
    """Capture the model paths we surface via environment variables."""
    return {
        "prime_hf": os.getenv("GAIA_PRIME_HF_MODEL") or "",
        "prime_gguf": os.getenv("GAIA_PRIME_GGUF") or "",
        "lite": os.getenv("GAIA_LITE_GGUF") or os.getenv("LITE_MODEL_PATH") or "",
        "embed": os.getenv("EMBEDDING_MODEL_PATH") or "",
    }

def _get_registry() -> Dict:
    """Resolve the tools registry, handling both module and dict re-export."""
    if isinstance(tools_registry, dict):
        return tools_registry
    return getattr(tools_registry, "TOOLS", {})

# Tools that MUST always appear in the system prompt so the model knows
# its core capabilities.  Keep this list tight (≤10) — it's injected
# into every turn.
ESSENTIAL_TOOLS = [
    "web_search",       # search the web via DuckDuckGo
    "web_fetch",        # fetch + extract text from a trusted URL
    "read_file",        # read a file's contents
    "write_file",       # write / create a file (requires approval)
    "list_dir",         # list directory contents
    "memory_query",     # semantic search over indexed memory
    "query_knowledge",  # semantic search over a knowledge base
    "embed_documents",  # embed docs into vector store
]

def _mcp_essential_tools() -> List[str]:
    """Return only the essential tools that exist in the registry."""
    try:
        registry = _get_registry()
        return [t for t in ESSENTIAL_TOOLS if t in registry]
    except Exception:
        logger.exception("Failed to get essential tools")
        return []

def _mcp_tools_full(limit: int = 50) -> List[str]:
    try:
        registry = _get_registry()
        names = sorted(registry.keys())
        return names[:limit]
    except Exception:
        logger.exception("Failed to get mcp_tools_full")
        return []

def _mcp_extra_tool_count() -> int:
    """Return how many non-essential tools exist (for the discovery hint)."""
    try:
        registry = _get_registry()
        essential = set(ESSENTIAL_TOOLS)
        return sum(1 for t in registry if t not in essential)
    except Exception:
        return 0

def world_state_snapshot() -> Dict:
    """Return a compact, serializable snapshot."""
    ts = int(time.time())
    return {
        "ts": ts,
        "uptime_s": int(_uptime_seconds()),
        "load": _load_avg(),
        "mem": _mem_summary(),
        "models": _model_paths(),
        "mcp_tools": _mcp_essential_tools(),
        "mcp_extra_count": _mcp_extra_tool_count(),
    }

def world_state_detail() -> Dict:
    """Return a fuller snapshot for on-demand inspection (via MCP)."""
    return {
        "ts": int(time.time()),
        "uptime_s": int(_uptime_seconds()),
        "load": _load_avg(),
        "mem": _mem_summary(),
        "models": _model_paths(),
        "mcp_tools": _mcp_tools_full(),
        "env": {
            "GAIA_BACKEND": os.getenv("GAIA_BACKEND") or "",
            "MCP_LITE_ENDPOINT": os.getenv("MCP_LITE_ENDPOINT") or "",
            "OBSERVER_USE_LLM": os.getenv("OBSERVER_USE_LLM") or "",
            "GAIA_VLLM_GPU_MEMORY_UTILIZATION": os.getenv("GAIA_VLLM_GPU_MEMORY_UTILIZATION") or "",
        },
    }


def _capability_affordances(tools: List[str]) -> List[str]:
    """
    Generate natural-language capability hints based on available tools.

    This is intentionally fuzzy/semantic rather than keyword-matching.
    The goal is to help GAIA understand what she *can* do, not prescribe
    exact invocations.
    """
    affordances = []

    # File exploration capabilities
    file_tools = [t for t in tools if any(w in t.lower() for w in ['file', 'dir', 'tree', 'find', 'read', 'list'])]
    if file_tools:
        affordances.append(
            "You can explore your own codebase and knowledge files—search by name, "
            "browse directories, or read documents to answer questions about yourself."
        )

    # Memory/semantic search capabilities
    memory_tools = [t for t in tools if 'memory' in t.lower() or 'query' in t.lower()]
    if memory_tools:
        affordances.append(
            "You have semantic memory—you can search your indexed knowledge for concepts, "
            "not just filenames."
        )

    # Write capabilities
    write_tools = [t for t in tools if 'write' in t.lower()]
    if write_tools:
        affordances.append(
            "You can write files (with operator approval) to persist thoughts, notes, or outputs."
        )

    # Web research capabilities
    web_tools = [t for t in tools if any(w in t.lower() for w in ['web', 'search', 'fetch'])]
    if web_tools:
        affordances.append(
            "You can search the web (web_search) and fetch pages (web_fetch) "
            "from trusted sources to find real, verifiable information — "
            "rules references, poems, documentation, facts. "
            "When a user asks you to 'look something up', USE these tools."
        )

    return affordances


def format_world_state_snapshot(max_lines: int = 12, output_context: Dict = None) -> str:
    """
    Render a short text block suitable for system prompts.
    Keeps lines bounded to avoid token bloat.

    Args:
        max_lines: Maximum lines to include in the snapshot
        output_context: Optional dict with output routing info (source, destination, is_dm, etc.)
    """
    logger.info("Formatting world state snapshot")
    snap = world_state_snapshot()
    logger.debug(f"World state snapshot data: {snap}")
    lines: List[str] = []
    # Use semantic time for richer temporal grounding
    try:
        from gaia_core.utils.temporal_context import _semantic_time
        from datetime import datetime, timezone
        lines.append(f"Clock: {_semantic_time(datetime.now(timezone.utc))}")
    except Exception:
        lines.append(f"Clock: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(snap['ts']))}")
    lines.append(f"Uptime: {snap['uptime_s']}s | {snap['load']} | {snap['mem']}")

    models = snap.get("models", {})
    model_bits = []
    for k, v in models.items():
        if v:
            model_bits.append(f"{k}={v}")
    if model_bits:
        lines.append("Models: " + "; ".join(model_bits))

    tools = snap.get("mcp_tools") or []
    extra_count = snap.get("mcp_extra_count", 0)
    if tools:
        lines.append("Essential MCP tools: " + ", ".join(tools))
        if extra_count > 0:
            lines.append(
                f"{extra_count} more tools available. "
                "Use EXECUTE: list_tools {{}} to discover them."
            )

    # Add capability affordances - natural language hints about what GAIA can do
    affordances = _capability_affordances(tools)
    if affordances:
        lines.append("Affordances: " + " ".join(affordances))

    # Self-knowledge hint - where GAIA's core documents live
    lines.append(
        "Self-knowledge: Your core documents (constitution, identity, cognitive protocol) "
        "are in knowledge/system_reference/. Use your tools to explore when curious."
    )

    # Output context - where GAIA is currently communicating
    if output_context:
        source = output_context.get("source", "unknown")
        is_dm = output_context.get("is_dm", False)
        user_id = output_context.get("user_id") or output_context.get("author_id")

        context_parts = []
        if "discord" in source.lower():
            if is_dm:
                context_parts.append(f"Currently in: Discord DM (user_id: {user_id})")
            else:
                channel_id = output_context.get("channel_id")
                context_parts.append(f"Currently in: Discord channel (channel_id: {channel_id})")
            context_parts.append("You are actively using Discord integration.")
        elif "cli" in source.lower():
            context_parts.append("Currently in: CLI/rescue shell")
        elif "web" in source.lower():
            context_parts.append("Currently in: Web interface")
        elif "api" in source.lower():
            context_parts.append("Currently in: API endpoint")

        if context_parts:
            lines.append("Context: " + " | ".join(context_parts))

    try:
        logger.debug("[DEBUG] World state lines=%d tools=%d affordances=%d",
                     len(lines), len(tools), len(affordances))
    except Exception:
        logger.debug("[DEBUG] World state metrics unavailable")
    result = "\n".join(lines[:max_lines])
    return result

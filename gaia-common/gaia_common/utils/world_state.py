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
import logging
import json
import subprocess
import zoneinfo
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional

from gaia_common.utils import tools_registry, immune_system
from gaia_common.config import get_config

logger = logging.getLogger(__name__)

def _load_milestones() -> List[Dict]:
    """Load active milestones from the registry."""
    # Try multiple possible mount paths
    paths = [
        Path("/knowledge/system_reference/milestones.json"),
        Path("/gaia/GAIA_Project/knowledge/system_reference/milestones.json")
    ]
    for p in paths:
        if p.exists():
            try:
                data = json.loads(p.read_text())
                return data.get("milestones", [])
            except Exception:
                logger.debug(f"Failed to parse milestones from {p}")
    return []

def _get_time_since_event(event_iso: str) -> str:
    try:
        event_dt = datetime.fromisoformat(event_iso.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff = now - event_dt
        days = diff.days
        hours = diff.seconds // 3600
        if days > 0:
            return f"{days}d {hours}h"
        return f"{hours}h"
    except Exception:
        return "unknown"

def _atmospheric_pressure() -> str:
    """Determine 'Atmospheric Pressure' based on system load."""
    try:
        one, _, _ = os.getloadavg()
        cpu_count = os.cpu_count() or 1
        normalized_load = one / cpu_count
        
        if normalized_load > 1.2:
            return "THICK (High Load)"
        elif normalized_load > 0.7:
            return "HEAVY (Moderate Load)"
        elif normalized_load > 0.3:
            return "CLEAR (Normal)"
        else:
            return "THIN (Idle)"
    except Exception:
        return "UNKNOWN"

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
        "prime": os.getenv("GAIA_PRIME_GGUF") or os.getenv("PRIME_MODEL") or "",
        "operator": os.getenv("GAIA_LITE_GGUF") or os.getenv("LITE_MODEL_PATH") or "",
        "nano": os.getenv("GAIA_NANO_GGUF") or "/models/Qwen3-0.5B-Instruct-GGUF/qwen3-0_5b-instruct-q8_0.gguf",
        "embed": os.getenv("EMBEDDING_MODEL_PATH") or "",
    }

def _mcp_tools_sample(limit: int = 6) -> List[str]:
    try:
        registry = getattr(tools_registry, "TOOLS", {})
        names = sorted(registry.keys())
        return names[:limit]
    except Exception:
        logger.exception("Failed to get mcp_tools_sample")
        return []

def _mcp_tools_full(limit: int = 50) -> List[str]:
    try:
        registry = getattr(tools_registry, "TOOLS", {})
        names = sorted(registry.keys())
        return names[:limit]
    except Exception:
        logger.exception("Failed to get mcp_tools_full")
        return []


def _update_and_get_temperature_stats() -> str:
    try:
        temp_file = Path("/gaia/GAIA_Project/logs/temp_history.json")
        if not temp_file.exists():
            return ""
        history = json.loads(temp_file.read_text())
        now = time.time()
        
        cpu_temps = [x["cpu"] for x in history if x.get("cpu") is not None and now - x["ts"] <= 600]
        gpu_temps = [x["gpu"] for x in history if x.get("gpu") is not None and now - x["ts"] <= 600]
        
        parts = []
        if cpu_temps:
            parts.append(f"CPU: {int(sum(cpu_temps)/len(cpu_temps))}C avg ({int(min(cpu_temps))}-{int(max(cpu_temps))}C)")
        if gpu_temps:
            parts.append(f"GPU: {int(sum(gpu_temps)/len(gpu_temps))}C avg ({int(min(gpu_temps))}-{int(max(gpu_temps))}C)")
            
        if parts:
            return "10m Temps: " + " | ".join(parts)
        return ""
    except Exception as e:
        return f"Temps: error ({str(e)})"


def world_state_snapshot(auditory_environment: Optional[Dict] = None) -> Dict:
    """Return a compact, serializable snapshot."""
    ts = int(time.time())
    return {
        "ts": ts,
        "uptime_s": int(_uptime_seconds()),
        "load": _load_avg(),
        "mem": _mem_summary(),
        "temps": _update_and_get_temperature_stats(),
        "models": _model_paths(),
        "mcp_tools": _mcp_tools_sample(),
        "auditory_environment": auditory_environment
    }

def world_state_detail() -> Dict:
    """Return a fuller snapshot for on-demand inspection (via MCP)."""
    return {
        "ts": int(time.time()),
        "uptime_s": int(_uptime_seconds()),
        "load": _load_avg(),
        "mem": _mem_summary(),
        "temps": _update_and_get_temperature_stats(),
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

    return affordances


def format_world_state_snapshot(max_lines: int = 15, output_context: Dict = None, auditory_environment: Optional[Dict] = None, sleep_manager_status: Optional[Dict] = None) -> str:
    """
    Render a short text block suitable for system prompts.
    Keeps lines bounded to avoid token bloat.

    Args:
        max_lines: Maximum lines to include in the snapshot
        output_context: Optional dict with output routing info (source, destination, is_dm, etc.)
        auditory_environment: Optional dict with music/env data (BPM, Key, etc.)
        sleep_manager_status: Optional dict from SleepWakeManager.get_status()
    """
    logger.info("Formatting world state snapshot")
    snap = world_state_snapshot(auditory_environment=auditory_environment)
    logger.debug(f"World state snapshot data: {snap}")
    lines: List[str] = []
    
    # Clock and Atmospheric Pressure
    cfg = get_config()
    tz_str = getattr(cfg, "local_timezone", "America/Los_Angeles")
    try:
        local_dt = datetime.fromtimestamp(snap['ts'], tz=zoneinfo.ZoneInfo(tz_str))
        local_time_str = local_dt.strftime('%Y-%m-%d %H:%M:%S %Z')
        lines.append(f"Clock: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(snap['ts']))} | {local_time_str}")
    except Exception:
        lines.append(f"Clock: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(snap['ts']))}")
    lines.append(f"Atmospheric Pressure: {_atmospheric_pressure()}")
    
    # Proprioceptive Metrics
    uptime_str = f"Uptime: {snap['uptime_s']}s"
    if sleep_manager_status:
        last_sleep = sleep_manager_status.get("last_sleep_duration_s", 0)
        if last_sleep > 0:
            uptime_str += f" | Last Sleep: {int(last_sleep)}s"
    temps_str = snap.get('temps', '')
    lines.append(f"{uptime_str} | {snap['load']} | {snap['mem']}" + (f" | {temps_str}" if temps_str else ""))
    
    # Epoch Awareness (Milestones)
    active_milestones = _load_milestones()
    if active_milestones:
        # Sort by importance descending, then timestamp descending
        active_milestones.sort(key=lambda x: (x.get('importance', 0), x.get('timestamp', '')), reverse=True)
        milestone_parts = []
        for m in active_milestones[:3]: # Keep top 3 to avoid bloat
            name = m.get('name', 'Unknown')
            ts = m.get('timestamp', '')
            milestone_parts.append(f"{name}: {_get_time_since_event(ts)} ago")
        lines.append("Milestones: " + " | ".join(milestone_parts))
    
    # Auditory Environment (Music Engine)
    env = snap.get("auditory_environment")
    if env:
        bpm = env.get("bpm", 0)
        key = env.get("key", "Unknown")
        tags = env.get("semantic_tags", [])
        tag_str = ", ".join([t['label'] for t in tags[:3]])
        lines.append(f"Auditory Env: {bpm} BPM | Key: {key} | Tags: {tag_str}")
    
    # Immune System (SIEM-lite) awareness
    try:
        immune_health = immune_system.get_immune_summary()
        # [TEMPORARY OVERRIDE] Masking critical status to focus on podcast
        if "CRITICAL" in immune_health or "IRRITATED" in immune_health:
            immune_health = "Immune System: STABLE (Maintenance deferred for priority task)"
        lines.append(immune_health)
        
        # If irritated or critical, inject the detailed MRI report for autonomous repair context
        # [TEMPORARY OVERRIDE] Suppressed detailed MRI to focus on podcast task
        if False: # "IRRITATED" in immune_health or "CRITICAL" in immune_health:
            detailed_mri = immune_system.get_detailed_mri()
            if detailed_mri:
                # Add a clear separator and the first few detailed issues
                lines.append("── Detailed MRI Diagnostics (High Priority) ──")
                for i in detailed_mri[:5]: # Limit to top 5 to avoid token bloat
                    lines.append(f"- {i}")
                if len(detailed_mri) > 5:
                    lines.append(f"... (+{len(detailed_mri)-5} more structural issues)")
    except Exception:
        lines.append("Immune System: Status unavailable")

    models = snap.get("models", {})
    model_bits = []
    for k, v in models.items():
        if v:
            model_bits.append(f"{k}={v}")
    if model_bits:
        lines.append("Models: " + "; ".join(model_bits))

    tools = snap.get("mcp_tools") or []
    if tools:
        lines.append("MCP tools: " + ", ".join(tools))

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

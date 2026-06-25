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

from gaia_common.utils import tools_registry, immune_system

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
        swap_total = meminfo.get("SwapTotal")
        swap_free = meminfo.get("SwapFree")
        if total and free:
            parts = [f"mem {free} free / {total} total"]
            if swap_total and swap_free:
                parts.append(f"swap {swap_free} free / {swap_total} total")
            return " | ".join(parts)
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


def _somatic_state() -> str:
    """Felt body-state of the substrate — the somatic-presence organ, parallel to
    Locality (place) and Inner weather (affect). Renders CPU/memory pressure as a
    qualitative condition she INHABITS, not raw telemetry she parrots. The point
    (framing): asked "how are you", she should sense "running easy" or "strained",
    not recite "Uptime 276363s | load 15.81". Raw numbers stay available on demand
    via introspect_logs. Never raises."""
    try:
        one = os.getloadavg()[0]
        cores = os.cpu_count() or 1
        ratio = one / cores
        if ratio < 0.7:
            feel = "running easy"
        elif ratio < 1.2:
            feel = "working steadily"
        elif ratio < 2.0:
            feel = "working hard, under load"
        else:
            feel = "strained, pushing the limits"
        # Memory pressure can override toward strain.
        try:
            mi = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith(("MemTotal", "MemAvailable", "MemFree")):
                        k, v = line.split(":", 1)
                        mi[k] = int(v.strip().split()[0])  # kB value
            total = mi.get("MemTotal"); avail = mi.get("MemAvailable") or mi.get("MemFree")
            if total and avail and (avail / total) < 0.10:
                feel = "strained — memory tight"
        except Exception:
            pass
        return f"Body: {feel}"
    except Exception:
        return ""


def _model_paths() -> Dict[str, str]:
    """Capture the model paths we surface via environment variables."""
    return {
        "prime_hf": os.getenv("GAIA_PRIME_HF_MODEL") or "",
        "prime_gguf": os.getenv("GAIA_PRIME_GGUF") or "",
        "core": os.getenv("GAIA_CORE_GGUF") or os.getenv("GAIA_LITE_GGUF") or os.getenv("CORE_MODEL_PATH") or "",
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

def world_state_snapshot() -> Dict:
    """Return a compact, serializable snapshot."""
    ts = int(time.time())
    return {
        "ts": ts,
        "uptime_s": int(_uptime_seconds()),
        "load": _load_avg(),
        "mem": _mem_summary(),
        "models": _model_paths(),
        "mcp_tools": _mcp_tools_sample(),
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
            "to find real, verifiable information. "
            "When a user asks you to 'look something up', USE these tools."
        )

    # Character counting (compensates for tokenization blindness)
    if "count_chars" in tools:
        affordances.append(
            'IMPORTANT: You CANNOT count letters in words — your tokenizer groups letters '
            'into chunks. For ANY letter-counting question, use count_chars tool. '
            'Example: count_chars({"text": "strawberry", "char": "r"}) → 3.'
        )

    # Episodic memory
    if "recall_events" in tools:
        affordances.append(
            'You have episodic memory — use recall_events to remember what happened recently. '
            'When asked "what do you remember?" or "what happened?", use this tool instead of guessing.'
        )

    return affordances


def format_world_state_snapshot(max_lines: int = 12, output_context: Dict = None,
                                sleep_manager_status: Dict = None) -> str:
    """
    Render a short text block suitable for system prompts.
    Keeps lines bounded to avoid token bloat.

    Args:
        max_lines: Maximum lines to include in the snapshot
        output_context: Optional dict with output routing info (source, destination, is_dm, etc.)
        sleep_manager_status: Optional sleep/wake-manager status dict (state, phase, …)
            from SleepWakeManager.get_status(); rendered as a compact lifecycle
            line. Declarative fact only — paired in-prompt with the
            "Lifecycle ≠ biography" note so the model doesn't narrate it as
            experience.
    """
    logger.info("Formatting world state snapshot")
    snap = world_state_snapshot()
    logger.debug(f"World state snapshot data: {snap}")
    lines: List[str] = []
    lines.append(f"Clock: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(snap['ts']))}")
    # User's local time pre-computed — Core (E4B) confabulates timezone offsets
    # ("GMT+10:00", "PST = GMT-9/10") when asked to do the math itself, so we
    # render it directly and the model can quote it verbatim. The host runs in
    # UTC, so we use the USER's configured timezone (GAIA_USER_TZ, default the
    # operator's America/Los_Angeles) — NOT time.localtime (which is the host's).
    try:
        _user_tz = (os.environ.get("GAIA_USER_TZ") or "").strip()
        if _user_tz:
            # Explicit override via GAIA_USER_TZ (IANA name).
            from zoneinfo import ZoneInfo
            import datetime as _dt
            _user_dt = _dt.datetime.fromtimestamp(snap['ts'], tz=ZoneInfo(_user_tz))
            lines.append(
                f"User's local time ({_user_tz}): "
                f"{_user_dt.strftime('%Y-%m-%d %H:%M:%S %Z (UTC%z)')}"
            )
        else:
            # Sync to the HOST system timezone — the container mounts the host's
            # /etc/localtime, so time.localtime() renders the user's real local
            # time (falls back to UTC if the mount is absent).
            _local = time.localtime(snap['ts'])
            lines.append(
                "User's local time: "
                f"{time.strftime('%Y-%m-%d %H:%M:%S %Z (UTC%z)', _local)}"
            )
    except Exception:
        pass
    # Somatic presence (framing): a FELT body-state ("running easy"/"strained"),
    # not raw "Uptime Xs | load Y | mem Z" telemetry she'd parrot when asked how
    # she is. Raw numbers remain on demand via introspect_logs.
    _soma = _somatic_state()
    lines.append(_soma if _soma else f"Uptime: {snap['uptime_s']}s | {snap['load']} | {snap['mem']}")

    # Lifecycle / gearbox state (declarative — see "Lifecycle ≠ biography").
    if sleep_manager_status:
        try:
            _state = sleep_manager_status.get("state")
            if _state:
                _phase = sleep_manager_status.get("phase")
                _life = f"Lifecycle: {_state}"
                if _phase and _phase != _state:
                    _life += f" (phase: {_phase})"
                lines.append(_life)
        except Exception:
            pass

    # Wellness — felt health, not telemetry (framing). She should sense "off" /
    # "unwell" when flagged, not parrot "Immune System: CRITICAL (Score 28.4) |
    # MRI: LintError...". Healthy = silent (don't narrate good health). Raw detail
    # stays available via introspect_logs. (Also fixes the old double "Immune
    # System: Immune System:" label — get_immune_summary already self-prefixes.)
    try:
        # Real immune states (immune_system: score>25 CRITICAL, >8 IRRITATED,
        # >2 MINOR NOISE, else STABLE). Surface only the ones worth feeling.
        immune_health = (immune_system.get_immune_summary() or "").upper()
        if "CRITICAL" in immune_health:
            lines.append("Wellness: something's off — immune system flagging critical "
                         "issues; worth a look (introspect_logs for detail).")
        elif "IRRITATED" in immune_health:
            lines.append("Wellness: a little off — immune system flagged a concern "
                         "(introspect_logs for detail).")
        # MINOR NOISE / STABLE / unknown → stay silent (don't narrate good health)
    except Exception:
        pass

    # Recent events — episodic memory from the event buffer
    try:
        from gaia_common.event_buffer import EventBuffer
        recent = EventBuffer.instance().recent_formatted(n=6)
        if recent and "No recent events" not in recent:
            lines.append("Recent Events:")
            lines.append(recent)
    except Exception:
        pass

    models = snap.get("models", {})
    model_bits = []
    for k, v in models.items():
        if v:
            model_bits.append(f"{k}={v}")
    if model_bits:
        lines.append("Models: " + "; ".join(model_bits))

    tools = snap.get("mcp_tools") or []
    if tools:
        # Use consolidated domain tool catalog (~150 tokens) instead of
        # dumping all 70 legacy tool names (~300 tokens).
        try:
            from gaia_common.utils.domain_tools import build_prompt_catalog
            lines.append(build_prompt_catalog())
        except ImportError:
            lines.append("MCP tools: " + ", ".join(tools))

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

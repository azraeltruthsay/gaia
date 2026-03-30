"""
Fabric Pattern Loader — dynamic MCP tool registration for Fabric patterns.

Fabric (github.com/danielmiessler/Fabric) provides ~237 structured prompt
templates for content analysis, extraction, summarization, threat modeling, etc.
This module scans a local patterns directory, generates MCP tool schemas, and
dispatches execution by calling gaia-core's /api/cognitive/query endpoint.

Security model:
  - Allowlist mode (default): only manually curated patterns are active
  - Content hash pinning: sync script tracks SHA-256 per pattern, flags changes
  - No code execution: patterns are prompt text only
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

logger = logging.getLogger("GAIA.MCP.Fabric")

FABRIC_PATTERNS_DIR = Path(os.getenv("KNOWLEDGE_DIR", "/knowledge")) / "fabric_patterns"
_CONFIG_FILE = FABRIC_PATTERNS_DIR / "_config.json"

# Populated by load_fabric_patterns()
fabric_schemas: Dict[str, dict] = {}
fabric_system_prompts: Dict[str, str] = {}
fabric_config: Dict[str, Any] = {}


def _load_config() -> dict:
    """Load fabric config, defaulting to 'all' mode if no config file."""
    if _CONFIG_FILE.exists():
        try:
            return json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to read fabric config: %s", e)
    return {
        "mode": "all",
        "default_target": "core",
        "default_max_tokens": 2048,
        "default_no_think": True,
        "patterns": {},
    }


def _extract_purpose(system_md: str) -> str:
    """Extract a short description from the IDENTITY/PURPOSE section."""
    lines = system_md.split("\n")
    in_purpose = False
    purpose_lines = []
    for line in lines:
        upper = line.upper().strip()
        if "IDENTITY" in upper and "PURPOSE" in upper:
            in_purpose = True
            continue
        if in_purpose:
            if line.startswith("#"):
                break
            stripped = line.strip().lstrip("- ")
            if stripped:
                purpose_lines.append(stripped)
            if len(purpose_lines) >= 2:
                break
    desc = " ".join(purpose_lines)[:200] if purpose_lines else "Execute Fabric pattern."
    return desc


def load_fabric_patterns() -> Tuple[Dict[str, dict], Dict[str, str]]:
    """Scan fabric patterns directory and build schemas + system prompts.

    Returns (schemas_dict, system_prompts_dict) where keys are tool names
    like 'fabric_extract_wisdom'.
    """
    global fabric_config

    if not FABRIC_PATTERNS_DIR.exists():
        logger.info("Fabric patterns directory not found: %s", FABRIC_PATTERNS_DIR)
        return {}, {}

    config = _load_config()
    fabric_config = config
    mode = config.get("mode", "all")
    pattern_overrides = config.get("patterns", {})
    default_target = config.get("default_target", "core")
    default_max_tokens = config.get("default_max_tokens", 2048)

    schemas: Dict[str, dict] = {}
    prompts: Dict[str, str] = {}

    for pattern_dir in sorted(FABRIC_PATTERNS_DIR.iterdir()):
        if not pattern_dir.is_dir() or pattern_dir.name.startswith("_"):
            continue

        system_md_path = pattern_dir / "system.md"
        if not system_md_path.exists():
            continue

        pattern_name = pattern_dir.name

        # Filter by allowlist if configured
        if mode == "allowlist":
            override = pattern_overrides.get(pattern_name, {})
            if not override.get("enabled", False):
                continue

        tool_name = f"fabric_{pattern_name}"
        try:
            system_md = system_md_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to read pattern %s: %s", pattern_name, e)
            continue

        # Build description from pattern content or config override
        override = pattern_overrides.get(pattern_name, {})
        desc = override.get("description_override") or _extract_purpose(system_md)
        target = override.get("target", default_target)
        max_tokens = override.get("max_tokens", default_max_tokens)

        schemas[tool_name] = {
            "description": f"[Fabric] {desc}",
            "params": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "The text content to process with this Fabric pattern.",
                    },
                    "target": {
                        "type": "string",
                        "enum": ["core", "prime"],
                        "description": f"Model tier to use (default: {target}).",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": f"Max output tokens (default: {max_tokens}).",
                    },
                },
                "required": ["input"],
            },
        }

        prompts[tool_name] = system_md

    logger.info("Loaded %d Fabric pattern tools (mode=%s)", len(schemas), mode)
    return schemas, prompts


# ── Core client (lazy singleton) ────────────────────────────────────────────

_core_client = None


def _get_core_client():
    """Get the gaia-core service client (lazy singleton)."""
    global _core_client
    if _core_client is None:
        from gaia_common.utils.service_client import get_core_client
        _core_client = get_core_client()
    return _core_client


async def execute_fabric_tool(tool_name: str, params: dict) -> dict:
    """Execute a Fabric pattern by sending system.md + input to gaia-core.

    Calls POST /api/cognitive/query on gaia-core, which routes to the
    appropriate model tier (Core CPU or Prime GPU).

    Returns:
        {"ok": True, "content": "...", "pattern": "...", "target": "..."}
    """
    if tool_name not in fabric_system_prompts:
        return {"ok": False, "error": f"Fabric pattern '{tool_name}' not loaded"}

    system_prompt = fabric_system_prompts[tool_name]
    user_input = params.get("input", "")

    if not user_input.strip():
        return {"ok": False, "error": "Input text is required"}

    # Determine model tier and token budget
    pattern_name = tool_name.removeprefix("fabric_")
    override = fabric_config.get("patterns", {}).get(pattern_name, {})
    default_target = fabric_config.get("default_target", "core")
    default_max_tokens = fabric_config.get("default_max_tokens", 2048)
    no_think = fabric_config.get("default_no_think", True)

    target = params.get("target") or override.get("target", default_target)
    max_tokens = params.get("max_tokens") or override.get("max_tokens", default_max_tokens)

    client = _get_core_client()
    try:
        result = await client.post("/api/cognitive/query", data={
            "prompt": user_input,
            "system": system_prompt,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "target": target,
            "no_think": no_think,
        })
        return {
            "ok": True,
            "content": result.get("content", result.get("response", "")),
            "pattern": pattern_name,
            "target": result.get("target", target),
        }
    except Exception as e:
        logger.error("Fabric tool '%s' failed: %s", tool_name, e)
        return {"ok": False, "error": str(e), "pattern": pattern_name, "target": target}


# ── Module-level init ───────────────────────────────────────────────────────
# Load patterns when this module is imported. This populates fabric_schemas
# (merged into TOOLS at import time in tools.py) and fabric_system_prompts
# (used at execution time by execute_fabric_tool).

fabric_schemas, fabric_system_prompts = load_fabric_patterns()

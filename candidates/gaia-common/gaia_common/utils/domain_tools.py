"""
Domain Tools Registry — consolidated tool definitions for GAIA.

Collapses 70 individual MCP tools into 9 domain tools, each with an
`action` parameter. Reduces model prompt token usage by ~70% while
preserving all capabilities and backward compatibility.

Usage:
    from gaia_common.utils.domain_tools import (
        DOMAIN_TOOLS, ACTION_TO_LEGACY, LEGACY_TO_DOMAIN, SENSITIVE_ACTIONS,
    )

    # Model selects: {"selected_tool": "file", "params": {"action": "read", "path": "/foo"}}
    # Dispatcher pops action, maps to legacy name, delegates to existing impl:
    legacy_name = ACTION_TO_LEGACY[("file", "read")]  # → "read_file"
"""

from typing import Dict, List, Set, Tuple

# ═══════════════════════════════════════════════════════════════════════════
# Domain Tool Definitions
# ═══════════════════════════════════════════════════════════════════════════
#
# Each domain has:
#   - description: shown in model prompt
#   - actions: dict of action_name → {params, maps_to, sensitive?}
#     - params: simplified param spec for prompt display
#     - maps_to: legacy tool name in tools_registry.py
#     - sensitive: True if action requires approval (default False)

DOMAIN_TOOLS: Dict[str, dict] = {

    # ── File Operations ────────────────────────────────────────────────
    "file": {
        "description": "File operations — read, write, list, search",
        "actions": {
            "read": {
                "params": {"path": "string"},
                "maps_to": "read_file",
            },
            "write": {
                "params": {"path": "string", "content": "string"},
                "maps_to": "write_file",
                "sensitive": True,
            },
            "list": {
                "params": {"path": "string"},
                "maps_to": "list_dir",
            },
            "tree": {
                "params": {"path": "string?", "max_depth": "integer?", "max_entries": "integer?"},
                "maps_to": "list_tree",
            },
            "find": {
                "params": {"query": "string", "root": "string?", "max_depth": "integer?", "max_results": "integer?"},
                "maps_to": "find_files",
            },
        },
    },

    # ── Shell ──────────────────────────────────────────────────────────
    "shell": {
        "description": "Execute shell commands — the do-anything fallback (approval required)",
        "actions": {
            "run": {
                "params": {"command": "string"},
                "maps_to": "run_shell",
                "sensitive": True,
            },
        },
    },

    # ── Web ────────────────────────────────────────────────────────────
    "web": {
        "description": "Web search and page fetch",
        "actions": {
            "search": {
                "params": {"query": "string", "content_type": "string?", "domain_filter": "string?", "max_results": "integer?"},
                "maps_to": "web_search",
            },
            "fetch": {
                "params": {"url": "string"},
                "maps_to": "web_fetch",
            },
        },
    },

    # ── Knowledge ──────────────────────────────────────────────────────
    "knowledge": {
        "description": "Knowledge base operations — query, add, index, manage",
        "actions": {
            "query": {
                "params": {"query": "string", "knowledge_base_name": "string?", "top_k": "integer?"},
                "maps_to": "query_knowledge",
            },
            "search": {
                "params": {"query": "string", "knowledge_base_name": "string"},
                "maps_to": "find_relevant_documents",
            },
            "memory": {
                "params": {"query": "string", "top_k": "integer?"},
                "maps_to": "memory_query",
            },
            "add": {
                "params": {"knowledge_base_name": "string", "file_path": "string"},
                "maps_to": "add_document",
            },
            "index": {
                "params": {"file_path": "string", "knowledge_base_name": "string?"},
                "maps_to": "index_document",
            },
            "embed": {
                "params": {"knowledge_base_name": "string"},
                "maps_to": "embed_documents",
            },
            "list": {
                "params": {},
                "maps_to": "list_knowledge_bases",
            },
            "status": {
                "params": {},
                "maps_to": "memory_status",
            },
            "rebuild": {
                "params": {"doc_dir": "string?"},
                "maps_to": "memory_rebuild_index",
                "sensitive": True,
            },
        },
    },

    # ── Audio ──────────────────────────────────────────────────────────
    "audio": {
        "description": "Audio capture, transcription, and inbox management",
        "actions": {
            "listen_start": {
                "params": {"mode": "string?", "comment_threshold": "string?"},
                "maps_to": "audio_listen_start",
                "sensitive": True,
            },
            "listen_stop": {
                "params": {},
                "maps_to": "audio_listen_stop",
            },
            "listen_status": {
                "params": {},
                "maps_to": "audio_listen_status",
            },
            "inbox_status": {
                "params": {},
                "maps_to": "audio_inbox_status",
            },
            "inbox_list": {
                "params": {},
                "maps_to": "audio_inbox_list",
            },
            "inbox_process": {
                "params": {},
                "maps_to": "audio_inbox_process",
            },
            "inbox_review": {
                "params": {"filename": "string"},
                "maps_to": "audio_inbox_review",
            },
        },
    },

    # ── Study & Adapters ───────────────────────────────────────────────
    "study": {
        "description": "LoRA adapter training and management",
        "actions": {
            "train": {
                "params": {"adapter_name": "string", "documents": "array", "tier": "integer?", "pillar": "string?", "description": "string?", "max_steps": "integer?"},
                "maps_to": "study_start",
            },
            "status": {
                "params": {},
                "maps_to": "study_status",
            },
            "cancel": {
                "params": {},
                "maps_to": "study_cancel",
            },
            "adapter_list": {
                "params": {"tier": "integer?"},
                "maps_to": "adapter_list",
            },
            "adapter_load": {
                "params": {"adapter_name": "string", "tier": "integer"},
                "maps_to": "adapter_load",
            },
            "adapter_unload": {
                "params": {"adapter_name": "string"},
                "maps_to": "adapter_unload",
            },
            "adapter_info": {
                "params": {"adapter_name": "string", "tier": "integer"},
                "maps_to": "adapter_info",
            },
            "adapter_delete": {
                "params": {"adapter_name": "string", "tier": "integer"},
                "maps_to": "adapter_delete",
            },
        },
    },

    # ── Introspect ─────────────────────────────────────────────────────
    "introspect": {
        "description": "Self-inspection — world state, logs, events, tool discovery",
        "actions": {
            "world": {
                "params": {},
                "maps_to": "world_state",
            },
            "recall": {
                "params": {"hours": "number?", "limit": "integer?", "cfr": "boolean?"},
                "maps_to": "recall_events",
            },
            "logs": {
                "params": {"service": "string", "lines": "integer?", "search": "string?", "level": "string?"},
                "maps_to": "introspect_logs",
            },
            "count_chars": {
                "params": {"text": "string", "char": "string"},
                "maps_to": "count_chars",
            },
            "tools": {
                "params": {},
                "maps_to": "list_tools",
            },
            "describe": {
                "params": {"tool_name": "string"},
                "maps_to": "describe_tool",
            },
        },
    },

    # ── Worldbuild (Kanka.io) ─────────────────────────────────────────
    "worldbuild": {
        "description": "Kanka.io campaign and worldbuilding management",
        "actions": {
            "campaigns": {"params": {}, "maps_to": "kanka_list_campaigns"},
            "search": {"params": {"query": "string", "campaign": "string?"}, "maps_to": "kanka_search"},
            "list": {"params": {"entity_type": "string", "campaign": "string?"}, "maps_to": "kanka_list_entities"},
            "get": {"params": {"entity_type": "string", "entity_id": "integer", "campaign": "string?"}, "maps_to": "kanka_get_entity"},
            "create": {"params": {"entity_type": "string", "name": "string", "entry": "string?", "campaign": "string?"}, "maps_to": "kanka_create_entity", "sensitive": True},
            "update": {"params": {"entity_type": "string", "entity_id": "integer", "fields": "object", "campaign": "string?"}, "maps_to": "kanka_update_entity", "sensitive": True},
        },
    },

    # ── Notebook (NotebookLM) ──────────────────────────────────────────
    "notebook": {
        "description": "Google NotebookLM — notebooks, sources, notes, AI chat",
        "actions": {
            "list": {"params": {}, "maps_to": "notebooklm_list_notebooks"},
            "get": {"params": {"notebook_id": "string"}, "maps_to": "notebooklm_get_notebook"},
            "sources": {"params": {"notebook_id": "string"}, "maps_to": "notebooklm_list_sources"},
            "notes": {"params": {"notebook_id": "string"}, "maps_to": "notebooklm_list_notes"},
            "artifacts": {"params": {"notebook_id": "string", "artifact_type": "string?"}, "maps_to": "notebooklm_list_artifacts"},
            "chat": {"params": {"notebook_id": "string", "question": "string"}, "maps_to": "notebooklm_chat"},
            "create_note": {"params": {"notebook_id": "string", "title": "string", "content": "string?"}, "maps_to": "notebooklm_create_note", "sensitive": True},
            "download_audio": {"params": {"notebook_id": "string", "artifact_id": "string?"}, "maps_to": "notebooklm_download_audio"},
        },
    },

    # ── Context (CFR + Fragments) ──────────────────────────────────────
    "context": {
        "description": "Document context management — CFR resolution tree, response fragments",
        "actions": {
            # CFR
            "ingest": {"params": {"file_path": "string", "doc_id": "string?"}, "maps_to": "cfr_ingest"},
            "focus": {"params": {"doc_id": "string", "section_index": "integer"}, "maps_to": "cfr_focus"},
            "compress": {"params": {"doc_id": "string", "section_index": "integer"}, "maps_to": "cfr_compress"},
            "expand": {"params": {"doc_id": "string", "section_index": "integer"}, "maps_to": "cfr_expand"},
            "synthesize": {"params": {"doc_id": "string"}, "maps_to": "cfr_synthesize"},
            "status": {"params": {"doc_id": "string?"}, "maps_to": "cfr_status"},
            "rolling": {"params": {"doc_id": "string", "target_section": "integer"}, "maps_to": "cfr_rolling_context"},
            # Fragments
            "fragment_write": {"params": {"parent_request_id": "string", "content": "string", "sequence": "integer?", "is_complete": "boolean?"}, "maps_to": "fragment_write"},
            "fragment_read": {"params": {"parent_request_id": "string"}, "maps_to": "fragment_read"},
            "fragment_assemble": {"params": {"parent_request_id": "string"}, "maps_to": "fragment_assemble"},
            "fragment_list": {"params": {}, "maps_to": "fragment_list_pending"},
            "fragment_clear": {"params": {"parent_request_id": "string?"}, "maps_to": "fragment_clear"},
        },
    },

    # ── Manage (Promotion + Blueprints) ────────────────────────────────
    "manage": {
        "description": "Service promotion, blueprints, and deployment management",
        "actions": {
            "blueprint": {"params": {"service_id": "string"}, "maps_to": "generate_blueprint"},
            "assess": {"params": {"service_id": "string"}, "maps_to": "assess_promotion"},
            "promote": {"params": {"service_id": "string", "verdict": "string", "recommendation": "string", "pipeline_cmd": "string", "check_summary": "string"}, "maps_to": "promotion_create_request", "sensitive": True},
            "promote_list": {"params": {"service_id": "string?", "status_filter": "string?"}, "maps_to": "promotion_list_requests"},
            "promote_status": {"params": {"request_id": "string"}, "maps_to": "promotion_request_status"},
        },
    },

    # ── Fabric Patterns ────────────────────────────────────────────────
    "fabric": {
        "description": "Run a Fabric analysis pattern (summarize, extract, analyze, etc.)",
        "actions": {},  # Dynamic — patterns discovered at runtime
        "dynamic": True,
        "params_override": {
            "pattern": {"type": "string", "description": "Fabric pattern name (e.g., summarize, extract_wisdom, improve_writing)"},
            "input": {"type": "string", "description": "Text to process through the pattern"},
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Derived Mappings (auto-generated from DOMAIN_TOOLS)
# ═══════════════════════════════════════════════════════════════════════════

def _build_mappings():
    """Build forward and reverse lookup tables from DOMAIN_TOOLS."""
    action_to_legacy: Dict[Tuple[str, str], str] = {}
    legacy_to_domain: Dict[str, Tuple[str, str]] = {}
    sensitive: Set[Tuple[str, str]] = set()
    all_actions: Dict[str, List[str]] = {}

    for domain, spec in DOMAIN_TOOLS.items():
        if spec.get("dynamic"):
            continue  # Fabric patterns handled separately
        domain_actions = []
        for action_name, action_spec in spec.get("actions", {}).items():
            legacy_name = action_spec["maps_to"]
            key = (domain, action_name)
            action_to_legacy[key] = legacy_name
            legacy_to_domain[legacy_name] = key
            if action_spec.get("sensitive"):
                sensitive.add(key)
            domain_actions.append(action_name)
        all_actions[domain] = domain_actions

    # Legacy aliases — tools that map to the same domain action
    legacy_to_domain["ai_write"] = ("file", "write")

    return action_to_legacy, legacy_to_domain, sensitive, all_actions


ACTION_TO_LEGACY, LEGACY_TO_DOMAIN, SENSITIVE_ACTIONS, DOMAIN_ACTIONS = _build_mappings()


# ═══════════════════════════════════════════════════════════════════════════
# Prompt Catalog — compact tool descriptions for model context
# ═══════════════════════════════════════════════════════════════════════════

def build_prompt_catalog() -> str:
    """
    Build a compact tool catalog for the model prompt.

    Returns ~150 tokens of tool descriptions (vs ~600 for the old allowlist).
    """
    lines = ["Available tools (use action parameter to specify operation):"]
    for domain, spec in DOMAIN_TOOLS.items():
        desc = spec["description"]
        if spec.get("dynamic"):
            # Fabric: special format
            lines.append(f"- fabric(pattern, input): {desc}")
            continue
        actions = DOMAIN_ACTIONS.get(domain, [])
        # Group actions concisely
        action_str = "|".join(actions[:8])
        if len(actions) > 8:
            action_str += f"|... ({len(actions)} total)"
        sensitive_mark = ""
        domain_sensitive = [a for a in actions if (domain, a) in SENSITIVE_ACTIONS]
        if domain_sensitive:
            sensitive_mark = f" [approval: {','.join(domain_sensitive)}]"
        lines.append(f"- {domain}(action): {action_str} — {desc}{sensitive_mark}")
    lines.append("")
    lines.append("Format: {\"selected_tool\": \"domain\", \"params\": \"{\\\"action\\\": \\\"verb\\\", ...}\"}")
    lines.append("For unlisted tools, use introspect(action=tools) to discover all available actions.")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_domain_call(domain: str, action: str) -> str:
    """
    Validate a domain tool call and return the legacy tool name.

    Raises ValueError if domain or action is invalid.
    """
    if domain not in DOMAIN_TOOLS:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(DOMAIN_TOOLS.keys())}")

    if domain == "fabric":
        # Fabric patterns are dynamic — action is the pattern name
        return f"fabric_{action}"

    key = (domain, action)
    if key not in ACTION_TO_LEGACY:
        available = DOMAIN_ACTIONS.get(domain, [])
        raise ValueError(f"Unknown action '{action}' for domain '{domain}'. Available: {available}")

    return ACTION_TO_LEGACY[key]


def is_sensitive(domain: str, action: str) -> bool:
    """Check if a domain+action requires approval."""
    return (domain, action) in SENSITIVE_ACTIONS

# app/utils/tools_registry.py
"""
Defines the schemas for all tools exposed by the MCP-lite server.
This acts as a central registry for tool discovery and validation.
"""

from typing import Any, Dict, List, Optional, Tuple

TOOLS = {
    "run_shell": {
        "description": "Executes a whitelisted shell command in a sandboxed environment.",
        "params": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute."}
            },
            "required": ["command"]
        }
    },
    "read_file": {
        "description": "Reads the entire content of a specified file.",
        "params": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The absolute path to the file."}
            },
            "required": ["path"]
        }
    },
    "write_file": {
        "description": "Writes content to a specified file.",
        "params": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The absolute path to the file."},
                "content": {"type": "string", "description": "The content to write."}
            },
            "required": ["path", "content"]
        }
    },
    "ai_write": {
        "description": "AI-initiated write helper used by approval flows (path absolute, content string).",
        "params": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The absolute path to write to."},
                "content": {"type": "string", "description": "The content to write."},
                "base_cwd": {"type": "string", "description": "Optional base cwd for relative paths."}
            },
            "required": ["path", "content"]
        }
    },
    "list_dir": {
        "description": "Lists the contents of a specified directory.",
        "params": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The absolute path to the directory."}
            },
            "required": ["path"]
        }
    },
    "list_tree": {
        "description": "Returns a bounded directory tree (safe depth/entry limits).",
        "params": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Optional root path (defaults to /gaia-assistant)."},
                "max_depth": {"type": "integer", "description": "Maximum depth to traverse (default 3, max 6)."},
                "max_entries": {"type": "integer", "description": "Maximum entries to include (default 200, max 1000)."}
            },
            "required": []
        }
    },
    "list_tools": {
        "description": "Lists all available tools on the server.",
        "params": {}
    },
    "world_state": {
        "description": "Returns an expanded dynamic world-state snapshot (telemetry + models + MCP tool list).",
        "params": {}
    },
    "describe_tool": {
        "description": "Returns the JSON schema for a specified tool.",
        "params": {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string", "description": "The name of the tool to describe."}
            },
            "required": ["tool_name"]
        }
    },
    "memory_status": {
        "description": "Summarize memory/index state (counts, paths, last build).",
        "params": {}
    },
    "memory_query": {
        "description": "Run a semantic memory lookup against the vector index.",
        "params": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language query."},
                "top_k": {"type": "integer", "description": "Number of results to return (default 5)."}
            },
            "required": ["query"]
        }
    },
    "memory_rebuild_index": {
        "description": "Rebuild the semantic memory index from core documents (requires approval).",
        "params": {
            "type": "object",
            "properties": {
                "doc_dir": {"type": "string", "description": "Optional doc directory to index (defaults to GAIA core docs)."}
            },
            "required": []
        }
    },
    "find_files": {
        "description": "Search for files whose names contain a query (case-insensitive, bounded depth).",
        "params": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Substring to search for in filenames (e.g., 'dev_matrix')."},
                "root": {"type": "string", "description": "Optional root path (default /gaia-assistant)."},
                "max_depth": {"type": "integer", "description": "Maximum depth to traverse (default 5, max 8)."},
                "max_results": {"type": "integer", "description": "Maximum number of results to return (default 50, max 200)."}
            },
            "required": ["query"]
        }
    },
    "find_relevant_documents": {
        "description": "Finds documents relevant to a query within a specified knowledge base.",
        "params": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The query to find relevant documents for."},
                "knowledge_base_name": {"type": "string", "description": "The name of the knowledge base to search in."}
            },
            "required": ["query", "knowledge_base_name"]
        }
    },
    # --- Response Fragmentation Tools ---
    # These tools allow GAIA to handle responses that exceed token limits
    # by storing, continuing, and assembling fragmented responses.
    "fragment_write": {
        "description": "Store a response fragment for later assembly. Use when output is truncated due to token limits.",
        "params": {
            "type": "object",
            "properties": {
                "parent_request_id": {"type": "string", "description": "UUID linking all fragments from the same request."},
                "sequence": {"type": "integer", "description": "Fragment ordering (0, 1, 2, ...)."},
                "content": {"type": "string", "description": "The actual text content of this fragment."},
                "continuation_hint": {"type": "string", "description": "Context for continuation (e.g., 'The Raven stanza 10/18')."},
                "is_complete": {"type": "boolean", "description": "True if this is the final fragment."},
                "token_count": {"type": "integer", "description": "Approximate token count for this fragment."}
            },
            "required": ["parent_request_id", "content"]
        }
    },
    "fragment_read": {
        "description": "Retrieve all fragments for a given request, sorted by sequence.",
        "params": {
            "type": "object",
            "properties": {
                "parent_request_id": {"type": "string", "description": "The UUID linking fragments."}
            },
            "required": ["parent_request_id"]
        }
    },
    "fragment_assemble": {
        "description": "Assemble fragments into a complete response. Checks for seam overlaps and completeness.",
        "params": {
            "type": "object",
            "properties": {
                "parent_request_id": {"type": "string", "description": "The UUID linking fragments."},
                "seam_overlap_check": {"type": "boolean", "description": "If true, attempt to detect/remove duplicate text at seams (default true)."}
            },
            "required": ["parent_request_id"]
        }
    },
    "fragment_list_pending": {
        "description": "List all pending (incomplete) fragment requests.",
        "params": {}
    },
    "fragment_clear": {
        "description": "Clear fragments. If parent_request_id provided, clears only that request's fragments; otherwise clears all.",
        "params": {
            "type": "object",
            "properties": {
                "parent_request_id": {"type": "string", "description": "Optional: specific request to clear."}
            },
            "required": []
        }
    },
    # --- Study Mode / LoRA Adapter Tools ---
    # These tools allow GAIA to learn new knowledge through QLoRA fine-tuning
    "study_start": {
        "description": "Start a study session to learn from documents. Creates a LoRA adapter with the specified knowledge.",
        "params": {
            "type": "object",
            "properties": {
                "adapter_name": {"type": "string", "description": "Unique name for the adapter (e.g., 'jabberwocky_poem')."},
                "documents": {"type": "array", "items": {"type": "string"}, "description": "List of file paths to learn from."},
                "tier": {"type": "integer", "description": "Adapter tier: 1=global (permanent), 2=user (persistent), 3=session (temporary). Default 3."},
                "pillar": {"type": "string", "description": "GAIA pillar: identity, memory, cognition, embodiment, or general. Default 'general'."},
                "description": {"type": "string", "description": "Human-readable description of what this adapter teaches."},
                "activation_triggers": {"type": "array", "items": {"type": "string"}, "description": "Keywords that should trigger loading this adapter."},
                "max_steps": {"type": "integer", "description": "Maximum training steps (default 100)."}
            },
            "required": ["adapter_name", "documents"]
        }
    },
    "study_status": {
        "description": "Get the current status of study mode (training progress, state, etc.).",
        "params": {}
    },
    "study_cancel": {
        "description": "Cancel an in-progress training session.",
        "params": {}
    },
    "adapter_list": {
        "description": "List all available LoRA adapters, optionally filtered by tier.",
        "params": {
            "type": "object",
            "properties": {
                "tier": {"type": "integer", "description": "Optional: filter by tier (1, 2, or 3)."}
            },
            "required": []
        }
    },
    "adapter_load": {
        "description": "Load a LoRA adapter for use in generation.",
        "params": {
            "type": "object",
            "properties": {
                "adapter_name": {"type": "string", "description": "Name of the adapter to load."},
                "tier": {"type": "integer", "description": "Tier where the adapter is stored."}
            },
            "required": ["adapter_name", "tier"]
        }
    },
    "adapter_unload": {
        "description": "Unload a currently loaded LoRA adapter.",
        "params": {
            "type": "object",
            "properties": {
                "adapter_name": {"type": "string", "description": "Name of the adapter to unload."}
            },
            "required": ["adapter_name"]
        }
    },
    "adapter_delete": {
        "description": "Delete a LoRA adapter (tier 1 adapters cannot be deleted).",
        "params": {
            "type": "object",
            "properties": {
                "adapter_name": {"type": "string", "description": "Name of the adapter to delete."},
                "tier": {"type": "integer", "description": "Tier where the adapter is stored."}
            },
            "required": ["adapter_name", "tier"]
        }
    },
    "adapter_info": {
        "description": "Get detailed information about a specific adapter.",
        "params": {
            "type": "object",
            "properties": {
                "adapter_name": {"type": "string", "description": "Name of the adapter."},
                "tier": {"type": "integer", "description": "Tier where the adapter is stored."}
            },
            "required": ["adapter_name", "tier"]
        }
    },
    # --- Knowledge Base Tools ---
    "embed_documents": {
        "description": "Embeds all documents in a knowledge base into the vector store.",
        "params": {
            "type": "object",
            "properties": {
                "knowledge_base_name": {"type": "string", "description": "The name of the knowledge base to embed."}
            },
            "required": ["knowledge_base_name"]
        }
    },
    "query_knowledge": {
        "description": "Run a semantic memory lookup against a knowledge base.",
        "params": {
            "type": "object",
            "properties": {
                "knowledge_base_name": {"type": "string", "description": "The name of the knowledge base to query."},
                "query": {"type": "string", "description": "Natural language query."},
                "top_k": {"type": "integer", "description": "Number of results to return (default 5)."}
            },
            "required": ["knowledge_base_name", "query"]
        }
    },
    "add_document": {
        "description": "Adds a new document to a knowledge base.",
        "params": {
            "type": "object",
            "properties": {
                "knowledge_base_name": {"type": "string", "description": "The name of the knowledge base to add the document to."},
                "file_path": {"type": "string", "description": "The path to the document to add."}
            },
            "required": ["knowledge_base_name", "file_path"]
        }
    }
}
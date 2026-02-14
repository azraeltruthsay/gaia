"""
Tool Selector Module

Responsible for:
1. Determining if a request needs MCP tool usage
2. Selecting the appropriate tool with low-temperature generation
3. Extracting structured parameters
4. Providing confidence scores for review

This module is part of the GCP Tool Routing System.
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple

from gaia_common.protocols.cognition_packet import (
    CognitionPacket, SelectedTool, ToolRoutingState, ToolExecutionStatus,
    ToolExecutionResult, ReflectionLog, DataField, Sketchpad
)

logger = logging.getLogger("GAIA.ToolSelector")

# ── Guided decoding JSON schemas ─────────────────────────────────────────
# These are passed to vLLM's guided_json parameter to guarantee structurally
# valid JSON output from the model.  The schemas are intentionally kept flat
# (no $ref, no oneOf) because xgrammar works best with simple schemas.

_TOOL_REVIEW_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["approved", "confidence", "reasoning"],
}

_TOOL_SELECT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "selected_tool": {"type": ["string", "null"]},
        "params": {"type": "object"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["selected_tool", "reasoning", "confidence"],
}


def _structured_json_kwargs(model: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Return the kwargs needed to enable structured JSON output for *model*.

    Supports two backends:
      - VLLMRemoteModel → ``guided_json=<schema>``
      - llama_cpp Llama → ``response_format={"type": "json_object", "schema": <schema>}``

    Returns an empty dict if the model type is unrecognised (graceful no-op).
    """
    model_type = type(model).__name__
    if model_type == "VLLMRemoteModel":
        return {"guided_json": schema}
    if model_type == "Llama":
        return {"response_format": {"type": "json_object", "schema": schema}}
    return {}


# ── Tool catalog ─────────────────────────────────────────────────────────
# Auto-generated from gaia_common.utils.tools_registry.TOOLS (the single
# source of truth for MCP tool schemas).  Internal tools that are handled
# directly by _execute_mcp_tool (not dispatched via JSON-RPC) are added
# separately below.
#
# The flat format expected by _build_tool_catalog():
#   { "tool_name": { "description": str, "params": [str, ...],
#                     "param_descriptions": {str: str}, "requires_approval": bool } }

from gaia_common.utils.tools_registry import TOOLS as _REGISTRY_TOOLS

# Tools requiring approval (mirrors gaia-mcp SENSITIVE_TOOLS)
_SENSITIVE_TOOLS = {"ai_write", "write_file", "run_shell", "memory_rebuild_index"}


def _registry_to_catalog(registry: Dict[str, Any]) -> Dict[str, Any]:
    """Convert tools_registry.TOOLS schema format into the flat format
    used by the tool selector prompt."""
    catalog: Dict[str, Any] = {}
    for name, schema in registry.items():
        params_schema = schema.get("params", {})
        props = params_schema.get("properties", {}) if isinstance(params_schema, dict) else {}
        catalog[name] = {
            "description": schema.get("description", ""),
            "params": list(props.keys()),
            "param_descriptions": {
                k: v.get("description", "") for k, v in props.items()
            },
            "requires_approval": name in _SENSITIVE_TOOLS,
        }
    return catalog


# Internal tools not in tools_registry (handled directly by _execute_mcp_tool)
_INTERNAL_TOOLS: Dict[str, Any] = {
    "ai.read": {
        "description": "Read a file from the filesystem",
        "params": ["path"],
        "param_descriptions": {"path": "Absolute path to the file to read"},
        "requires_approval": False,
    },
    "ai.write": {
        "description": "Write content to a file on the filesystem",
        "params": ["path", "content"],
        "param_descriptions": {
            "path": "Absolute path to the file to write",
            "content": "Content to write to the file",
        },
        "requires_approval": True,
    },
    "ai.execute": {
        "description": "Execute a shell command",
        "params": ["command"],
        "param_descriptions": {"command": "Shell command to execute"},
        "requires_approval": True,
    },
    "embedding.query": {
        "description": "Query the vector database for semantic search",
        "params": ["query", "top_k"],
        "param_descriptions": {
            "query": "Search query text",
            "top_k": "Number of results to return (default: 5)",
        },
        "requires_approval": False,
    },
}

# Merged catalog: registry tools + internal tools (internal wins on collision)
AVAILABLE_TOOLS: Dict[str, Any] = {**_registry_to_catalog(_REGISTRY_TOOLS), **_INTERNAL_TOOLS}

# Tools to show in the LLM selection prompt.  The full AVAILABLE_TOOLS
# catalog has 30+ entries — dumping all of them overwhelms a 3B model.
# This allowlist keeps the prompt focused on tools a user request would
# actually need.  Everything else still works via the JSON-RPC fallback
# if selected by name.
_PROMPT_TOOLS = {
    # File operations
    "read_file", "write_file", "ai.read", "ai.write",
    # Shell
    "run_shell", "ai.execute",
    # Search & knowledge
    "web_search", "web_fetch", "embedding.query",
    "memory_query", "find_files", "find_relevant_documents",
    "query_knowledge", "add_document",
    # Directory listing
    "list_dir", "list_tree",
}


def needs_tool_routing(packet: CognitionPacket, user_input: str) -> bool:
    """
    Determine if the request likely needs MCP tool usage.

    This is a fast heuristic check before invoking the LLM for selection.
    Returns True if we should route to tool selection.
    """
    lowered = user_input.lower()

    # File operation indicators
    file_indicators = [
        "read ", "open ", "show ", "view ", "cat ",
        "write ", "save ", "create file",
        ".md", ".txt", ".json", ".py", ".yaml", ".yml",
        "/gaia", "/knowledge", "/app", "/docs"
    ]

    # Command execution indicators
    exec_indicators = [
        "run ", "execute ", "shell ", "command ",
        "ls ", "pwd", "git ", "docker "
    ]

    # Search/query indicators
    search_indicators = [
        "search ", "find ", "look for ", "where is ",
        "semantic search", "query "
    ]

    for indicator in file_indicators + exec_indicators + search_indicators:
        if indicator in lowered:
            logger.debug(f"Tool routing triggered by indicator: '{indicator}'")
            return True

    # Check if available_mcp_tools in context suggests tool availability
    if packet.context.available_mcp_tools:
        logger.debug("Tool routing triggered by available_mcp_tools in context")
        return True

    return False


def select_tool(
    packet: CognitionPacket,
    user_input: str,
    model,
    temperature: float = 0.15
) -> Tuple[Optional[SelectedTool], List[SelectedTool]]:
    """
    Use low-temperature LLM generation to select the appropriate tool.

    Args:
        packet: The cognition packet with context
        user_input: The user's request
        model: LLM model to use for selection
        temperature: Generation temperature (default 0.15 for determinism)

    Returns:
        Tuple of (primary_selection, alternative_selections)
    """
    # Build tool catalog for prompt
    tool_catalog = _build_tool_catalog()

    # Build selection prompt
    prompt = f"""You are a tool selector for GAIA. Given the user's request, select the most appropriate tool.

AVAILABLE TOOLS:
{tool_catalog}

USER REQUEST: {user_input}

CONTEXT FROM PACKET:
- Session: {packet.header.session_id}
- Intent: {packet.intent.user_intent}
- Available tools: {packet.context.available_mcp_tools or 'all'}

Respond ONLY with valid JSON in this exact format:
{{
    "selected_tool": "tool_name",
    "params": {{"param1": "value1"}},
    "reasoning": "Brief explanation of why this tool was selected",
    "confidence": 0.0 to 1.0,
    "alternatives": [
        {{"tool": "alt_tool", "params": {{}}, "reason": "why this could also work"}}
    ]
}}

If NO tool is appropriate, respond with:
{{
    "selected_tool": null,
    "reasoning": "Why no tool is needed",
    "confidence": 1.0
}}
"""

    messages = [
        {"role": "system", "content": "You are a precise tool selector. Output only valid JSON."},
        {"role": "user", "content": prompt}
    ]

    try:
        # Use low temperature for deterministic selection
        extra_kwargs = _structured_json_kwargs(model, _TOOL_SELECT_SCHEMA)
        if extra_kwargs:
            logger.debug("Tool selector: using structured JSON decoding (%s)", type(model).__name__)

        result = model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=500,
            top_p=0.9,
            stream=False,
            **extra_kwargs,
        )

        # Extract response content
        content = _extract_content(result)
        logger.debug(f"Tool selector raw response: {content[:200]}...")

        # Try to extract JSON from the response (handle markdown code blocks)
        json_content = _extract_json_from_response(content)

        # Parse JSON response
        selection = json.loads(json_content)

        if selection.get("selected_tool") is None:
            logger.info(f"Tool selector determined no tool needed: {selection.get('reasoning')}")
            return None, []

        primary = SelectedTool(
            tool_name=selection["selected_tool"],
            params=selection.get("params", {}),
            selection_reasoning=selection.get("reasoning", ""),
            selection_confidence=selection.get("confidence", 0.5)
        )

        alternatives = []
        for alt in selection.get("alternatives", []):
            alternatives.append(SelectedTool(
                tool_name=alt.get("tool", ""),
                params=alt.get("params", {}),
                selection_reasoning=alt.get("reason", ""),
                selection_confidence=0.0  # Alternatives don't have confidence scores
            ))

        logger.info(f"Tool selected: {primary.tool_name} (confidence={primary.selection_confidence})")
        return primary, alternatives

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse tool selection JSON: {e}")
        return None, []
    except Exception as e:
        logger.error(f"Tool selection failed: {e}")
        return None, []


def review_selection(
    packet: CognitionPacket,
    selected_tool: SelectedTool,
    model,
    temperature: float = 0.3
) -> Tuple[float, str]:
    """
    Have the Prime model review the tool selection for confidence.

    Args:
        packet: The cognition packet with context
        selected_tool: The tool that was selected
        model: LLM model to use for review
        temperature: Generation temperature

    Returns:
        Tuple of (confidence_score, reasoning)
    """
    review_prompt = f"""Review this tool selection for the given request.

USER REQUEST: {packet.content.original_prompt}

SELECTED TOOL: {selected_tool.tool_name}
PARAMETERS: {json.dumps(selected_tool.params)}
SELECTION REASONING: {selected_tool.selection_reasoning}

Evaluate:
1. Is this the right tool for the task?
2. Are the parameters correct and safe?
3. Could this cause unintended side effects?

Respond with JSON:
{{
    "approved": true/false,
    "confidence": 0.0 to 1.0,
    "reasoning": "Your assessment"
}}
"""

    messages = [
        {"role": "system", "content": "You are a careful reviewer ensuring tool selections are appropriate."},
        {"role": "user", "content": review_prompt}
    ]

    try:
        extra_kwargs = _structured_json_kwargs(model, _TOOL_REVIEW_SCHEMA)
        if extra_kwargs:
            logger.debug("Tool review: using structured JSON decoding (%s)", type(model).__name__)

        result = model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=200,
            stream=False,
            **extra_kwargs,
        )

        content = _extract_content(result)
        json_content = _extract_json_from_response(content)
        review = json.loads(json_content)

        confidence = review.get("confidence", 0.0)
        reasoning = review.get("reasoning", "")

        if not review.get("approved", False):
            confidence = min(confidence, 0.5)  # Cap confidence if not approved

        logger.info(f"Tool review: approved={review.get('approved')}, confidence={confidence}")
        return confidence, reasoning

    except Exception as e:
        # Review model failed (often JSON parse errors from small models).
        # Fall back to the original selection confidence instead of returning 0.0,
        # which would block ALL tool usage when the review model can't produce JSON.
        fallback = selected_tool.selection_confidence
        logger.warning(f"Review failed ({e}); using selection confidence {fallback:.2f} as fallback")
        return fallback, f"Review failed: {e}; using selection confidence"


def _build_tool_catalog() -> str:
    """Build a formatted catalog of available tools for the selection prompt.

    Only includes tools in ``_PROMPT_TOOLS`` to keep the prompt compact
    enough for small models.
    """
    lines = []
    for tool_name, info in AVAILABLE_TOOLS.items():
        if tool_name not in _PROMPT_TOOLS:
            continue
        lines.append(f"- {tool_name}: {info['description']}")
        if info['params']:
            lines.append(f"  Parameters: {', '.join(info['params'])}")
        if info.get("requires_approval"):
            lines.append(f"  Note: Requires user approval")
    return "\n".join(lines)


def _extract_content(result) -> str:
    """Extract content from various LLM response formats."""
    if isinstance(result, dict) and "choices" in result:
        choice = result["choices"][0]
        if isinstance(choice, dict):
            if "message" in choice:
                return choice["message"].get("content", "")
            return choice.get("text", "")
    return str(result)


def _extract_json_from_response(content: str) -> str:
    """
    Extract JSON from a response that might be wrapped in markdown code blocks.

    Handles:
    - ```json ... ```
    - ``` ... ```
    - Raw JSON
    """
    # Try to find JSON in code blocks first
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    matches = re.findall(code_block_pattern, content)
    if matches:
        return matches[0].strip()

    # Try to find raw JSON object
    json_pattern = r'\{[\s\S]*\}'
    matches = re.findall(json_pattern, content)
    if matches:
        return matches[0]

    # Return as-is and let JSON parser handle it
    return content.strip()


def initialize_tool_routing(packet: CognitionPacket) -> CognitionPacket:
    """
    Initialize tool routing state on a packet if not present.

    Returns the packet with tool_routing initialized.
    """
    if packet.tool_routing is None:
        packet.tool_routing = ToolRoutingState()
    return packet


def inject_tool_result_into_packet(packet: CognitionPacket) -> CognitionPacket:
    """
    Inject tool execution result into the packet's context for final response.

    This adds the result as a DataField and to the sketchpad for easy reference.
    """
    if not packet.tool_routing or not packet.tool_routing.execution_result:
        return packet

    result = packet.tool_routing.execution_result
    tool = packet.tool_routing.selected_tool

    # Create a data field with the tool result
    result_field = DataField(
        key="tool_result",
        value={
            "tool": tool.tool_name if tool else "unknown",
            "params": tool.params if tool else {},
            "success": result.success,
            "output": result.output,
            "error": result.error
        },
        type="tool_execution_result",
        source="mcp_client"
    )

    packet.content.data_fields.append(result_field)

    # Also add to sketchpad for easy reference
    packet.reasoning.sketchpad.append(Sketchpad(
        slot="latest_tool_result",
        content=result.output if result.success else result.error,
        content_type="tool_output"
    ))

    return packet

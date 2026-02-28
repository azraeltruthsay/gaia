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
    "additionalProperties": False,
}

_TOOL_SELECT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "selected_tool": {"type": "string"},
        "params": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["selected_tool", "params", "reasoning", "confidence"],
    "additionalProperties": False,
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
    # Self-introspection
    "introspect_logs",
}

# ── Timeout-protected Llama calls ────────────────────────────────────────
_LLAMA_TIMEOUT_S = 30  # seconds before falling back to unconstrained generation


def _llama_with_timeout(model, timeout_s: int = _LLAMA_TIMEOUT_S, **kwargs):
    """Run Llama.create_chat_completion with a timeout.

    llama-cpp's grammar-constrained generation can hang indefinitely on
    complex schemas.  This wrapper runs the call in a thread and enforces
    a time limit.  On timeout it raises TimeoutError so the caller can
    retry without grammar constraints.

    NOTE: We must NOT use ``with ThreadPoolExecutor`` because __exit__
    calls ``shutdown(wait=True)`` which blocks until the hung thread
    finishes — defeating the timeout entirely.  Instead we create the
    pool, grab the future, and on timeout call ``shutdown(wait=False)``
    so the caller is unblocked immediately.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(model.create_chat_completion, **kwargs)
    try:
        result = future.result(timeout=timeout_s)
        pool.shutdown(wait=False)
        return result
    except FuturesTimeout:
        pool.shutdown(wait=False, cancel_futures=True)
        logger.warning("Llama structured JSON call timed out after %ds", timeout_s)
        raise TimeoutError(f"Llama call timed out after {timeout_s}s")


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

    # Self-introspection / diagnostic indicators
    introspection_indicators = [
        "introspect", "your logs", "my logs", "diagnos",
        "check your", "check my", "service logs",
    ]

    for indicator in file_indicators + exec_indicators + search_indicators + introspection_indicators:
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

    # Build selection prompt — schema uses string params (not object) to avoid
    # llama-cpp grammar hangs with unconstrained objects.
    prompt = f"""You are a tool selector for GAIA. Given the user's request, select the most appropriate tool.

AVAILABLE TOOLS:
{tool_catalog}

USER REQUEST: {user_input}

CONTEXT FROM PACKET:
- Session: {packet.header.session_id}
- Intent: {packet.intent.user_intent}
- Available tools: {packet.context.available_mcp_tools or 'all'}

Respond ONLY with valid JSON. Use exactly these four fields:
{{
    "selected_tool": "tool_name",
    "params": "{{\\"param1\\": \\"value1\\"}}",
    "reasoning": "Brief explanation of why this tool was selected",
    "confidence": 0.9
}}

If NO tool is appropriate, set selected_tool to empty string:
{{
    "selected_tool": "",
    "params": "{{}}",
    "reasoning": "Why no tool is needed",
    "confidence": 1.0
}}
"""

    messages = [
        {"role": "system", "content": "You are a precise tool selector. Output only valid JSON with exactly four fields: selected_tool, params, reasoning, confidence."},
        {"role": "user", "content": prompt}
    ]

    try:
        extra_kwargs = _structured_json_kwargs(model, _TOOL_SELECT_SCHEMA)
        is_llama = type(model).__name__ == "Llama"
        if extra_kwargs:
            logger.debug("Tool selector: using structured JSON decoding (%s)", type(model).__name__)

        call_kwargs = dict(
            messages=messages,
            temperature=temperature,
            max_tokens=500,
            top_p=0.9,
            stream=False,
            **extra_kwargs,
        )

        # Use timeout wrapper for Llama to prevent grammar-constrained hangs
        try:
            if is_llama:
                result = _llama_with_timeout(model, _LLAMA_TIMEOUT_S, **call_kwargs)
            else:
                result = model.create_chat_completion(**call_kwargs)
        except TimeoutError:
            # Grammar-constrained generation hung — cannot safely retry on the
            # same model instance (llama-cpp is not thread-safe and the hung
            # thread may still be using it).  Bail out gracefully.
            logger.warning("Tool selection timed out; skipping tool routing for this turn")
            return None, []

        # Extract response content
        content = _extract_content(result)
        logger.debug(f"Tool selector raw response: {content[:200]}...")

        # Try to extract JSON from the response (handle markdown code blocks)
        json_content = _extract_json_from_response(content)

        # Parse JSON response
        selection = json.loads(json_content)

        # Handle both null and empty string as "no tool"
        selected = selection.get("selected_tool")
        if selected is None or selected == "":
            logger.info(f"Tool selector determined no tool needed: {selection.get('reasoning')}")
            return None, []

        # Parse params — may be a JSON string or an object (depending on
        # whether grammar constraints were active)
        raw_params = selection.get("params", {})
        logger.debug("Tool selector raw params (type=%s): %s", type(raw_params).__name__, str(raw_params)[:200])
        if isinstance(raw_params, str):
            try:
                raw_params = json.loads(raw_params) if raw_params else {}
            except (json.JSONDecodeError, TypeError):
                logger.warning("Failed to parse params string as JSON: %s", raw_params[:100])
                raw_params = {}
        if not isinstance(raw_params, dict):
            raw_params = {}

        primary = SelectedTool(
            tool_name=selected,
            params=raw_params,
            selection_reasoning=selection.get("reasoning", ""),
            selection_confidence=selection.get("confidence", 0.5)
        )

        # Parse alternatives if present (from unconstrained fallback responses)
        alternatives = []
        for alt in selection.get("alternatives", []):
            alt_params = alt.get("params", {})
            if isinstance(alt_params, str):
                try:
                    alt_params = json.loads(alt_params) if alt_params else {}
                except (json.JSONDecodeError, TypeError):
                    alt_params = {}
            alternatives.append(SelectedTool(
                tool_name=alt.get("tool", ""),
                params=alt_params if isinstance(alt_params, dict) else {},
                selection_reasoning=alt.get("reason", ""),
                selection_confidence=0.0
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
        is_llama = type(model).__name__ == "Llama"
        if extra_kwargs:
            logger.debug("Tool review: using structured JSON decoding (%s)", type(model).__name__)

        call_kwargs = dict(
            messages=messages,
            temperature=temperature,
            max_tokens=200,
            stream=False,
            **extra_kwargs,
        )

        # Use timeout wrapper for Llama
        try:
            if is_llama:
                result = _llama_with_timeout(model, _LLAMA_TIMEOUT_S, **call_kwargs)
            else:
                result = model.create_chat_completion(**call_kwargs)
        except TimeoutError:
            # Cannot safely retry on the same model instance.
            # Auto-approve the selection rather than blocking the pipeline.
            logger.warning("Review timed out; auto-approving tool selection")
            return True, 0.5, "Review timed out — auto-approved"

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

    # Try to find raw JSON object (non-greedy to avoid merging multiple objects)
    json_pattern = r'\{[\s\S]*?\}'
    matches = re.findall(json_pattern, content)
    if matches:
        # If multiple matches, try them one by one or take the most complete one
        # For tool selection, the first one is usually the intended one
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

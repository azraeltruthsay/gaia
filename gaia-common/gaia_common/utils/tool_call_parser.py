"""
Tool Call Parser — detects and executes inline tool calls from model output.

Intercepts <tool_call>...</tool_call> tags in streaming model output,
executes the tool via MCP, and injects <tool_result>...</tool_result>
back into the generation context.

This replaces the 3-step heuristic pipeline (intent detection → LLM
selection → LLM review) with model-native tool calling.

Usage:
    parser = ToolCallParser(mcp_client)

    # During streaming generation:
    for token in model.generate(...):
        result = parser.feed(token)
        if result.type == "text":
            yield result.text          # Normal text, pass through
        elif result.type == "tool_executing":
            yield "[calling tool...]"   # Optional: show user
        elif result.type == "tool_result":
            # Inject result back into model context for continuation
            inject_into_context(result.tool_result_xml)
"""

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any

logger = logging.getLogger("GAIA.ToolCallParser")

# Tags
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESULT_OPEN = "<tool_result>"
TOOL_RESULT_CLOSE = "</tool_result>"


class ParseEventType(Enum):
    TEXT = "text"
    TOOL_CALL_DETECTED = "tool_call_detected"
    TOOL_EXECUTING = "tool_executing"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"


@dataclass
class ParseEvent:
    """Event emitted by the parser during streaming."""
    type: ParseEventType
    text: str = ""
    tool_name: str = ""
    tool_action: str = ""
    tool_params: Dict[str, Any] = None
    tool_result: Any = None
    tool_result_xml: str = ""
    error: str = ""


class ToolCallParser:
    """
    Streaming parser for inline tool calls in model output.

    Accumulates tokens, detects <tool_call>...</tool_call> boundaries,
    parses the JSON payload, and signals the caller to execute + reinject.
    """

    def __init__(self):
        self._buffer = ""
        self._in_tool_call = False
        self._tool_calls_found: List[Dict] = []

    def reset(self):
        """Reset parser state for a new generation."""
        self._buffer = ""
        self._in_tool_call = False
        self._tool_calls_found = []

    def feed(self, token: str) -> List[ParseEvent]:
        """
        Feed a token to the parser.

        Returns a list of ParseEvents:
        - TEXT events for normal content
        - TOOL_CALL_DETECTED when a complete tool call is found
        """
        events = []
        self._buffer += token

        while True:
            if not self._in_tool_call:
                # Look for tool call opening tag
                open_idx = self._buffer.find(TOOL_CALL_OPEN)
                if open_idx == -1:
                    # No tag found — emit all buffered text except last few chars
                    # (which might be a partial tag like "<tool_")
                    safe_len = len(self._buffer) - len(TOOL_CALL_OPEN)
                    if safe_len > 0:
                        events.append(ParseEvent(type=ParseEventType.TEXT, text=self._buffer[:safe_len]))
                        self._buffer = self._buffer[safe_len:]
                    break
                else:
                    # Emit text before the tag
                    if open_idx > 0:
                        events.append(ParseEvent(type=ParseEventType.TEXT, text=self._buffer[:open_idx]))
                    self._buffer = self._buffer[open_idx + len(TOOL_CALL_OPEN):]
                    self._in_tool_call = True

            if self._in_tool_call:
                # Look for closing tag
                close_idx = self._buffer.find(TOOL_CALL_CLOSE)
                if close_idx == -1:
                    # Haven't received the full tool call yet — wait for more tokens
                    break
                else:
                    # Extract the tool call JSON
                    tool_json = self._buffer[:close_idx].strip()
                    self._buffer = self._buffer[close_idx + len(TOOL_CALL_CLOSE):]
                    self._in_tool_call = False

                    # Parse the tool call
                    event = self._parse_tool_call(tool_json)
                    events.append(event)
                    self._tool_calls_found.append({
                        "tool": event.tool_name,
                        "action": event.tool_action,
                        "params": event.tool_params,
                    })

        return events

    def flush(self) -> List[ParseEvent]:
        """Flush any remaining buffered text."""
        events = []
        if self._buffer:
            if self._in_tool_call:
                # Unclosed tool call — emit as text (malformed)
                logger.warning("Unclosed <tool_call> tag — emitting as text")
                events.append(ParseEvent(type=ParseEventType.TEXT, text=TOOL_CALL_OPEN + self._buffer))
            else:
                events.append(ParseEvent(type=ParseEventType.TEXT, text=self._buffer))
            self._buffer = ""
            self._in_tool_call = False
        return events

    def _parse_tool_call(self, json_str: str) -> ParseEvent:
        """Parse a tool call JSON string into a ParseEvent."""
        try:
            data = json.loads(json_str)
            tool = data.get("tool", "")
            action = data.get("action", "")
            # Remove tool and action from params, keep the rest
            params = {k: v for k, v in data.items() if k not in ("tool", "action")}

            logger.info("Tool call detected: %s(action=%s, %s)", tool, action, params)

            return ParseEvent(
                type=ParseEventType.TOOL_CALL_DETECTED,
                tool_name=tool,
                tool_action=action,
                tool_params=params,
            )
        except json.JSONDecodeError as e:
            logger.warning("Malformed tool call JSON: %s — %s", json_str[:100], e)
            return ParseEvent(
                type=ParseEventType.TOOL_ERROR,
                error=f"Malformed tool call: {e}",
                text=f"{TOOL_CALL_OPEN}{json_str}{TOOL_CALL_CLOSE}",
            )

    @property
    def tool_calls(self) -> List[Dict]:
        """Return all tool calls detected so far."""
        return self._tool_calls_found


def format_tool_result(result: Any) -> str:
    """Format a tool execution result as XML for context injection."""
    if isinstance(result, dict):
        result_str = json.dumps(result, default=str, ensure_ascii=False)
    elif isinstance(result, str):
        result_str = result
    else:
        result_str = str(result)

    # Truncate very long results to avoid context overflow
    if len(result_str) > 4000:
        result_str = result_str[:4000] + "\n... (truncated)"

    return f"{TOOL_RESULT_OPEN}{result_str}{TOOL_RESULT_CLOSE}"


def build_tool_schema_injection() -> str:
    """
    Build the tool schema string to inject in the system prompt.

    This is the SAME format used in Primary School training data,
    so the model recognizes the syntax at inference time.
    """
    try:
        from gaia_common.utils.domain_tools import DOMAIN_TOOLS, DOMAIN_ACTIONS, SENSITIVE_ACTIONS
    except ImportError:
        return ""

    lines = ["You have these tools available:"]
    for domain, spec in DOMAIN_TOOLS.items():
        if spec.get("dynamic"):
            lines.append(f"- fabric(pattern, input): {spec['description']}")
            continue
        actions = DOMAIN_ACTIONS.get(domain, [])
        action_str = "|".join(actions[:8])
        if len(actions) > 8:
            action_str += f"|... ({len(actions)} total)"
        lines.append(f"- {domain}(action): {action_str}")

    lines.append("")
    lines.append(f"Call tools inline: {TOOL_CALL_OPEN}{{\"tool\":\"domain\",\"action\":\"verb\",...}}{TOOL_CALL_CLOSE}")
    lines.append(f"Results appear as: {TOOL_RESULT_OPEN}...{TOOL_RESULT_CLOSE}")
    return "\n".join(lines)

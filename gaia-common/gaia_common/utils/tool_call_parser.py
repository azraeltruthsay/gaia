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

# Tags — the parser recognizes both <tool_call> and <tool_response> as call tags,
# since some model variants emit one or the other depending on training data.
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"
TOOL_RESULT_OPEN = "<tool_result>"
TOOL_RESULT_CLOSE = "</tool_result>"

# Meta-verb format — Unified Skill Architecture.
# Uses Gemma 4's NATIVE special tokens (single token IDs 46-51):
#   <|tool>verb(param=value, ...)<tool|>     — model emits a tool call
#   <|tool_response>...content...<tool_response|>  — injected result
# These are single tokens in the Gemma 4 vocabulary, not multi-token sequences.
META_TOOL_OPEN = "<|tool>"
META_TOOL_CLOSE = "<tool|>"
META_RESULT_OPEN = "<|tool_response>"
META_RESULT_CLOSE = "<tool_response|>"
# Also support the 4-token fallback for non-Gemma models
META_TOOL_OPEN_ALT = "<|tool|>"
META_TOOL_CLOSE_ALT = "<|/tool|>"

# Regex for parsing verb(param=value, param=value) format
_META_VERB_RE = re.compile(
    r'^(\w+)\((.*)\)$', re.DOTALL
)
# Parse key=value or key="value with spaces"
_META_PARAM_RE = re.compile(
    r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|(\S+))'
)


def parse_meta_verb(raw: str) -> Optional[Dict[str, Any]]:
    """Parse a meta-verb call string into tool_name and params.

    Examples:
        'search(query="current time")' → {"tool": "search", "params": {"query": "current time"}}
        'do(skill="web-search", input="bitcoin")' → {"tool": "do", "params": {"skill": "web-search", ...}}
        'remember(fact="GAIA uses Gemma 4")' → {"tool": "remember", "params": {"fact": "..."}}

    Returns:
        Dict with "tool" and "params" keys, or None if parsing fails.
    """
    raw = raw.strip()
    m = _META_VERB_RE.match(raw)
    if not m:
        return None

    verb = m.group(1)
    args_str = m.group(2).strip()

    params = {}
    if args_str:
        for pm in _META_PARAM_RE.finditer(args_str):
            key = pm.group(1)
            # Groups 2, 3, 4 are the three capture alternatives (double-quoted, single-quoted, unquoted)
            value = pm.group(2) if pm.group(2) is not None else (
                pm.group(3) if pm.group(3) is not None else pm.group(4)
            )
            # Parse booleans and numbers
            if value is not None:
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                else:
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        try:
                            value = float(value)
                        except (ValueError, TypeError):
                            pass
            params[key] = value

    return {"tool": verb, "params": params}


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
        self._is_meta_verb = False
        self._tool_calls_found: List[Dict] = []

    def reset(self):
        """Reset parser state for a new generation."""
        self._buffer = ""
        self._in_tool_call = False
        self._is_meta_verb = False
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
                # Look for tool call opening tags — check all supported formats.
                # Priority: Gemma 4 native tokens first, then XML-style tags.
                _candidates = [
                    (self._buffer.find(META_TOOL_OPEN), META_TOOL_OPEN, META_TOOL_CLOSE, True),
                    (self._buffer.find(TOOL_CALL_OPEN), TOOL_CALL_OPEN, TOOL_CALL_CLOSE, False),
                    (self._buffer.find(TOOL_RESPONSE_OPEN), TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE, False),
                ]
                # Pick the earliest match
                best = None
                for idx, otag, ctag, is_meta in _candidates:
                    if idx != -1 and (best is None or idx < best[0]):
                        best = (idx, otag, ctag, is_meta)

                if best is None:
                    open_idx = -1
                else:
                    open_idx, open_tag, close_tag, self._is_meta_verb = best

                if open_idx == -1:
                    # No tag found — emit all buffered text except last few chars
                    # (which might be a partial tag like "<tool_" or "<|tool")
                    _longest_tag = max(len(META_TOOL_OPEN), len(TOOL_RESPONSE_OPEN))
                    safe_len = len(self._buffer) - _longest_tag
                    if safe_len > 0:
                        events.append(ParseEvent(type=ParseEventType.TEXT, text=self._buffer[:safe_len]))
                        self._buffer = self._buffer[safe_len:]
                    break
                else:
                    # Emit text before the tag
                    if open_idx > 0:
                        events.append(ParseEvent(type=ParseEventType.TEXT, text=self._buffer[:open_idx]))
                    self._buffer = self._buffer[open_idx + len(open_tag):]
                    self._in_tool_call = True
                    self._active_close_tag = close_tag

            if self._in_tool_call:
                # Look for closing tag
                close_tag = getattr(self, '_active_close_tag', TOOL_CALL_CLOSE)
                close_idx = self._buffer.find(close_tag)
                if close_idx == -1:
                    # Haven't received the full tool call yet — wait for more tokens
                    break
                else:
                    # Extract the tool call content
                    tool_content = self._buffer[:close_idx].strip()
                    self._buffer = self._buffer[close_idx + len(close_tag):]
                    self._in_tool_call = False

                    # Parse based on format
                    if getattr(self, '_is_meta_verb', False):
                        # Meta-verb format: verb(param=value, ...)
                        parsed = parse_meta_verb(tool_content)
                        if parsed:
                            event = ParseEvent(
                                type=ParseEventType.TOOL_CALL_DETECTED,
                                tool_name=parsed["tool"],
                                tool_params=parsed["params"],
                            )
                        else:
                            event = ParseEvent(
                                type=ParseEventType.TOOL_ERROR,
                                error=f"Failed to parse meta-verb: {tool_content[:80]}",
                                text=META_TOOL_OPEN + tool_content + META_TOOL_CLOSE,
                            )
                    else:
                        # JSON format: {"tool": "...", "action": "...", ...}
                        event = self._parse_tool_call(tool_content)

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
        """Parse a tool call JSON string into a ParseEvent.

        Includes repair logic for common model malformations:
        - ``"tool":"name":"web_search"`` → ``"tool_name":"web_search"``
        - ``"action":"query":`` → ``"action":"query",``
        """
        import re as _re
        raw = json_str.strip()

        # Repair: "tool":"name":"actual" → "tool_name":"actual"
        # Small models often emit this double-colon pattern
        repaired = _re.sub(
            r'"tool"\s*:\s*"name"\s*:\s*"',
            '"tool_name":"',
            raw,
        )
        # Repair: "action":"verb": "value" → "action":"verb", "value_key": "value"
        repaired = _re.sub(
            r'"action"\s*:\s*"(\w+)"\s*:\s*"',
            r'"action":"\1", "query":"',
            repaired,
        )

        for attempt in (repaired, raw):
            try:
                data = json.loads(attempt)
                # Normalize: accept both "tool" and "tool_name" keys
                tool = data.get("tool") or data.get("tool_name") or data.get("name") or ""
                action = data.get("action", "")
                # Remove tool and action from params, keep the rest
                params = {k: v for k, v in data.items() if k not in ("tool", "tool_name", "name", "action")}

                if attempt != raw:
                    logger.info("Tool call JSON repaired: %s → %s", raw[:80], attempt[:80])

                logger.info("Tool call detected: %s(action=%s, %s)", tool, action, params)

                return ParseEvent(
                    type=ParseEventType.TOOL_CALL_DETECTED,
                    tool_name=tool,
                    tool_action=action,
                    tool_params=params,
                )
            except json.JSONDecodeError:
                continue

        logger.warning("Malformed tool call JSON (repair failed): %s", raw[:100])
        return ParseEvent(
            type=ParseEventType.TOOL_ERROR,
            error=f"Malformed tool call JSON",
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

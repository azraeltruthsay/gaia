# GCP Tool Routing System - Implementation Plan

**Date:** 2026-01-23
**Status:** Design Phase
**Author:** Claude Code / Azrael

---

## 1. Overview

This document outlines the implementation plan for a structured MCP tool routing system integrated with the GAIA Cognition Packet (GCP). The goal is to eliminate hallucinated tool patterns (like `<|Start of Memory Helper Tool Call|>`) by routing tool decisions through a controlled, low-temperature selection process with confidence review.

### Current Problem

The model sometimes generates hallucinated tool call patterns instead of using the real MCP tools. This happens because:
1. Tool instructions are embedded in persona text
2. Model temperature is optimized for conversation, not structured selection
3. No confidence gate before tool execution
4. No packet reinjection loop for multi-stage processing

### Proposed Solution

A packet-based tool routing system with:
- **Packet ID tracking** for multi-stage processing
- **Intent detection** decides if MCP tool is needed
- **Low-temperature tool selection** with structured output
- **Confidence review** before execution
- **Result injection** back into packet for final response

---

## 2. Architecture Flow

```
[User Input]
     │
     ▼
[Create GCP Packet] ─────────────────────────────────────┐
     │                                                    │
     ▼                                                    │
[INTENT DETECTION] ◄──────────────────────────────────────┤
     │                                                    │
     ├─── needs_tool=False ──► [NORMAL RESPONSE PATH] ───►│
     │                                                    │
     └─── needs_tool=True                                 │
           │                                              │
           ▼                                              │
     [TOOL SELECTION MODULE]                              │
     │  - Low temperature (0.1-0.2)                       │
     │  - Structured JSON output                          │
     │  - Select tool + params from available_mcp_tools   │
           │                                              │
           ▼                                              │
     [CONFIDENCE REVIEW]                                  │
     │  - Prime model reviews selection                   │
     │  - confidence >= 0.7 → proceed                     │
     │  - confidence < 0.7 → ask user or skip             │
           │                                              │
           ├─── confidence LOW ──► [ASK USER / SKIP] ────►│
           │                                              │
           └─── confidence HIGH                           │
                 │                                        │
                 ▼                                        │
           [EXECUTE MCP TOOL]                             │
           │  - Call via mcp_client                       │
           │  - Capture result/error                      │
                 │                                        │
                 ▼                                        │
           [INJECT RESULT INTO PACKET]                    │
           │  - packet.reasoning.tool_result = result     │
           │  - packet.status.state = PROCESSING          │
                 │                                        │
                 ▼                                        │
           [REINJECT FOR FINAL RESPONSE] ─────────────────┘
                 │
                 ▼
           [Response to User]
```

---

## 3. Data Structures

### 3.1 New: ToolRoutingState (cognition_packet.py)

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

class ToolExecutionStatus(Enum):
    PENDING = "pending"           # Tool selected but not yet executed
    AWAITING_CONFIDENCE = "awaiting_confidence"  # Waiting for confidence review
    APPROVED = "approved"         # Confidence passed, ready to execute
    EXECUTED = "executed"         # Tool successfully executed
    FAILED = "failed"             # Tool execution failed
    SKIPPED = "skipped"           # Low confidence, skipped
    USER_DENIED = "user_denied"   # User rejected tool use


@dataclass_json
@dataclass
class SelectedTool:
    """Represents a tool selected for potential execution."""
    tool_name: str                              # e.g., "ai.read", "ai.execute"
    params: Dict[str, Any] = field(default_factory=dict)
    selection_reasoning: str = ""               # Why this tool was selected
    selection_confidence: float = 0.0           # Model's confidence in selection


@dataclass_json
@dataclass
class ToolExecutionResult:
    """Result from executing an MCP tool."""
    success: bool
    output: Any = None                          # Tool output (varies by tool)
    error: Optional[str] = None                 # Error message if failed
    execution_time_ms: int = 0


@dataclass_json
@dataclass
class ToolRoutingState:
    """
    Tracks the state of tool routing within a cognition packet.

    This dataclass captures the full lifecycle of tool selection and execution:
    1. Intent detection flags needs_tool
    2. Tool selector picks a tool with reasoning
    3. Confidence review approves/rejects
    4. Execution produces result
    5. Result is available for final response generation
    """
    # Decision flags
    needs_tool: bool = False                    # Intent detection decided tool needed
    routing_requested: bool = False             # Packet marked for tool routing loop

    # Selection state
    selected_tool: Optional[SelectedTool] = None
    alternative_tools: List[SelectedTool] = field(default_factory=list)

    # Confidence review
    review_confidence: float = 0.0              # Prime model's review confidence
    review_reasoning: str = ""                  # Prime model's review reasoning

    # Execution state
    execution_status: ToolExecutionStatus = ToolExecutionStatus.PENDING
    execution_result: Optional[ToolExecutionResult] = None

    # Reinjection tracking
    reinjection_count: int = 0                  # How many times this packet was reinjected
    max_reinjections: int = 3                   # Safety limit
```

### 3.2 Integration with CognitionPacket

Add to `CognitionPacket` class:

```python
@dataclass_json
@dataclass
class CognitionPacket:
    # ... existing fields ...

    # NEW: Tool routing state for MCP integration
    tool_routing: Optional[ToolRoutingState] = None
```

### 3.3 New Intent Flag

Update `SystemTask` enum in `cognition_packet.py`:

```python
class SystemTask(Enum):
    INTENT_DETECTION = "IntentDetection"
    RESEARCH = "Research"
    GENERATE_DRAFT = "GenerateDraft"
    REFINE = "Refine"
    VALIDATE = "Validate"
    DECISION = "Decision"
    STREAM = "Stream"
    TRIGGER_ACTION = "TriggerAction"
    TOOL_ROUTING = "ToolRouting"          # NEW: Packet needs tool routing
    TOOL_EXECUTION = "ToolExecution"      # NEW: Tool approved, execute
```

---

## 4. Module Design

### 4.1 tool_selector.py (NEW FILE)

Location: `app/cognition/tool_selector.py`

```python
"""
Tool Selector Module

Responsible for:
1. Determining if a request needs MCP tool usage
2. Selecting the appropriate tool with low-temperature generation
3. Extracting structured parameters
4. Providing confidence scores for review
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from app.cognition.cognition_packet import (
    CognitionPacket, SelectedTool, ToolRoutingState, ToolExecutionStatus
)

logger = logging.getLogger("GAIA.ToolSelector")

# Tool definitions with descriptions for selection prompt
AVAILABLE_TOOLS = {
    "ai.read": {
        "description": "Read a file from the filesystem",
        "params": ["path"],
        "param_descriptions": {
            "path": "Absolute path to the file to read"
        },
        "examples": [
            {"path": "/gaia-assistant/knowledge/system_reference/dev_matrix.json"},
            {"path": "/gaia-assistant/docs/gaia_core_blueprint.md"}
        ]
    },
    "ai.write": {
        "description": "Write content to a file on the filesystem",
        "params": ["path", "content"],
        "param_descriptions": {
            "path": "Absolute path to the file to write",
            "content": "Content to write to the file"
        },
        "requires_approval": True
    },
    "ai.execute": {
        "description": "Execute a shell command",
        "params": ["command"],
        "param_descriptions": {
            "command": "Shell command to execute"
        },
        "requires_approval": True
    },
    "embedding.query": {
        "description": "Query the vector database for semantic search",
        "params": ["query", "top_k"],
        "param_descriptions": {
            "query": "Search query text",
            "top_k": "Number of results to return (default: 5)"
        }
    }
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
            return True

    # Check if available_mcp_tools in context suggests tool availability
    if packet.context.available_mcp_tools:
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
        result = model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=500,
            top_p=0.9,
            stream=False
        )

        # Extract response content
        content = _extract_content(result)

        # Parse JSON response
        selection = json.loads(content)

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
        result = model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=200,
            stream=False
        )

        content = _extract_content(result)
        review = json.loads(content)

        confidence = review.get("confidence", 0.0)
        reasoning = review.get("reasoning", "")

        if not review.get("approved", False):
            confidence = min(confidence, 0.5)  # Cap confidence if not approved

        return confidence, reasoning

    except Exception as e:
        logger.error(f"Review failed: {e}")
        return 0.0, f"Review failed: {e}"


def _build_tool_catalog() -> str:
    """Build a formatted catalog of available tools for the selection prompt."""
    lines = []
    for tool_name, info in AVAILABLE_TOOLS.items():
        lines.append(f"- {tool_name}: {info['description']}")
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
```

### 4.2 Modifications to intent_detection.py

Add tool routing detection:

```python
# In model_intent_detection() or detect_intent()

def _detect_tool_routing_needed(text: str) -> bool:
    """
    Detect if the request likely needs MCP tool routing.
    """
    lowered = (text or "").lower()

    # Strong tool indicators
    tool_patterns = [
        r"read\s+(?:the\s+)?(?:file|document)",
        r"(?:show|display|open|view)\s+(?:me\s+)?(?:the\s+)?[\w/]+\.(md|txt|json|py)",
        r"execute\s+(?:the\s+)?(?:command|script)",
        r"run\s+(?:the\s+)?(?:command|script)",
        r"search\s+(?:for|the)\s+",
        r"find\s+(?:the\s+)?(?:file|document)",
    ]

    for pattern in tool_patterns:
        if re.search(pattern, lowered):
            return True

    return False

# Update detect_intent() to return tool_routing intent:
def detect_intent(...) -> Plan:
    # ... existing logic ...

    # Check for tool routing
    if _detect_tool_routing_needed(text):
        return Plan(intent="tool_routing", read_only=False)

    # ... rest of existing logic ...
```

### 4.3 Modifications to agent_core.py

Add tool routing loop to the cognitive cycle:

```python
# In AgentCore class

def _run_tool_routing_loop(
    self,
    packet: CognitionPacket,
    user_input: str
) -> CognitionPacket:
    """
    Execute the tool routing loop for packets that need MCP tools.

    This method:
    1. Selects appropriate tool with low-temp generation
    2. Reviews selection for confidence
    3. Executes if confidence is high
    4. Injects result back into packet

    Returns the modified packet.
    """
    from app.cognition.tool_selector import (
        needs_tool_routing, select_tool, review_selection,
        AVAILABLE_TOOLS
    )
    from app.utils import mcp_client

    # Initialize tool routing state if not present
    if packet.tool_routing is None:
        packet.tool_routing = ToolRoutingState()

    # Safety check: prevent infinite loops
    if packet.tool_routing.reinjection_count >= packet.tool_routing.max_reinjections:
        logger.warning(f"Max reinjections reached for packet {packet.header.packet_id}")
        packet.tool_routing.execution_status = ToolExecutionStatus.SKIPPED
        return packet

    packet.tool_routing.reinjection_count += 1

    # Step 1: Tool Selection (low temperature)
    logger.info(f"Tool routing: selecting tool for packet {packet.header.packet_id}")

    # Use Lite model for selection if available, otherwise Prime
    selection_model = self.model_pool.lite or self.model_pool.prime

    primary_tool, alternatives = select_tool(
        packet=packet,
        user_input=user_input,
        model=selection_model,
        temperature=0.15
    )

    if primary_tool is None:
        logger.info("Tool selector determined no tool needed")
        packet.tool_routing.needs_tool = False
        packet.tool_routing.execution_status = ToolExecutionStatus.SKIPPED
        return packet

    packet.tool_routing.needs_tool = True
    packet.tool_routing.selected_tool = primary_tool
    packet.tool_routing.alternative_tools = alternatives

    # Step 2: Confidence Review (Prime model)
    logger.info(f"Tool routing: reviewing selection {primary_tool.tool_name}")

    review_model = self.model_pool.prime or selection_model
    confidence, reasoning = review_selection(
        packet=packet,
        selected_tool=primary_tool,
        model=review_model,
        temperature=0.3
    )

    packet.tool_routing.review_confidence = confidence
    packet.tool_routing.review_reasoning = reasoning

    # Step 3: Confidence Gate
    CONFIDENCE_THRESHOLD = 0.7

    if confidence < CONFIDENCE_THRESHOLD:
        logger.warning(f"Tool selection confidence too low: {confidence} < {CONFIDENCE_THRESHOLD}")
        packet.tool_routing.execution_status = ToolExecutionStatus.SKIPPED
        # Add to reasoning log
        packet.reasoning.reflection_log.append(ReflectionLog(
            step="tool_routing",
            summary=f"Tool {primary_tool.tool_name} skipped due to low confidence ({confidence:.2f})",
            confidence=confidence
        ))
        return packet

    # Step 4: Execute Tool
    logger.info(f"Tool routing: executing {primary_tool.tool_name}")
    packet.tool_routing.execution_status = ToolExecutionStatus.APPROVED

    try:
        result = self._execute_mcp_tool(primary_tool)
        packet.tool_routing.execution_result = result
        packet.tool_routing.execution_status = (
            ToolExecutionStatus.EXECUTED if result.success
            else ToolExecutionStatus.FAILED
        )

        # Add result to reasoning log
        packet.reasoning.reflection_log.append(ReflectionLog(
            step="tool_execution",
            summary=f"Executed {primary_tool.tool_name}: {'success' if result.success else 'failed'}",
            confidence=confidence
        ))

    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        packet.tool_routing.execution_status = ToolExecutionStatus.FAILED
        packet.tool_routing.execution_result = ToolExecutionResult(
            success=False,
            error=str(e)
        )

    return packet


def _execute_mcp_tool(self, tool: SelectedTool) -> ToolExecutionResult:
    """
    Execute an MCP tool and return the result.
    """
    from app.utils import mcp_client
    import time

    start_time = time.time()

    try:
        if tool.tool_name == "ai.read":
            result = mcp_client.ai_read(tool.params.get("path", ""))
        elif tool.tool_name == "ai.write":
            result = mcp_client.ai_write(
                tool.params.get("path", ""),
                tool.params.get("content", "")
            )
        elif tool.tool_name == "ai.execute":
            result = mcp_client.ai_execute(
                tool.params.get("command", ""),
                dry_run=not self.config.constants.get("ALLOW_SHELL_EXECUTION", False)
            )
        elif tool.tool_name == "embedding.query":
            result = mcp_client.embedding_query(
                tool.params.get("query", ""),
                top_k=tool.params.get("top_k", 5)
            )
        else:
            return ToolExecutionResult(
                success=False,
                error=f"Unknown tool: {tool.tool_name}"
            )

        elapsed_ms = int((time.time() - start_time) * 1000)

        return ToolExecutionResult(
            success=result.get("ok", False),
            output=result,
            error=result.get("error"),
            execution_time_ms=elapsed_ms
        )

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        return ToolExecutionResult(
            success=False,
            error=str(e),
            execution_time_ms=elapsed_ms
        )
```

---

## 5. Integration Points

### 5.1 Main Cognitive Loop Integration

In `AgentCore.process()` or equivalent entry point:

```python
def process(self, user_input: str, session_id: str, ...) -> Generator:
    # ... create packet ...

    # Intent detection
    plan = detect_intent(user_input, self.config, lite_llm=self.model_pool.lite)

    # Check if tool routing is needed
    if plan.intent == "tool_routing" or needs_tool_routing(packet, user_input):
        packet = self._run_tool_routing_loop(packet, user_input)

        # If tool was executed, inject result into prompt for final response
        if packet.tool_routing and packet.tool_routing.execution_result:
            self._inject_tool_result_into_context(packet)

    # Continue with normal response generation
    # ... existing code ...
```

### 5.2 Context Injection for Final Response

```python
def _inject_tool_result_into_context(self, packet: CognitionPacket):
    """
    Inject tool execution result into the packet's context for final response.
    """
    if not packet.tool_routing or not packet.tool_routing.execution_result:
        return

    result = packet.tool_routing.execution_result
    tool = packet.tool_routing.selected_tool

    # Create a data field with the tool result
    result_field = DataField(
        key="tool_result",
        value={
            "tool": tool.tool_name,
            "params": tool.params,
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
```

---

## 6. Implementation Order

### Phase 1: Data Structures (cognition_packet.py)
1. Add `ToolExecutionStatus` enum
2. Add `SelectedTool` dataclass
3. Add `ToolExecutionResult` dataclass
4. Add `ToolRoutingState` dataclass
5. Add `tool_routing` field to `CognitionPacket`
6. Update `SystemTask` enum with new states

### Phase 2: Tool Selector Module (NEW: tool_selector.py)
1. Create `AVAILABLE_TOOLS` catalog
2. Implement `needs_tool_routing()` heuristic
3. Implement `select_tool()` with low-temp generation
4. Implement `review_selection()` confidence check
5. Add unit tests

### Phase 3: Agent Core Integration (agent_core.py)
1. Add `_run_tool_routing_loop()` method
2. Add `_execute_mcp_tool()` method
3. Add `_inject_tool_result_into_context()` method
4. Integrate into main processing loop
5. Add logging and error handling

### Phase 4: Intent Detection Updates (intent_detection.py)
1. Add `_detect_tool_routing_needed()` function
2. Add `"tool_routing"` intent to valid intents
3. Update `detect_intent()` to check for tool routing

### Phase 5: Testing
1. Unit tests for tool_selector.py
2. Integration tests for tool routing loop
3. End-to-end test with Discord connector
4. Test with various tool scenarios (read, write, execute, query)

---

## 7. Configuration

Add to `gaia_constants.json`:

```json
{
    "TOOL_ROUTING": {
        "ENABLED": true,
        "SELECTION_TEMPERATURE": 0.15,
        "REVIEW_TEMPERATURE": 0.3,
        "CONFIDENCE_THRESHOLD": 0.7,
        "MAX_REINJECTIONS": 3,
        "ALLOW_WRITE_TOOLS": false,
        "ALLOW_EXECUTE_TOOLS": false
    }
}
```

---

## 8. Safety Considerations

1. **Dry Run by Default**: Write and execute tools should be dry-run by default
2. **Confidence Gate**: Never execute below threshold without user approval
3. **Reinjection Limit**: Prevent infinite loops with max_reinjections
4. **Approval Flow**: Integrate with existing MCP approval system for dangerous operations
5. **Logging**: Comprehensive logging for all tool operations
6. **Audit Trail**: Tool executions recorded in packet's reasoning log

---

## 9. Success Metrics

1. **Elimination of hallucinated tool calls**: Model no longer generates `<|Start of Memory Helper Tool Call|>` patterns
2. **Accurate tool selection**: Tools selected match user intent >90% of the time
3. **Confidence calibration**: High-confidence selections succeed, low-confidence appropriately skipped
4. **Response quality**: Final responses correctly incorporate tool results
5. **Performance**: Tool routing adds <500ms latency to processing

---

## 10. Future Enhancements

1. **Tool Chains**: Support for multi-step tool workflows
2. **Learning**: Track tool selection patterns for improved heuristics
3. **Custom Tools**: Allow dynamic tool registration via MCP
4. **Parallel Execution**: Execute multiple independent tools concurrently
5. **Caching**: Cache frequently-used tool results (e.g., file reads)

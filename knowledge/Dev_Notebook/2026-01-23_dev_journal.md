# Dev Journal - 2026-01-23

## Session Focus: GCP Tool Routing System - IMPLEMENTATION COMPLETE

### Summary

This session focused on designing AND implementing a comprehensive solution to the hallucinated tool call problem. Instead of the model generating patterns like `<|Start of Memory Helper Tool Call|>`, we implemented a structured GCP-based tool routing system.

**Status: ALL PHASES IMPLEMENTED** ✅

---

## Problem Statement

GAIA's model sometimes generates hallucinated tool call patterns instead of using real MCP tools. Root causes:
1. Tool instructions embedded loosely in persona text
2. Conversation-optimized temperature (0.7) not suitable for structured selection
3. No confidence gate before tool execution
4. No packet reinjection loop for multi-stage processing

---

## Solution: GCP Tool Routing System

### Architecture Overview

```
[User Input] → [GCP Packet] → [Intent Detection]
    ↓
[needs_tool?] ─No→ [Normal Response]
    ↓ Yes
[Tool Selection] (temp=0.15, structured JSON)
    ↓
[Confidence Review] (Prime model)
    ↓
[confidence ≥ 0.7?] ─No→ [Skip/Ask User]
    ↓ Yes
[Execute MCP Tool]
    ↓
[Inject Result → Packet]
    ↓
[Final Response Generation]
```

### Key Design Decisions

1. **Low-Temperature Selection**: Tool selection uses temp=0.15 for deterministic, structured output
2. **Two-Stage Review**: Selection model picks tool, Prime model reviews for confidence
3. **Confidence Threshold**: 0.7 default, configurable via `gaia_constants.json`
4. **Reinjection Limit**: Max 3 reinjections to prevent infinite loops
5. **Audit Trail**: All tool operations logged in packet's `reasoning.reflection_log`

---

## Implementation Status

### Phase 1: Data Structures ✅ COMPLETE
- Added `ToolExecutionStatus` enum to `cognition_packet.py`
- Added `SelectedTool` dataclass
- Added `ToolExecutionResult` dataclass
- Added `ToolRoutingState` dataclass
- Added `tool_routing` field to `CognitionPacket`
- Added `TOOL_ROUTING` and `TOOL_EXECUTION` to `SystemTask` enum

### Phase 2: Tool Selector Module ✅ COMPLETE
- Created `app/cognition/tool_selector.py`
- Implemented `needs_tool_routing()` - fast heuristic check
- Implemented `select_tool()` - low-temp structured JSON selection
- Implemented `review_selection()` - confidence review
- Implemented `initialize_tool_routing()` - packet initialization
- Implemented `inject_tool_result_into_packet()` - result injection
- Added `AVAILABLE_TOOLS` catalog with ai.read, ai.write, ai.execute, embedding.query

### Phase 3: Agent Core Integration ✅ COMPLETE
- Added `_run_tool_routing_loop()` method to AgentCore
- Added `_execute_mcp_tool()` method
- Added `_should_use_tool_routing()` method
- Integrated into main `run_turn()` loop after intent detection
- Added comprehensive logging via `ts_write()`

### Phase 4: Intent Detection Updates ✅ COMPLETE
- Added `_detect_tool_routing_request()` function
- Added `"tool_routing"` to valid intents list
- Integrated detection before model-based intent detection
- Patterns detect explicit MCP tool requests, file operations, command execution

### Phase 5: Configuration ✅ COMPLETE
- Added `TOOL_ROUTING` section to `gaia_constants.json`

---

## Files Modified

| File | Status | Changes |
|------|--------|---------|
| `app/cognition/cognition_packet.py` | MODIFIED | Added ToolExecutionStatus, SelectedTool, ToolExecutionResult, ToolRoutingState, updated CognitionPacket and SystemTask |
| `app/cognition/tool_selector.py` | **NEW** | Complete tool selection module with AVAILABLE_TOOLS catalog |
| `app/cognition/agent_core.py` | MODIFIED | Added imports, _run_tool_routing_loop(), _execute_mcp_tool(), _should_use_tool_routing(), integrated into run_turn() |
| `app/cognition/nlu/intent_detection.py` | MODIFIED | Added _detect_tool_routing_request(), added tool_routing to valid intents |
| `app/gaia_constants.json` | MODIFIED | Added TOOL_ROUTING configuration section |
| `Dev_Notebook/2026-01-23_gcp_tool_routing_plan.md` | **NEW** | Full implementation plan |
| `Dev_Notebook/2026-01-23_dev_journal.md` | **NEW** | This journal |

---

## Configuration Added

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

## New Data Structures

```python
# In cognition_packet.py

class ToolExecutionStatus(Enum):
    PENDING = "pending"
    AWAITING_CONFIDENCE = "awaiting_confidence"
    APPROVED = "approved"
    EXECUTED = "executed"
    FAILED = "failed"
    SKIPPED = "skipped"
    USER_DENIED = "user_denied"

@dataclass
class SelectedTool:
    tool_name: str
    params: Dict[str, Any]
    selection_reasoning: str
    selection_confidence: float

@dataclass
class ToolExecutionResult:
    success: bool
    output: Any
    error: Optional[str]
    execution_time_ms: int

@dataclass
class ToolRoutingState:
    needs_tool: bool
    routing_requested: bool
    selected_tool: Optional[SelectedTool]
    alternative_tools: List[SelectedTool]
    review_confidence: float
    review_reasoning: str
    execution_status: ToolExecutionStatus
    execution_result: Optional[ToolExecutionResult]
    reinjection_count: int
    max_reinjections: int = 3
```

---

## Available Tools Catalog

```python
AVAILABLE_TOOLS = {
    "ai.read": {
        "description": "Read a file from the filesystem",
        "params": ["path"],
        "requires_approval": False
    },
    "ai.write": {
        "description": "Write content to a file on the filesystem",
        "params": ["path", "content"],
        "requires_approval": True
    },
    "ai.execute": {
        "description": "Execute a shell command",
        "params": ["command"],
        "requires_approval": True
    },
    "embedding.query": {
        "description": "Query the vector database for semantic search",
        "params": ["query", "top_k"],
        "requires_approval": False
    }
}
```

---

## Safety Features Implemented

1. **Dry Run Default**: Write and execute tools disabled by default (`ALLOW_WRITE_TOOLS=false`, `ALLOW_EXECUTE_TOOLS=false`)
2. **Confidence Gate**: Execution skipped if confidence < 0.7
3. **Reinjection Limit**: Hard cap at 3 iterations to prevent infinite loops
4. **Audit Trail**: Every tool operation logged in packet's `reasoning.reflection_log` and via `ts_write()`
5. **Structured Selection**: Low-temperature (0.15) JSON output prevents hallucinated formats

---

## Previous Session Fixes (Still in Working Tree)

### Think Tag Stripping
- Added `_strip_think_tags_robust()` to `output_router.py`
- Handles unclosed tags, loops, truncated tags
- Fallback message if stripping leaves empty response

### Build Script
- Added `--no-cache` to `gaia_start.sh` for reliable code updates

### Documentation
- Added stop token explanation to `hf_prompting.py`
- Explains why `</think>` is intentionally NOT a stop token

---

## Next Steps

1. **Test the implementation** - Rebuild and test with Discord
2. **Verify tool selection** - Test with explicit file read requests
3. **Monitor logs** - Check `ts_write()` output for tool routing stages
4. **Tune thresholds** - Adjust confidence threshold if needed based on real-world results

---

## Testing Commands

```bash
# Rebuild and start GAIA
./gaia_start.sh

# Or manually:
docker compose -f docker-compose.single.yml down
docker compose -f docker-compose.single.yml build --no-cache gaia-assistant gaia-mcp-lite
docker compose -f docker-compose.single.yml up -d gaia-assistant gaia-mcp-lite
docker exec -it gaia-assistant python3 gaia_rescue.py --discord

# Test prompts to trigger tool routing:
# - "read the file /gaia-assistant/CLAUDE.md"
# - "show me the contents of /knowledge/system_reference/dev_matrix.json"
# - "use mcp to read /gaia-assistant/docs/gaia_core_blueprint.md"
```

---

## Notes

- The existing `mcp_client.py` already has `ai_read()`, `ai_write()`, `ai_execute()`, and `embedding_query()` functions ready to use
- Intent detection uses low-temp (0.0) for structured output - tool selection uses 0.15
- The `CognitionPacket` now has `tool_routing` field for the complete lifecycle
- Tool routing integrates AFTER intent detection but BEFORE the slim prompt path
- Results are injected into `content.data_fields` and `reasoning.sketchpad` for downstream use

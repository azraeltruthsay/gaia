# CognitionPacket v0.3 Schema Reference

> Source: `gaia-common/gaia_common/protocols/cognition_packet.py`

## Top-Level Structure

```
CognitionPacket
├── version: str                        # "0.3"
├── header: Header                      # Identity, persona, routing
├── intent: Intent                      # Detected user goal + confidence
├── context: Context                    # History, cheatsheets, constraints
├── content: Content                    # Original prompt, data, attachments
├── reasoning: Reasoning                # Reflection logs, sketchpad, evaluations
├── response: Response                  # Candidate text, confidence, tool calls
├── governance: Governance              # Safety, audit, signatures, privacy
├── metrics: Metrics                    # Token usage, latency, resources
├── status: Status                      # State machine + observer trace
├── council: Optional[Council]          # Multi-agent voting
├── tool_routing: Optional[ToolRoutingState]  # GCP tool routing
├── loop_state: Optional[LoopState]     # Loop detection + recovery
└── goal_state: Optional[GoalState]     # Persistent user goal tracking
```

## Key Enums

- **PersonaRole**: DEFAULT, CODEMIND, ANALYST, GISEA, NARRATOR, OTHER
- **Origin**: USER, SYSTEM, AGENT, SIDECAR
- **TargetEngine**: PRIME, LITE, CODEMIND, COUNCIL
- **SystemTask**: INTENT_DETECTION, RESEARCH, GENERATE_DRAFT, REFINE, VALIDATE, DECISION, STREAM, TRIGGER_ACTION, TOOL_ROUTING, TOOL_EXECUTION
- **PacketState**: INITIALIZED → PROCESSING → AWAITING_TOOL → AWAITING_HUMAN → READY_TO_STREAM → COMPLETED | ABORTED
- **ToolExecutionStatus**: PENDING → AWAITING_CONFIDENCE → APPROVED → EXECUTED | FAILED | SKIPPED | USER_DENIED | AWAITING_APPROVAL
- **OutputDestination**: CLI, WEB, DISCORD, API, WEBHOOK, LOG, BROADCAST, AUDIO

## Tool Routing (GCP)

```
ToolRoutingState
├── needs_tool: bool                    # Intent detection flagged tool needed
├── routing_requested: bool             # Packet marked for tool routing loop
├── selected_tool: Optional[SelectedTool]
├── alternative_tools: List[SelectedTool]
├── review_confidence: float            # Prime review score
├── review_reasoning: str               # Approval/rejection justification
├── execution_status: ToolExecutionStatus
├── execution_result: Optional[ToolExecutionResult]
└── reinjection_count: int              # Max 3
```

## Output Routing

```
OutputRouting
├── primary: DestinationTarget          # Main destination
├── secondary: List[DestinationTarget]  # Additional destinations
├── suppress_echo: bool                 # Don't echo to origin
├── addressed_to_gaia: bool             # Was GAIA explicitly addressed?
└── source_destination: Optional[OutputDestination]
```

## Packet Lifecycle

1. **Created** by `packet_factory.build_packet()` — sets version, header, initial status
2. **Intent detection** populates `intent` (SystemTask + confidence)
3. **Knowledge enhancement** enriches `context` (RAG, history, cheatsheets)
4. **Cognitive dispatch** routes by SystemTask to handler
5. **Tool routing** (if needed) populates `tool_routing` with selection + execution
6. **Generation** calls Prime/Lite, populates `response`
7. **Output routing** determines destination(s) and formats response
8. **Status** moves to COMPLETED, `metrics` finalized

## Implementation Notes

- Uses `@dataclass_json` + `@dataclass` (NOT Pydantic) — for serialization compatibility
- Factory: `gaia_common.utils.packet_factory.build_packet()`
- All fields have sensible defaults — partial construction is safe
- Serialization: `.to_dict()` / `.from_dict()` via dataclasses_json

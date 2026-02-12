from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Literal
from enum import Enum
import json
import hashlib
from dataclasses_json import dataclass_json

# --- Enums for Type Safety ---
class PersonaRole(Enum):
    DEFAULT = "Default"
    CODEMIND = "CodeMind"
    ANALYST = "Analyst"
    GISEA = "GISEA"
    NARRATOR = "Narrator"
    OTHER = "Other"

class Origin(Enum):
    USER = "user"
    SYSTEM = "system"
    AGENT = "agent"
    SIDECAR = "sidecar"

class TargetEngine(Enum):
    PRIME = "Prime"
    LITE = "Lite"
    CODEMIND = "CodeMind"
    COUNCIL = "Council"

class SystemTask(Enum):
    INTENT_DETECTION = "IntentDetection"
    RESEARCH = "Research"
    GENERATE_DRAFT = "GenerateDraft"
    REFINE = "Refine"
    VALIDATE = "Validate"
    DECISION = "Decision"
    STREAM = "Stream"
    TRIGGER_ACTION = "TriggerAction"
    TOOL_ROUTING = "ToolRouting"          # Packet needs tool routing
    TOOL_EXECUTION = "ToolExecution"      # Tool approved, execute


class ToolExecutionStatus(Enum):
    """Status of tool execution within a cognition packet."""
    PENDING = "pending"                   # Tool selected but not yet executed
    AWAITING_CONFIDENCE = "awaiting_confidence"  # Waiting for confidence review
    APPROVED = "approved"                 # Confidence passed, ready to execute
    EXECUTED = "executed"                 # Tool successfully executed
    FAILED = "failed"                     # Tool execution failed
    SKIPPED = "skipped"                   # Low confidence, skipped
    USER_DENIED = "user_denied"           # User rejected tool use

class PacketState(Enum):
    INITIALIZED = "initialized"
    PROCESSING = "processing"
    AWAITING_TOOL = "awaiting_tool"
    AWAITING_HUMAN = "awaiting_human"
    READY_TO_STREAM = "ready_to_stream"
    COMPLETED = "completed"
    ABORTED = "aborted"

class OutputDestination(Enum):
    """
    Output destinations for the spinal column routing.
    Each destination represents a different interface where GAIA can send responses.
    """
    CLI = "cli"                    # Command-line interface (rescue shell, terminal)
    WEB = "web"                    # Web console / browser interface
    DISCORD = "discord"           # Discord channel
    API = "api"                    # REST API response
    WEBHOOK = "webhook"           # Outbound webhook
    LOG = "log"                    # Logging only (no user-facing output)
    BROADCAST = "broadcast"       # All registered destinations

# --- Header ---
@dataclass_json
@dataclass
class Persona:
    identity_id: str
    persona_id: str
    role: PersonaRole
    tone_hint: Optional[str] = None
    safety_profile_id: Optional[str] = None
    traits: Dict[str, Any] = field(default_factory=dict)

@dataclass_json
@dataclass
class Routing:
    target_engine: TargetEngine
    allow_parallel: bool = False
    priority: int = 5
    deadline_iso: Optional[str] = None
    queue_id: Optional[str] = None

@dataclass_json
@dataclass
class DestinationTarget:
    """
    Specifies a single output destination with optional channel/target info.
    Part of the spinal column routing system.
    """
    destination: OutputDestination
    channel_id: Optional[str] = None      # e.g., Discord channel ID, WebSocket session ID
    user_id: Optional[str] = None         # Target user (for DMs or specific routing)
    reply_to_message_id: Optional[str] = None  # For threaded replies (Discord, etc.)
    persona_override: Optional[str] = None     # Override persona for this destination
    format_hint: Optional[str] = None     # e.g., "markdown", "plain", "embed"
    metadata: Dict[str, Any] = field(default_factory=dict)  # Destination-specific extras

@dataclass_json
@dataclass
class OutputRouting:
    """
    Controls where the response should be delivered.
    Supports multi-destination routing from a single response.
    """
    primary: DestinationTarget                           # Main destination
    secondary: List[DestinationTarget] = field(default_factory=list)  # Additional destinations
    suppress_echo: bool = False                          # Don't echo to origin
    addressed_to_gaia: bool = True                       # Was GAIA explicitly addressed?
    source_destination: Optional[OutputDestination] = None  # Where the input came from

@dataclass_json
@dataclass
class Model:
    name: str
    provider: str
    context_window_tokens: int
    max_output_tokens: Optional[int] = None
    response_buffer_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None
    stop: List[str] = field(default_factory=list)
    tool_permissions: List[str] = field(default_factory=list)
    allow_tools: bool = True

@dataclass_json
@dataclass
class OperationalStatus:
    status: Optional[str] = None
    model: Optional[str] = None
    observer: Optional[str] = None
    confidence: Optional[float] = None
    intent: Optional[str] = None

@dataclass_json
@dataclass
class Header:
    datetime: str
    session_id: str
    packet_id: str
    sub_id: str
    persona: Persona
    origin: Origin
    routing: Routing
    model: Model
    parent_packet_id: Optional[str] = None
    lineage: List[str] = field(default_factory=list)
    output_routing: Optional[OutputRouting] = None  # Spinal column destination routing
    operational_status: Optional[OperationalStatus] = None # New field

# --- Intent ---
@dataclass_json
@dataclass
class Intent:
    user_intent: str
    system_task: SystemTask
    confidence: float
    tags: List[str] = field(default_factory=list)

# --- Context ---
@dataclass_json
@dataclass
class SessionHistoryRef:
    type: str
    value: str

@dataclass_json
@dataclass
class RelevantHistorySnippet:
    id: str
    role: str
    summary: str

@dataclass_json
@dataclass
class Cheatsheet:
    id: str
    title: str
    version: str
    pointer: str
    ref_type: Optional[str] = None

@dataclass_json
@dataclass
class Constraints:
    max_tokens: int
    time_budget_ms: int
    safety_mode: str
    policies: List[str] = field(default_factory=list)

@dataclass_json
@dataclass
class Context:
    session_history_ref: SessionHistoryRef
    cheatsheets: List[Cheatsheet]
    constraints: Constraints
    relevant_history_snippet: List[RelevantHistorySnippet] = field(default_factory=list)
    available_mcp_tools: Optional[List[str]] = None

# --- Content ---
@dataclass_json
@dataclass
class DataField:
    key: str
    value: Any
    type: Optional[str] = None
    source: Optional[str] = None
    vector_refs: List[str] = field(default_factory=list)

@dataclass_json
@dataclass
class Attachment:
    name: str
    mime: str
    content_hash: str
    bytes: Optional[int] = None
    location: Optional[str] = None

@dataclass_json
@dataclass
class Content:
    original_prompt: str
    data_fields: List[DataField] = field(default_factory=list)
    attachments: List[Attachment] = field(default_factory=list)

# --- Reasoning ---
@dataclass_json
@dataclass
class ReflectionLog:
    step: str
    summary: str
    confidence: Optional[float] = None

@dataclass_json
@dataclass
class Sketchpad:
    slot: str
    content: Any
    content_type: Optional[str] = None
    expires_at: Optional[str] = None

@dataclass_json
@dataclass
class ResponseFragment:
    """
    Represents a fragment of a response that exceeded token limits.
    Used for fragmentation/rehydration of long-form content.
    """
    fragment_id: str                      # UUID for this fragment
    parent_request_id: str                # Links fragments from same request
    sequence: int                         # 0, 1, 2, ... ordering
    content: str                          # The actual text content
    continuation_hint: str = ""           # Context for continuation (e.g., "The Raven stanza 10/18")
    is_complete: bool = False             # True for final fragment
    token_count: int = 0                  # Approximate tokens in this fragment
    created_at: Optional[str] = None      # ISO timestamp

@dataclass_json
@dataclass
class Evaluation:
    name: str
    passed: bool
    score: Optional[float] = None
    notes: Optional[str] = None

@dataclass_json
@dataclass
class Reasoning:
    reflection_log: List[ReflectionLog] = field(default_factory=list)
    sketchpad: List[Sketchpad] = field(default_factory=list)
    evaluations: List[Evaluation] = field(default_factory=list)

# --- Tool Routing (GCP Tool Routing System) ---
@dataclass_json
@dataclass
class SelectedTool:
    """Represents a tool selected for potential execution."""
    tool_name: str                                          # e.g., "ai.read", "ai.execute"
    params: Dict[str, Any] = field(default_factory=dict)
    selection_reasoning: str = ""                           # Why this tool was selected
    selection_confidence: float = 0.0                       # Model's confidence in selection


@dataclass_json
@dataclass
class ToolExecutionResult:
    """Result from executing an MCP tool."""
    success: bool
    output: Any = None                                      # Tool output (varies by tool)
    error: Optional[str] = None                             # Error message if failed
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
    needs_tool: bool = False                                # Intent detection decided tool needed
    routing_requested: bool = False                         # Packet marked for tool routing loop

    # Selection state
    selected_tool: Optional[SelectedTool] = None
    alternative_tools: List[SelectedTool] = field(default_factory=list)

    # Confidence review
    review_confidence: float = 0.0                          # Prime model's review confidence
    review_reasoning: str = ""                              # Prime model's review reasoning

    # Execution state
    execution_status: ToolExecutionStatus = ToolExecutionStatus.PENDING
    execution_result: Optional[ToolExecutionResult] = None

    # Reinjection tracking
    reinjection_count: int = 0                              # How many times this packet was reinjected
    max_reinjections: int = 3                               # Safety limit


# --- Response ---
@dataclass_json
@dataclass
class ToolCall:
    name: str
    args: Optional[Dict[str, Any]] = None
    dry_run: bool = True

@dataclass_json
@dataclass
class SidecarAction:
    action_type: str
    params: Optional[Dict[str, Any]] = None

@dataclass_json
@dataclass
class Response:
    candidate: str
    confidence: float
    stream_proposal: bool
    tool_calls: List[ToolCall] = field(default_factory=list)
    sidecar_actions: List[SidecarAction] = field(default_factory=list)

# --- Governance ---
@dataclass_json
@dataclass
class Safety:
    execution_allowed: bool
    allowed_commands_whitelist_id: Optional[str] = None
    dry_run: bool = True

@dataclass_json
@dataclass
class Signatures:
    header_hash: Optional[str] = None
    content_hash: Optional[str] = None
    signed_by: Optional[str] = None
    signature: Optional[str] = None

@dataclass_json
@dataclass
class Audit:
    created_by: Optional[str] = None
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    reviewers: List[str] = field(default_factory=list)
    decision: Optional[str] = None

@dataclass_json
@dataclass
class Privacy:
    pii_detected: Optional[bool] = None
    pii_notes: Optional[str] = None

@dataclass_json
@dataclass
class Governance:
    safety: Safety
    signatures: Signatures = field(default_factory=Signatures)
    audit: Audit = field(default_factory=Audit)
    privacy: Privacy = field(default_factory=Privacy)

# --- Loop Detection State ---
@dataclass_json
@dataclass
class LoopAttempt:
    """Record of a previous loop recovery attempt."""
    approach_summary: str
    failed_at: str

@dataclass_json
@dataclass
class LoopState:
    """Tracks loop detection and recovery state within a cognition packet.

    Captures information about detected loops for context preservation
    across resets and for injection into recovery prompts.
    """
    detected_at: Optional[str] = None
    loop_type: Optional[str] = None
    pattern: Optional[str] = None
    pattern_hash: Optional[str] = None
    reset_count: int = 0
    confidence: float = 0.0
    previous_attempts: List[LoopAttempt] = field(default_factory=list)
    recovery_context: Optional[str] = None
    triggered_by: List[str] = field(default_factory=list)
    in_recovery: bool = False
    warned: bool = False
    override_active: bool = False

# --- Council & Metrics & Status ---
@dataclass_json
@dataclass
class Vote:
    agent: str
    score: float
    rationale: Optional[str] = None

@dataclass_json
@dataclass
class Council:
    mode: Optional[str] = "solo"
    participants: List[str] = field(default_factory=list)
    votes: List[Vote] = field(default_factory=list)

@dataclass_json
@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    projected_tokens: Optional[int] = None

@dataclass_json
@dataclass
class SystemResources:
    """Represents a snapshot of system resource usage."""
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    gpu_usage: Dict[str, Any] = field(default_factory=dict)
    hardware_profile: Dict[str, Any] = field(default_factory=dict)

@dataclass_json
@dataclass
class Metrics:
    token_usage: TokenUsage
    latency_ms: int
    cost_estimate: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    resources: Optional[SystemResources] = None
    semantic_probe: Optional[Dict[str, Any]] = None


@dataclass_json
@dataclass
class Status:
    finalized: bool
    state: PacketState
    next_steps: List[str] = field(default_factory=list)
    observer_trace: List[str] = field(default_factory=list)

# --- Main Packet ---
@dataclass_json
@dataclass
class CognitionPacket:
    version: str
    header: Header
    intent: Intent
    context: Context
    content: Content
    reasoning: Reasoning
    response: Response
    governance: Governance
    metrics: Metrics
    status: Status
    schema_id: Optional[str] = None
    council: Optional[Council] = None
    tool_routing: Optional[ToolRoutingState] = None  # GCP Tool Routing System state
    loop_state: Optional[LoopState] = None            # Loop detection and recovery state

    def to_json(self, **kwargs) -> str:
        """Serializes the packet to a JSON string with sorted keys for stability."""
        # The to_json method from dataclasses_json.DataClassJsonMixin doesn't directly accept sort_keys.
        # We need to dump to a dict first, then use json.dumps.
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def compute_hashes(self):
        """Computes and sets the header and content hashes for integrity checks."""
        # Exclude the signatures themselves from the hash calculation
        header_dict = self.header.to_dict()
        content_dict = self.content.to_dict()
        # dataclasses_json may leave Enum instances in the dict; convert them to primitives
        def _normalize(obj):
            from enum import Enum
            if isinstance(obj, dict):
                return {k: _normalize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_normalize(v) for v in obj]
            if isinstance(obj, Enum):
                return obj.value
            return obj

        header_str = json.dumps(_normalize(header_dict), sort_keys=True)
        content_str = json.dumps(_normalize(content_dict), sort_keys=True)
        
        self.governance.signatures.header_hash = hashlib.sha256(header_str.encode('utf-8')).hexdigest()
        self.governance.signatures.content_hash = hashlib.sha256(content_str.encode('utf-8')).hexdigest()

    def to_serializable_dict(self) -> Dict:
        """Return a version of the packet as a plain dict with enums converted to primitive values.

        Useful for logging, telemetry, and JSON dumps where Enum instances would otherwise fail.
        """
        def _normalize(obj):
            from enum import Enum
            if isinstance(obj, dict):
                return {k: _normalize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_normalize(v) for v in obj]
            if isinstance(obj, Enum):
                return obj.value
            return obj

        return _normalize(self.to_dict())

    def validate(self, schema: Dict):
        """Validates the packet against the formal JSON schema."""
        from jsonschema import validate
        validate(instance=self.to_dict(), schema=schema)

    def check_token_budget(self) -> bool:
        """Checks if projected tokens are within the model's response buffer."""
        if self.metrics.token_usage.projected_tokens and self.header.model.response_buffer_tokens and self.header.model.max_output_tokens:
            if self.metrics.token_usage.projected_tokens > self.header.model.max_output_tokens - self.header.model.response_buffer_tokens:
                return False
        return True

__all__ = [
    # Main packet
    "CognitionPacket",
    # Enums
    "PersonaRole",
    "Origin",
    "TargetEngine",
    "SystemTask",
    "ToolExecutionStatus",
    "PacketState",
    "OutputDestination",
    # Header components
    "Persona",
    "Routing",
    "DestinationTarget",
    "OutputRouting",
    "Model",
    "OperationalStatus",
    "Header",
    # Intent
    "Intent",
    # Context
    "SessionHistoryRef",
    "RelevantHistorySnippet",
    "Cheatsheet",
    "Constraints",
    "Context",
    # Content
    "DataField",
    "Attachment",
    "Content",
    # Reasoning
    "ReflectionLog",
    "Sketchpad",
    "ResponseFragment",
    "Evaluation",
    "Reasoning",
    # Tool routing
    "SelectedTool",
    "ToolExecutionResult",
    "ToolRoutingState",
    # Response
    "ToolCall",
    "SidecarAction",
    "Response",
    # Governance
    "Safety",
    "Signatures",
    "Audit",
    "Privacy",
    "Governance",
    # Council, Metrics, Status
    "Vote",
    "Council",
    "TokenUsage",
    "SystemResources",
    "Metrics",
    "Status",
    # Loop Detection
    "LoopAttempt",
    "LoopState",
]

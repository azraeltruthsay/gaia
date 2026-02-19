"""
gaia_common/models/blueprint.py

Blueprint schema for GAIA service self-modelling.

Every service in GAIA is described by a BlueprintModel. Blueprints exist in
two distinct epistemic states:

  CANDIDATE  — prescriptive (what we intend to build) or unvalidated (just
               discovered / first reflection). Lives in
               knowledge/blueprints/candidates/{service_id}.yaml.
               Never rendered in the live graph.

  LIVE       — descriptive (what actually exists, validated by the promotion
               pipeline and at least one reflection cycle). Lives in
               knowledge/blueprints/{service_id}.yaml.
               The graph renders exclusively from live blueprints.

Two generation modes exist:

  DISCOVERY  — cold-start analysis of a service with no prior blueprint.
               Produces genesis=True output with per-section confidence scores.
               Triggered by the promotion pipeline or the builder panel.

  REFLECTION — warm update against an existing blueprint during a sleep cycle.
               Produces a diff-shaped output. Clears genesis=True once the
               first successful reflection validates the model against reality.

The schema is the lingua franca for:
  - gaia-study   (writer — discovery & reflection)
  - gaia-web     (reader — graph topology API, markdown rendering)
  - builder panel (seed generator — prescriptive candidate blueprints)
  - promote_candidate.sh (gate — candidate blueprint required for promotion)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


# ── Enumerations ─────────────────────────────────────────────────────────────


class BlueprintStatus(str, Enum):
    """Lifecycle state of the blueprint itself (not the service)."""
    CANDIDATE = "candidate"   # unvalidated — prescriptive or freshly discovered
    LIVE = "live"             # validated — descriptive, rendered in graph
    ARCHIVED = "archived"     # service retired


class GeneratedBy(str, Enum):
    """What process created or last updated this blueprint."""
    MANUAL_SEED = "manual_seed"       # hand-authored bootstrap
    BUILDER_SEED = "builder_seed"     # visual graph designer session
    DISCOVERY = "discovery"           # cold-start code analysis by study
    REFLECTION = "reflection"         # warm sleep-cycle update by study
    GAIA_INITIATED = "gaia_initiated" # GAIA identified a capability gap herself


class ServiceStatus(str, Enum):
    """Runtime status of the service described by this blueprint."""
    LIVE = "live"
    CANDIDATE = "candidate"
    PLANNED = "planned"
    DEPRECATED = "deprecated"


class Severity(str, Enum):
    DEGRADED = "degraded"   # service continues in reduced capacity
    PARTIAL = "partial"     # some requests fail, others succeed
    FATAL = "fatal"         # service cannot function at all


class ConfidenceLevel(str, Enum):
    HIGH = "high"     # directly evidenced in code / config
    MEDIUM = "medium" # inferred from patterns, mostly reliable
    LOW = "low"       # speculative — requires human review


class InterfaceDirection(str, Enum):
    INBOUND = "inbound"   # this service receives on this interface
    OUTBOUND = "outbound" # this service sends on this interface


class InterfaceStatus(str, Enum):
    ACTIVE = "active"
    PLANNED = "planned"
    DEPRECATED = "deprecated"


class TransportType(str, Enum):
    HTTP_REST = "http_rest"
    WEBSOCKET = "websocket"
    EVENT = "event"           # async event bus (topic-based)
    DIRECT_CALL = "direct_call"  # in-process / library mode
    MCP = "mcp"               # JSON-RPC tool protocol
    GRPC = "grpc"
    SSE = "sse"               # server-sent events (streaming)


class VolumeAccess(str, Enum):
    RO = "ro"
    RW = "rw"


# ── Interface models ──────────────────────────────────────────────────────────


class HttpRestInterface(BaseModel):
    """Synchronous HTTP REST endpoint."""
    type: Literal[TransportType.HTTP_REST] = TransportType.HTTP_REST
    path: str
    method: str = "POST"
    input_schema: Optional[str] = None   # schema name e.g. "CognitionPacket"
    output_schema: Optional[str] = None


class WebSocketInterface(BaseModel):
    """Persistent bidirectional WebSocket channel."""
    type: Literal[TransportType.WEBSOCKET] = TransportType.WEBSOCKET
    path: str
    protocol: Optional[str] = None  # sub-protocol if any


class SseInterface(BaseModel):
    """Server-sent events streaming endpoint."""
    type: Literal[TransportType.SSE] = TransportType.SSE
    path: str
    event_types: List[str] = Field(default_factory=list)


class EventInterface(BaseModel):
    """Async event bus interface (topic-based publish/subscribe)."""
    type: Literal[TransportType.EVENT] = TransportType.EVENT
    topic: str
    payload_schema: Optional[str] = None


class DirectCallInterface(BaseModel):
    """In-process function call — for embedded / library-mode deployment."""
    type: Literal[TransportType.DIRECT_CALL] = TransportType.DIRECT_CALL
    symbol: str  # e.g. "gaia_core.api.process_packet"


class McpInterface(BaseModel):
    """JSON-RPC tool protocol (MCP) dispatch."""
    type: Literal[TransportType.MCP] = TransportType.MCP
    target_service: Optional[str] = None  # outbound: which service receives
    methods: List[str] = Field(default_factory=list)


class GrpcInterface(BaseModel):
    """gRPC typed RPC."""
    type: Literal[TransportType.GRPC] = TransportType.GRPC
    proto: str        # relative path to .proto file
    rpc: str          # RPC method name
    service: Optional[str] = None  # protobuf service name


# Transport discriminated union — add new transports here as they are introduced
InterfaceTransport = Union[
    HttpRestInterface,
    WebSocketInterface,
    SseInterface,
    EventInterface,
    DirectCallInterface,
    McpInterface,
    GrpcInterface,
]


class NegotiatedTransport(BaseModel):
    """
    Multiple transports for the same logical interface, with a preference order.
    Lets the schema capture upgrade paths without requiring manual annotation.

    Example: REST today, gRPC when ready — the graph shows "upgrade available".
    """
    transports: List[InterfaceTransport] = Field(min_length=2)
    preferred: TransportType
    upgrade_note: Optional[str] = None


class Interface(BaseModel):
    """
    A single named interface exposed or consumed by this service.

    The graph derives edges from interface pairs:
      edge exists if  A.outbound.topic/path == B.inbound.topic/path
                      AND transport types are compatible

    Edges are never stored — always derived at render time. This means
    adding a new service with matching interfaces automatically wires it
    into the graph without any manual topology definition.
    """
    id: str = Field(description="Stable identifier for this interface within the service")
    direction: InterfaceDirection
    transport: Union[InterfaceTransport, NegotiatedTransport] = Field(
        discriminator=None  # handled by Union — Pydantic v2 resolves by shape
    )
    description: str
    status: InterfaceStatus = InterfaceStatus.ACTIVE


# ── Dependency models ─────────────────────────────────────────────────────────


class ServiceDependency(BaseModel):
    """Another GAIA service this service calls or receives from."""
    id: str                          # service identifier e.g. "gaia-prime"
    role: str                        # how it's used e.g. "inference"
    required: bool = True            # False = graceful degradation available
    fallback: Optional[str] = None   # e.g. "groq-api", "gguf-local"


class VolumeDependency(BaseModel):
    name: str
    access: VolumeAccess
    purpose: Optional[str] = None
    mount_path: Optional[str] = None  # e.g. "/shared", "/knowledge"


class ExternalApiDependency(BaseModel):
    name: str            # e.g. "groq", "openai"
    purpose: str
    required: bool = False


class Dependencies(BaseModel):
    services: List[ServiceDependency] = Field(default_factory=list)
    volumes: List[VolumeDependency] = Field(default_factory=list)
    external_apis: List[ExternalApiDependency] = Field(default_factory=list)


# ── Source file models ────────────────────────────────────────────────────────


class SourceFile(BaseModel):
    """
    A source file that implements part of this service.

    The role field is what makes this more than a file list — it lets the LLM
    in a builder session say "load the core_logic file" rather than guessing
    which file matters. Also lets the freshness checker understand *why* a
    file is referenced, not just whether it exists.
    """
    path: str   # relative to GAIA_ROOT e.g. "candidates/gaia-core/gaia_core/main.py"
    role: str   # entrypoint | core_logic | tool_routing | config | test | proto | other
    file_type: Optional[str] = None  # python | dockerfile | shell | yaml | json | other


# ── Internal architecture models ─────────────────────────────────────────────


class InternalComponent(BaseModel):
    """
    A logical module or subsystem within a service.

    Groups related source files, maps to external interfaces, and lists
    key classes/functions so the architecture graph is self-documenting
    without needing to read the source files directly.
    """
    id: str                                              # e.g. "cognition_engine"
    label: str                                           # Human-readable "Cognition Engine"
    description: str                                     # What this component does
    source_files: List[str] = Field(default_factory=list)  # refs to SourceFile.path values
    key_classes: List[str] = Field(default_factory=list)   # e.g. ["CognitiveDispatcher"]
    key_functions: List[str] = Field(default_factory=list) # e.g. ["dispatch()", "detect_goal()"]
    exposes_interfaces: List[str] = Field(default_factory=list)  # Interface.id refs this component implements
    consumes_interfaces: List[str] = Field(default_factory=list) # Interface.id refs this component calls


class InternalEdge(BaseModel):
    """
    A wiring connection between two internal components.

    Captures data flow within a service — how modules call each other,
    what data they pass, and the transport mechanism (usually function calls).
    """
    from_component: str       # InternalComponent.id
    to_component: str         # InternalComponent.id
    label: str                # e.g. "dispatches packets to"
    transport: str = "function_call"  # function_call | event | queue | import
    data_flow: Optional[str] = None   # e.g. "CognitionPacket", "session_id"


class ServiceArchitecture(BaseModel):
    """
    Internal architecture of a service — components and their wiring.

    This is the 'zoom in' layer: the service graph shows boxes connected
    by edges; clicking a box reveals this internal component graph with
    its own nodes and edges. Together they provide a complete structural
    map from system level down to module level.
    """
    components: List[InternalComponent] = Field(default_factory=list)
    edges: List[InternalEdge] = Field(default_factory=list)


# ── Failure mode models ───────────────────────────────────────────────────────


class FailureMode(BaseModel):
    """
    A known failure condition and how this service responds to it.

    This is the n8n node-level error isolation principle made explicit:
    a service failing doesn't equal the system failing. The graph can render
    failure modes as amber edges rather than binary connected/disconnected.
    """
    condition: str        # e.g. "gaia-prime unavailable"
    response: str         # e.g. "fallback to Groq API"
    severity: Severity
    auto_recovers: bool = True


# ── Runtime models ────────────────────────────────────────────────────────────


class SecurityConfig(BaseModel):
    no_new_privileges: bool = False
    cap_drop: List[str] = Field(default_factory=list)
    cap_add: List[str] = Field(default_factory=list)


class Runtime(BaseModel):
    port: Optional[int] = None
    base_image: Optional[str] = None
    gpu: bool = False
    startup_cmd: Optional[str] = None
    health_check: Optional[str] = None
    gpu_count: Optional[Union[int, Literal["all"]]] = None
    user: Optional[str] = None
    dockerfile: Optional[str] = None
    compose_service: Optional[str] = None
    security: Optional[SecurityConfig] = None


# ── Intent models (sticky notes) ─────────────────────────────────────────────


class Intent(BaseModel):
    """
    Design rationale layer — the 'sticky notes' of the blueprint.

    Initially human-authored or seeded by the builder panel. Over reflection
    cycles, study enriches this layer by inferring intent from code patterns,
    comments, and observed behaviour.

    open_questions is where GAIA surfaces her own uncertainty about her design.
    These are not errors — they are honest acknowledgements of incomplete
    self-knowledge. The dashboard renders them as review prompts.
    """
    purpose: str
    design_decisions: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(
        default_factory=list,
        description="Unresolved questions surfaced during reflection. "
                    "Cleared when answered via human review or subsequent reflection."
    )
    cognitive_role: Optional[str] = None  # e.g. "The Brain", "The Voice"


# ── Confidence models ─────────────────────────────────────────────────────────


class SectionConfidence(BaseModel):
    """
    Per-section confidence scores for freshly discovered blueprints.

    HIGH   = directly evidenced (port in config, route decorator in code)
    MEDIUM = inferred from patterns (dependency implied by import)
    LOW    = speculative (intent inferred from variable names / comments)

    Scores upgrade over reflection cycles as evidence accumulates.
    They never downgrade automatically — only a human review or a reflection
    that finds contradicting evidence can lower a score.
    """
    runtime: ConfidenceLevel = ConfidenceLevel.HIGH
    contract: ConfidenceLevel = ConfidenceLevel.HIGH
    dependencies: ConfidenceLevel = ConfidenceLevel.MEDIUM
    failure_modes: ConfidenceLevel = ConfidenceLevel.MEDIUM
    intent: ConfidenceLevel = ConfidenceLevel.LOW


# ── Meta model ────────────────────────────────────────────────────────────────


class BlueprintMeta(BaseModel):
    """
    Metadata about the blueprint itself — not the service it describes.

    This is the audit trail for GAIA's self-knowledge:
      - genesis=True means "first impression, not yet validated by experience"
      - generated_by tells you whether to trust it and how much
      - last_reflected timestamps when study last checked this against reality
      - confidence tracks per-section reliability

    The genesis flag is cleared by the first successful REFLECTION cycle that
    finds the blueprint accurate. That transition is a meaningful cognitive
    event logged in the dev journal.
    """
    status: BlueprintStatus = BlueprintStatus.CANDIDATE
    genesis: bool = Field(
        default=True,
        description="True until a reflection cycle validates this blueprint against "
                    "real service behaviour. Cleared automatically on first successful reflection."
    )
    generated_by: GeneratedBy
    blueprint_version: str = "0.1"
    schema_version: str = "1.0"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_reflected: Optional[datetime] = None
    promoted_at: Optional[datetime] = None
    confidence: SectionConfidence = Field(default_factory=SectionConfidence)

    # Diff tracking — populated when this blueprint was produced by REFLECTION
    # against a prior version. Empty for DISCOVERY / MANUAL_SEED.
    reflection_notes: Optional[str] = Field(
        default=None,
        description="Natural language summary of what changed in the last reflection "
                    "cycle. Written to prime.md checkpoint on wake."
    )
    divergence_score: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description="For promoted services: diff score between candidate blueprint "
                    "(prescriptive) and live blueprint (descriptive). "
                    "LOW = faithful implementation. HIGH = flagged for review."
    )


# ── Top-level BlueprintModel ──────────────────────────────────────────────────


class BlueprintModel(BaseModel):
    """
    Canonical self-model for a single GAIA service.

    Two epistemic variants share this schema:

    CANDIDATE blueprints (prescriptive)
      - Live in knowledge/blueprints/candidates/{service_id}.yaml
      - Describe intended behaviour (builder panel output)
      - Never rendered in the live graph
      - Required gate for promotion pipeline

    LIVE blueprints (descriptive)
      - Live in knowledge/blueprints/{service_id}.yaml
      - Describe actual behaviour (study discovery / reflection output)
      - Graph renders exclusively from these
      - Markdown generated from this YAML — never hand-authored

    Graph topology is derived, not stored:
      edge exists between A and B if A has an outbound interface whose
      topic/path matches B's inbound interface of compatible transport type.
      Edges self-assemble when new services are added with matching interfaces.
    """

    # Identity
    id: str = Field(description="Canonical service identifier e.g. 'gaia-core'")
    version: str = Field(description="Service version this blueprint describes")
    role: str = Field(description="Human-readable label for graph node e.g. 'The Brain'")
    service_status: ServiceStatus = ServiceStatus.LIVE

    # Core sections
    runtime: Runtime = Field(default_factory=Runtime)
    interfaces: List[Interface] = Field(
        default_factory=list,
        description="All inbound and outbound interfaces. Graph edges derived from these."
    )
    dependencies: Dependencies = Field(default_factory=Dependencies)
    source_files: List[SourceFile] = Field(
        default_factory=list,
        description="Source files with roles. Used by freshness checker and LLM builder sessions."
    )
    failure_modes: List[FailureMode] = Field(default_factory=list)
    intent: Optional[Intent] = None
    architecture: Optional[ServiceArchitecture] = Field(
        default=None,
        description="Internal component graph — modules, wiring, key classes/functions. "
                    "The 'zoom in' layer rendered when clicking a service node."
    )

    # Blueprint metadata
    meta: BlueprintMeta

    model_config = {"use_enum_values": False}  # keep enum objects for type safety

    @model_validator(mode="after")
    def validate_live_blueprint_has_intent(self) -> "BlueprintModel":
        """
        Live blueprints should have an intent section — it's the one section
        that can't be mechanically derived and requires cognitive effort to produce.
        Warn (don't fail) if missing, since discovery may produce it on first pass.
        """
        if (
            self.meta.status == BlueprintStatus.LIVE
            and not self.meta.genesis
            and self.intent is None
        ):
            # Non-fatal: live blueprint exists but intent not yet populated.
            # Reflection cycle should fill this in. Surface as open question.
            pass
        return self

    @model_validator(mode="after")
    def validate_interface_ids_unique(self) -> "BlueprintModel":
        ids = [i.id for i in self.interfaces]
        if len(ids) != len(set(ids)):
            duplicates = [i for i in ids if ids.count(i) > 1]
            raise ValueError(f"Interface ids must be unique within a service. Duplicates: {duplicates}")
        return self

    def inbound_interfaces(self) -> List[Interface]:
        return [i for i in self.interfaces if i.direction == InterfaceDirection.INBOUND]

    def outbound_interfaces(self) -> List[Interface]:
        return [i for i in self.interfaces if i.direction == InterfaceDirection.OUTBOUND]

    def active_interfaces(self) -> List[Interface]:
        return [i for i in self.interfaces if i.status == InterfaceStatus.ACTIVE]

    def open_questions(self) -> List[str]:
        """Convenience accessor for dashboard review queue."""
        return self.intent.open_questions if self.intent else []

    def is_validated(self) -> bool:
        """True once genesis is cleared — blueprint has been tested against reality."""
        return not self.meta.genesis

    def to_graph_node(self) -> Dict[str, Any]:
        """
        Minimal representation for graph topology API.
        Full blueprint fetched separately on node click.
        """
        return {
            "id": self.id,
            "role": self.role,
            "service_status": self.service_status.value,
            "blueprint_status": self.meta.status.value,
            "genesis": self.meta.genesis,
            "port": self.runtime.port,
            "gpu": self.runtime.gpu,
            "open_question_count": len(self.open_questions()),
            "interface_count": len(self.interfaces),
            "confidence": {
                k: v.value if isinstance(v, ConfidenceLevel) else v
                for k, v in self.meta.confidence.model_dump().items()
            },
        }


# ── Graph topology models (derived, never stored) ─────────────────────────────


class GraphEdge(BaseModel):
    """
    A derived connection between two services.
    Never persisted — computed at render time from blueprint interfaces.

    Visual rendering hints:
      transport_type → edge style  (REST=solid, WebSocket=double, Event=dashed,
                                    SSE=animated, MCP=dotted)
      status         → edge colour (active=blue, planned=grey, deprecated=red)
      has_fallback   → edge weight (thicker if degraded-but-alive path exists)
    """
    from_service: str
    to_service: str
    interface_id_from: str
    interface_id_to: str
    transport_type: TransportType
    status: InterfaceStatus
    description: str
    has_fallback: bool = False  # True if source dependency has required=False


class GraphTopology(BaseModel):
    """
    Full derived graph — nodes from live blueprints, edges from interface matching.
    Returned by GET /api/blueprints/graph.
    Recomputed on every request — always reflects current blueprint state.
    """
    nodes: List[Dict[str, Any]]  # BlueprintModel.to_graph_node() outputs
    edges: List[GraphEdge]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    blueprint_count: int = 0
    pending_review_count: int = 0  # genesis=True blueprints awaiting validation

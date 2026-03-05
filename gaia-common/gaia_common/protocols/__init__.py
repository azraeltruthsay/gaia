"""
Protocol definitions for GAIA services.

This module contains the core data structures that flow between services:
- CognitionPacket: The unified thought representation (v0.3)
- GCP: Generative Council Protocol message schemas (TODO)
"""

from .cognition_packet import (
    # Main packet
    CognitionPacket,
    # Enums
    PersonaRole,
    Origin,
    TargetEngine,
    SystemTask,
    ToolExecutionStatus,
    PacketState,
    OutputDestination,
    # Header components
    Persona,
    Routing,
    DestinationTarget,
    OutputRouting,
    Model,
    Header,
    # Intent
    Intent,
    # Context
    SessionHistoryRef,
    RelevantHistorySnippet,
    Cheatsheet,
    Constraints,
    Context,
    # Content
    DataField,
    Attachment,
    Content,
    # Reasoning
    ReflectionLog,
    Sketchpad,
    ResponseFragment,
    Evaluation,
    Reasoning,
    # Tool routing
    SelectedTool,
    ToolExecutionResult,
    ToolRoutingState,
    # Response
    ToolCall,
    SidecarAction,
    Response,
    # Governance
    Safety,
    Signatures,
    Audit,
    Privacy,
    Governance,
    # Council, Metrics, Status
    Vote,
    Council,
    TokenUsage,
    SystemResources,
    Metrics,
    Status,
    # Loop Detection
    LoopAttempt,
    LoopState,
    # Goal Detection
    GoalConfidence,
    DetectedGoal,
    GoalState,
)

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
    # Goal Detection
    "GoalConfidence",
    "DetectedGoal",
    "GoalState",
]

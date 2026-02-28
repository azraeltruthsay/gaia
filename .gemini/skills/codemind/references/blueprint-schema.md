# BlueprintModel Schema Reference

> Source: `gaia-common/gaia_common/models/blueprint.py`

## Top-Level Structure

```python
class BlueprintModel(BaseModel):
    id: str                               # "gaia-core", "gaia-web"
    version: str                          # Service version
    role: str                             # "The Brain", "The Face"
    service_status: ServiceStatus         # LIVE | CANDIDATE | PLANNED | DEPRECATED

    runtime: Runtime                      # Port, image, GPU, health_check, startup_cmd
    interfaces: List[Interface]           # Inbound/outbound contracts (graph edges derived from these)
    dependencies: Dependencies            # Service, volume, external API deps
    source_files: List[SourceFile]        # Files with roles (entrypoint, core_logic, test)
    failure_modes: List[FailureMode]      # Known failures + recovery strategies
    intent: Optional[Intent]              # Design rationale, cognitive role, open questions
    architecture: Optional[ServiceArchitecture]  # Internal component graph

    meta: BlueprintMeta                   # Audit trail: status, genesis, timestamps, confidence
```

## Epistemic States

| Status | Meaning | Location |
|--------|---------|----------|
| CANDIDATE | Prescriptive (intent) or freshly discovered (unvalidated) | `knowledge/blueprints/candidates/{service_id}.yaml` |
| LIVE | Descriptive (actual behavior, validated) | `knowledge/blueprints/{service_id}.yaml` |

LIVE blueprints are the **only** source for graph rendering. Candidates never appear in the graph.

## Interface Model (Discriminated Unions)

```python
class Interface(BaseModel):
    id: str                       # "rest_process", "jsonrpc_tools"
    direction: InterfaceDirection  # INBOUND | OUTBOUND
    transport: InterfaceTransport  # Discriminated union (see below)
    description: str
    status: InterfaceStatus       # ACTIVE | PLANNED | DEPRECATED
```

Transport types: `HttpRestInterface`, `WebSocketInterface`, `SseInterface`, `EventInterface`, `DirectCallInterface`, `McpInterface`, `GrpcInterface` — discriminated by `type` field matching `TransportType` enum.

## Internal Architecture

```python
class InternalComponent(BaseModel):
    id: str                           # "cognition_engine"
    label: str                        # "Cognition Engine"
    description: str
    source_files: List[str]           # Paths relative to GAIA_ROOT
    key_classes: List[str]
    key_functions: List[str]
    exposes_interfaces: List[str]     # Interface.id values implemented
    consumes_interfaces: List[str]    # Interface.id values called

class InternalEdge(BaseModel):
    from_component: str
    to_component: str
    label: str                        # "dispatches packets to"
    transport: str = "function_call"
    data_flow: Optional[str]          # "CognitionPacket", "session_id"
```

## Metadata & Confidence

```python
class BlueprintMeta(BaseModel):
    status: BlueprintStatus = CANDIDATE
    genesis: bool = True              # Cleared after first reflection cycle
    generated_by: GeneratedBy         # MANUAL_SEED | BUILDER_SEED | DISCOVERY | REFLECTION | GAIA_INITIATED
    confidence: SectionConfidence     # Per-section HIGH/MEDIUM/LOW scores
    divergence_score: Optional[float] # 0.0–1.0 (LOW=faithful, HIGH=needs review)
    reflection_notes: Optional[str]
```

Confidence levels: HIGH (directly evidenced), MEDIUM (inferred from patterns), LOW (speculative).

## Key Enums

- `ServiceStatus`: LIVE, CANDIDATE, PLANNED, DEPRECATED
- `BlueprintStatus`: CANDIDATE, LIVE, ARCHIVED
- `TransportType`: HTTP_REST, WEBSOCKET, SSE, EVENT, DIRECT_CALL, MCP, GRPC
- `InterfaceDirection`: INBOUND, OUTBOUND
- `InterfaceStatus`: ACTIVE, PLANNED, DEPRECATED
- `GeneratedBy`: MANUAL_SEED, BUILDER_SEED, DISCOVERY, REFLECTION, GAIA_INITIATED
- `ConfidenceLevel`: HIGH, MEDIUM, LOW
- `Severity`: DEGRADED, PARTIAL, FATAL

## Helpers

- `blueprint.inbound_interfaces()` → list of inbound interfaces
- `blueprint.outbound_interfaces()` → list of outbound interfaces
- `blueprint.active_interfaces()` → list of ACTIVE interfaces
- `blueprint.open_questions()` → list of unresolved questions from intent section
- `blueprint.is_validated()` → `True` if `genesis=False`

## Graph Derivation (Not Stored)

Graph edges are derived at render time, not persisted:
- If service A has outbound interface matching service B's inbound interface (path + transport), an edge exists
- `has_fallback: True` if dependency is `required=False`

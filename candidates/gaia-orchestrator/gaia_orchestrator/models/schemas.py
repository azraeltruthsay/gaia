"""
Pydantic models for the GAIA Orchestrator API.

Defines request/response schemas for GPU management, container lifecycle,
handoff protocol, and notifications.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
import uuid


# =============================================================================
# GPU Management Models
# =============================================================================

class GPUOwner(str, Enum):
    """Valid GPU owners in the GAIA ecosystem."""
    NONE = "none"
    CORE = "gaia-core"
    CORE_CANDIDATE = "gaia-core-candidate"
    STUDY = "gaia-study"
    STUDY_CANDIDATE = "gaia-study-candidate"


class GPUAcquireRequest(BaseModel):
    """Request to acquire GPU ownership."""
    requester: GPUOwner = Field(..., description="Service requesting GPU")
    reason: str = Field(..., description="Why GPU is needed")
    timeout_seconds: int = Field(default=300, description="Max wait time if queued")
    priority: int = Field(default=0, description="Higher = more urgent")


class GPUAcquireResponse(BaseModel):
    """Response to GPU acquire request."""
    success: bool
    lease_id: Optional[str] = None
    message: str
    queue_position: Optional[int] = None


class GPUMemoryInfo(BaseModel):
    """GPU memory status."""
    total_mb: int
    used_mb: int
    free_mb: int


class GPUStatus(BaseModel):
    """Current GPU ownership and state."""
    owner: GPUOwner = GPUOwner.NONE
    lease_id: Optional[str] = None
    reason: Optional[str] = None
    acquired_at: Optional[datetime] = None
    memory: Optional[GPUMemoryInfo] = None
    queue: List[str] = Field(default_factory=list, description="Waiting requesters")


# =============================================================================
# Container Lifecycle Models
# =============================================================================

class ContainerState(str, Enum):
    """Possible container states."""
    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    STOPPING = "stopping"
    ERROR = "error"
    UNKNOWN = "unknown"


class ServiceHealth(BaseModel):
    """Health status of a single service."""
    name: str
    state: ContainerState
    port: int
    healthy: bool = False
    last_check: Optional[datetime] = None


class ContainerStatus(BaseModel):
    """Status of all containers in a stack."""
    live: Dict[str, ServiceHealth] = Field(default_factory=dict)
    candidate: Dict[str, ServiceHealth] = Field(default_factory=dict)


class ContainerStartRequest(BaseModel):
    """Request to start containers."""
    gpu_enabled: bool = Field(default=False, description="Request GPU for this stack")
    services: Optional[List[str]] = Field(default=None, description="Specific services, or all if None")


class ContainerSwapRequest(BaseModel):
    """Request to swap a service from live to candidate."""
    service: str = Field(..., description="Service to swap: core, mcp, study, web")
    target: str = Field(default="candidate", description="Target: 'candidate' or 'live'")


# =============================================================================
# Handoff Protocol Models
# =============================================================================

class HandoffPhase(str, Enum):
    """Phases in a GPU handoff."""
    INITIATED = "initiated"
    RELEASING_GPU = "releasing_gpu"
    WAITING_CUDA_CLEANUP = "waiting_cuda_cleanup"
    TRANSFERRING_OWNERSHIP = "transferring_ownership"
    SIGNALING_RECIPIENT = "signaling_recipient"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HandoffType(str, Enum):
    """Types of handoff operations."""
    PRIME_TO_STUDY = "prime_to_study"
    STUDY_TO_PRIME = "study_to_prime"
    LIVE_TO_CANDIDATE = "live_to_candidate"
    CANDIDATE_TO_LIVE = "candidate_to_live"


class HandoffRequest(BaseModel):
    """Request to initiate a GPU handoff."""
    handoff_type: HandoffType
    reason: str = Field(default="", description="Optional reason for handoff")
    timeout_seconds: int = Field(default=120, description="Max time for handoff to complete")


class HandoffStatus(BaseModel):
    """Status of a handoff operation."""
    handoff_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    handoff_type: HandoffType
    phase: HandoffPhase = HandoffPhase.INITIATED
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    source: GPUOwner
    destination: GPUOwner
    error: Optional[str] = None
    progress_pct: int = 0


# =============================================================================
# Notification Models
# =============================================================================

class NotificationType(str, Enum):
    """Types of notifications."""
    ORACLE_FALLBACK = "oracle_fallback"
    GPU_RELEASED = "gpu_released"
    GPU_ACQUIRED = "gpu_acquired"
    HANDOFF_STARTED = "handoff_started"
    HANDOFF_COMPLETED = "handoff_completed"
    HANDOFF_FAILED = "handoff_failed"
    SERVICE_ERROR = "service_error"


class OracleNotification(BaseModel):
    """Notification when Oracle fallback is used."""
    fallback_model: str = Field(..., description="Model being used as fallback")
    original_role: str = Field(..., description="Role that needed the model")
    reason: str = Field(default="", description="Why fallback was needed")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Notification(BaseModel):
    """Generic notification message."""
    notification_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    notification_type: NotificationType
    title: str
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# State Persistence Models
# =============================================================================

class OrchestratorState(BaseModel):
    """Complete orchestrator state for persistence."""
    gpu: GPUStatus = Field(default_factory=GPUStatus)
    containers: ContainerStatus = Field(default_factory=ContainerStatus)
    active_handoff: Optional[HandoffStatus] = None
    handoff_history: List[HandoffStatus] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

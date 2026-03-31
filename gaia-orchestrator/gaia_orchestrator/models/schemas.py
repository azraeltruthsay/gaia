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
    AUDIO = "gaia-audio"


class GPUState(str, Enum):
    """GPU ownership state for the watch rotation protocol."""
    IDLE = "idle"                   # Core+Nano own GPU. Prime sleeping. SAE/ROME work possible.
    FOCUSING = "focusing"           # Prime owns GPU. Core+Nano on CPU fallback.
    TRANSITIONING = "transitioning" # Handoff in progress. No inference requests accepted.


class TierDevice(str, Enum):
    """Current device for a cognitive tier."""
    GPU_SAFETENSORS = "gpu_safetensors"   # Safetensors loaded on GPU (full capability)
    GPU_GGUF = "gpu_gguf"                 # GGUF on GPU via llama-server
    CPU_GGUF = "cpu_gguf"                 # GGUF on CPU via llama-server (fallback)
    GPU_VLLM = "gpu_vllm"                 # Safetensors via vLLM (Prime only)
    UNLOADED = "unloaded"                 # Not loaded anywhere
    ON_DEMAND = "on_demand"               # Loaded when needed, unloaded after (audio)


class TierStatus(BaseModel):
    """Status of a single cognitive tier."""
    name: str                                       # nano, core, prime
    role: str                                       # reflex, operator, thinker
    device: TierDevice = TierDevice.UNLOADED
    model_path: str = ""
    vram_mb: int = 0
    kv_cache_warm: bool = False                     # True if identity prefix is cached
    last_transition: Optional[datetime] = None
    inference_endpoint: str = ""                    # URL where this tier is serving


class GPUWatchState(BaseModel):
    """Complete GPU watch rotation state."""
    gpu_state: GPUState = GPUState.IDLE
    tiers: Dict[str, TierStatus] = Field(default_factory=lambda: {
        "nano": TierStatus(name="nano", role="reflex"),
        "core": TierStatus(name="core", role="operator"),
        "prime": TierStatus(name="prime", role="thinker"),
    })
    last_transition: Optional[datetime] = None
    transition_reason: str = ""
    transitions_total: int = 0


# Legacy — NanoGPUMode still used by existing backoff system
class NanoGPUMode(str, Enum):
    """Nano model GPU placement mode."""
    GPU = "gpu"
    CPU = "cpu"


class NanoGPUStatus(BaseModel):
    """Current Nano GPU/CPU placement status."""
    mode: NanoGPUMode = NanoGPUMode.GPU
    last_transition: Optional[datetime] = None
    reason: Optional[str] = None
    transitions: int = 0


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
    SERVICE_HEALTH_CHANGE = "service_health_change"
    HA_STATUS_CHANGE = "ha_status_change"


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
    nano: NanoGPUStatus = Field(default_factory=NanoGPUStatus)
    watch: GPUWatchState = Field(default_factory=GPUWatchState)
    containers: ContainerStatus = Field(default_factory=ContainerStatus)
    active_handoff: Optional[HandoffStatus] = None
    handoff_history: List[HandoffStatus] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    lifecycle: Optional[Dict] = Field(default=None, description="Lifecycle machine state for persistence across restarts")

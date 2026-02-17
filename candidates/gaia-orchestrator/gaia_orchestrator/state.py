"""
State persistence for GAIA Orchestrator.

Manages saving/loading orchestrator state to disk for crash recovery
and state inspection.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import asyncio
from contextlib import asynccontextmanager

from .config import get_config
from .models.schemas import (
    OrchestratorState,
    GPUStatus,
    GPUOwner,
    ContainerStatus,
    HandoffStatus,
)

logger = logging.getLogger("GAIA.Orchestrator.State")


class StateManager:
    """Manages orchestrator state with disk persistence."""

    def __init__(self, state_dir: Optional[Path] = None):
        config = get_config()
        self.state_dir = state_dir or config.state_dir
        self.state_file = self.state_dir / config.state_file
        self._state: OrchestratorState = OrchestratorState()
        self._lock = asyncio.Lock()
        self._dirty = False

    async def initialize(self) -> None:
        """Initialize state manager, loading existing state if present."""
        # Ensure state directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Load existing state if available
        if self.state_file.exists():
            try:
                await self._load_state()
                logger.info(f"Loaded existing state from {self.state_file}")
            except Exception as e:
                logger.warning(f"Failed to load state, starting fresh: {e}")
                self._state = OrchestratorState()
        else:
            logger.info("No existing state found, starting fresh")

    async def _load_state(self) -> None:
        """Load state from disk."""
        async with self._lock:
            content = self.state_file.read_text()
            data = json.loads(content)
            self._state = OrchestratorState.model_validate(data)

    async def _save_state(self) -> None:
        """Save state to disk."""
        self._state.last_updated = datetime.now(timezone.utc)
        content = self._state.model_dump_json(indent=2)

        # Write atomically via temp file
        temp_file = self.state_file.with_suffix(".tmp")
        temp_file.write_text(content)
        temp_file.rename(self.state_file)

        self._dirty = False
        logger.debug(f"State saved to {self.state_file}")

    async def save(self) -> None:
        """Explicitly save state to disk."""
        async with self._lock:
            await self._save_state()

    @asynccontextmanager
    async def modify(self):
        """Context manager for modifying state with automatic save."""
        async with self._lock:
            yield self._state
            await self._save_state()

    @property
    def state(self) -> OrchestratorState:
        """Get current state (read-only view)."""
        return self._state

    # =========================================================================
    # GPU State Accessors
    # =========================================================================

    async def get_gpu_status(self) -> GPUStatus:
        """Get current GPU ownership status."""
        return self._state.gpu

    async def set_gpu_owner(
        self,
        owner: GPUOwner,
        lease_id: str,
        reason: str
    ) -> None:
        """Set GPU ownership."""
        async with self.modify() as state:
            state.gpu.owner = owner
            state.gpu.lease_id = lease_id
            state.gpu.reason = reason
            state.gpu.acquired_at = datetime.now(timezone.utc)

    async def release_gpu(self) -> None:
        """Release GPU ownership."""
        async with self.modify() as state:
            state.gpu.owner = GPUOwner.NONE
            state.gpu.lease_id = None
            state.gpu.reason = None
            state.gpu.acquired_at = None

    async def add_to_gpu_queue(self, requester: str) -> int:
        """Add requester to GPU queue, return position."""
        async with self.modify() as state:
            if requester not in state.gpu.queue:
                state.gpu.queue.append(requester)
            return state.gpu.queue.index(requester) + 1

    async def remove_from_gpu_queue(self, requester: str) -> None:
        """Remove requester from GPU queue."""
        async with self.modify() as state:
            if requester in state.gpu.queue:
                state.gpu.queue.remove(requester)

    # =========================================================================
    # Container State Accessors
    # =========================================================================

    async def get_container_status(self) -> ContainerStatus:
        """Get container status for all stacks."""
        return self._state.containers

    async def update_container_status(self, status: ContainerStatus) -> None:
        """Update container status."""
        async with self.modify() as state:
            state.containers = status

    # =========================================================================
    # Handoff State Accessors
    # =========================================================================

    async def get_active_handoff(self) -> Optional[HandoffStatus]:
        """Get currently active handoff, if any."""
        return self._state.active_handoff

    async def start_handoff(self, handoff: HandoffStatus) -> None:
        """Start a new handoff operation."""
        async with self.modify() as state:
            state.active_handoff = handoff

    async def update_handoff(self, handoff: HandoffStatus) -> None:
        """Update active handoff status."""
        async with self.modify() as state:
            state.active_handoff = handoff

    async def complete_handoff(self, handoff: HandoffStatus) -> None:
        """Complete handoff and move to history."""
        async with self.modify() as state:
            handoff.completed_at = datetime.now(timezone.utc)
            state.handoff_history.append(handoff)
            state.active_handoff = None

            # Keep only last 100 handoffs in history
            if len(state.handoff_history) > 100:
                state.handoff_history = state.handoff_history[-100:]

    async def get_handoff_by_id(self, handoff_id: str) -> Optional[HandoffStatus]:
        """Get handoff by ID from active or history."""
        if self._state.active_handoff and self._state.active_handoff.handoff_id == handoff_id:
            return self._state.active_handoff

        for h in reversed(self._state.handoff_history):
            if h.handoff_id == handoff_id:
                return h

        return None


# Singleton instance
_state_manager: Optional[StateManager] = None


async def get_state_manager() -> StateManager:
    """Get the singleton state manager instance."""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
        await _state_manager.initialize()
    return _state_manager


async def reset_state_manager() -> None:
    """Reset state manager singleton (for testing)."""
    global _state_manager
    _state_manager = None

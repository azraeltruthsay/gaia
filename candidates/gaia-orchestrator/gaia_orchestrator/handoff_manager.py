"""
Handoff manager for GAIA Orchestrator.

Coordinates GPU handoff between services:
- Prime (Core) <-> Study for training sessions
- Live <-> Candidate for testing
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional

from .config import get_config
from .state import StateManager
from .gpu_manager import GPUManager
from .models.schemas import (
    HandoffStatus,
    HandoffPhase,
    HandoffType,
    HandoffRequest,
    GPUOwner,
)

logger = logging.getLogger("GAIA.Orchestrator.Handoff")


class HandoffManager:
    """Manages GPU handoff operations between services."""

    def __init__(self, state_manager: StateManager, gpu_manager: Optional[GPUManager] = None):
        self.state_manager = state_manager
        self.gpu_manager = gpu_manager
        self.config = get_config()

    async def _update_handoff_phase(
        self,
        handoff: HandoffStatus,
        phase: HandoffPhase,
        progress: int,
        error: Optional[str] = None
    ) -> HandoffStatus:
        """Update handoff phase and persist."""
        handoff.phase = phase
        handoff.progress_pct = progress
        if error:
            handoff.error = error
        await self.state_manager.update_handoff(handoff)
        logger.info(f"Handoff {handoff.handoff_id[:8]}: {phase.value} ({progress}%)")
        return handoff

    async def start_prime_to_study(self, request: HandoffRequest) -> HandoffStatus:
        """
        Initiate GPU handoff from Core (Prime) to Study.

        Protocol:
        1. Create handoff record
        2. Request Core to release GPU via /gpu/release
        3. Wait for CUDA cleanup (poll nvidia-smi)
        4. Update GPU lease to Study
        5. Signal Study via /study/gpu-ready
        """
        # Check if a handoff is already in progress
        active = await self.state_manager.get_active_handoff()
        if active and active.phase not in (HandoffPhase.COMPLETED, HandoffPhase.FAILED, HandoffPhase.CANCELLED):
            raise RuntimeError(f"Handoff already in progress: {active.handoff_id}")

        # Check current GPU owner
        gpu_status = await self.state_manager.get_gpu_status()
        if gpu_status.owner not in (GPUOwner.CORE, GPUOwner.NONE):
            raise RuntimeError(f"Cannot handoff - GPU owned by {gpu_status.owner.value}, expected gaia-core")

        # Create handoff record
        handoff = HandoffStatus(
            handoff_id=str(uuid.uuid4()),
            handoff_type=HandoffType.PRIME_TO_STUDY,
            phase=HandoffPhase.INITIATED,
            source=GPUOwner.CORE,
            destination=GPUOwner.STUDY,
            started_at=datetime.utcnow(),
            progress_pct=0,
        )
        await self.state_manager.start_handoff(handoff)
        logger.info(f"Starting Prime->Study handoff: {handoff.handoff_id}")

        try:
            # Phase 1: Request GPU release from Core
            handoff = await self._update_handoff_phase(handoff, HandoffPhase.RELEASING_GPU, 10)

            if self.gpu_manager:
                success = await self.gpu_manager.request_release_from_core()
                if not success:
                    raise RuntimeError("Core failed to acknowledge GPU release")

            # Phase 2: Wait for CUDA cleanup
            handoff = await self._update_handoff_phase(handoff, HandoffPhase.WAITING_CUDA_CLEANUP, 30)

            if self.gpu_manager:
                cleanup_ok = await self.gpu_manager.wait_for_gpu_cleanup(
                    timeout=request.timeout_seconds
                )
                if not cleanup_ok:
                    raise RuntimeError("GPU cleanup timeout - VRAM not released")

            # Phase 3: Transfer ownership
            handoff = await self._update_handoff_phase(handoff, HandoffPhase.TRANSFERRING_OWNERSHIP, 60)

            lease_id = str(uuid.uuid4())
            await self.state_manager.set_gpu_owner(
                GPUOwner.STUDY,
                lease_id,
                "prime_to_study_handoff"
            )

            # Phase 4: Signal Study
            handoff = await self._update_handoff_phase(handoff, HandoffPhase.SIGNALING_RECIPIENT, 80)

            if self.gpu_manager:
                await self.gpu_manager.signal_study_gpu_ready()

            # Complete
            handoff = await self._update_handoff_phase(handoff, HandoffPhase.COMPLETED, 100)
            await self.state_manager.complete_handoff(handoff)

            logger.info(f"Prime->Study handoff complete: {handoff.handoff_id}")
            return handoff

        except Exception as e:
            logger.exception(f"Handoff failed: {e}")
            handoff = await self._update_handoff_phase(
                handoff,
                HandoffPhase.FAILED,
                handoff.progress_pct,
                str(e)
            )
            await self.state_manager.complete_handoff(handoff)
            raise

    async def start_study_to_prime(self, request: HandoffRequest) -> HandoffStatus:
        """
        Initiate GPU handoff from Study back to Core (Prime).

        Protocol:
        1. Create handoff record
        2. Request Study to release GPU
        3. Wait for CUDA cleanup
        4. Update GPU lease to Core
        5. Request Core to reclaim GPU via /gpu/reclaim
        """
        # Check if a handoff is already in progress
        active = await self.state_manager.get_active_handoff()
        if active and active.phase not in (HandoffPhase.COMPLETED, HandoffPhase.FAILED, HandoffPhase.CANCELLED):
            raise RuntimeError(f"Handoff already in progress: {active.handoff_id}")

        # Check current GPU owner
        gpu_status = await self.state_manager.get_gpu_status()
        if gpu_status.owner not in (GPUOwner.STUDY, GPUOwner.NONE):
            raise RuntimeError(f"Cannot handoff - GPU owned by {gpu_status.owner.value}, expected gaia-study")

        # Create handoff record
        handoff = HandoffStatus(
            handoff_id=str(uuid.uuid4()),
            handoff_type=HandoffType.STUDY_TO_PRIME,
            phase=HandoffPhase.INITIATED,
            source=GPUOwner.STUDY,
            destination=GPUOwner.CORE,
            started_at=datetime.utcnow(),
            progress_pct=0,
        )
        await self.state_manager.start_handoff(handoff)
        logger.info(f"Starting Study->Prime handoff: {handoff.handoff_id}")

        try:
            # Phase 1: Request GPU release from Study
            handoff = await self._update_handoff_phase(handoff, HandoffPhase.RELEASING_GPU, 10)

            if self.gpu_manager:
                success = await self.gpu_manager.signal_study_gpu_release()
                if not success:
                    raise RuntimeError("Study failed to acknowledge GPU release")

            # Phase 2: Wait for CUDA cleanup
            handoff = await self._update_handoff_phase(handoff, HandoffPhase.WAITING_CUDA_CLEANUP, 30)

            if self.gpu_manager:
                cleanup_ok = await self.gpu_manager.wait_for_gpu_cleanup(
                    timeout=request.timeout_seconds
                )
                if not cleanup_ok:
                    raise RuntimeError("GPU cleanup timeout - VRAM not released")

            # Phase 3: Transfer ownership
            handoff = await self._update_handoff_phase(handoff, HandoffPhase.TRANSFERRING_OWNERSHIP, 60)

            lease_id = str(uuid.uuid4())
            await self.state_manager.set_gpu_owner(
                GPUOwner.CORE,
                lease_id,
                "study_to_prime_handoff"
            )

            # Phase 4: Signal Core to reclaim
            handoff = await self._update_handoff_phase(handoff, HandoffPhase.SIGNALING_RECIPIENT, 80)

            if self.gpu_manager:
                await self.gpu_manager.request_reclaim_by_core()

            # Complete
            handoff = await self._update_handoff_phase(handoff, HandoffPhase.COMPLETED, 100)
            await self.state_manager.complete_handoff(handoff)

            logger.info(f"Study->Prime handoff complete: {handoff.handoff_id}")
            return handoff

        except Exception as e:
            logger.exception(f"Handoff failed: {e}")
            handoff = await self._update_handoff_phase(
                handoff,
                HandoffPhase.FAILED,
                handoff.progress_pct,
                str(e)
            )
            await self.state_manager.complete_handoff(handoff)
            raise

    async def cancel_handoff(self, handoff_id: str) -> HandoffStatus:
        """Cancel an in-progress handoff."""
        handoff = await self.state_manager.get_handoff_by_id(handoff_id)
        if handoff is None:
            raise ValueError(f"Handoff not found: {handoff_id}")

        if handoff.phase in (HandoffPhase.COMPLETED, HandoffPhase.FAILED, HandoffPhase.CANCELLED):
            raise ValueError(f"Handoff already terminated: {handoff.phase.value}")

        handoff = await self._update_handoff_phase(
            handoff,
            HandoffPhase.CANCELLED,
            handoff.progress_pct,
            "Cancelled by user"
        )
        await self.state_manager.complete_handoff(handoff)

        logger.info(f"Handoff cancelled: {handoff_id}")
        return handoff

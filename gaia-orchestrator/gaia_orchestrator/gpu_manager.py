"""
GPU management for GAIA Orchestrator.

Monitors GPU state via pynvml and coordinates GPU ownership
between services.
"""

import asyncio
import logging
from typing import Optional

from .config import get_config
from .state import StateManager
from .models.schemas import GPUMemoryInfo, GPUOwner

logger = logging.getLogger("GAIA.Orchestrator.GPU")

# Try to import pynvml, but don't fail if not available
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("pynvml not available - GPU monitoring will be limited")


class GPUManager:
    """Manages GPU resources and monitors VRAM usage."""

    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.config = get_config()
        self._nvml_initialized = False

    def _ensure_nvml(self) -> bool:
        """Ensure NVML is initialized."""
        if not PYNVML_AVAILABLE:
            return False

        if not self._nvml_initialized:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize NVML: {e}")
                return False

        return True

    async def get_memory_info(self) -> Optional[GPUMemoryInfo]:
        """Get GPU memory usage information."""
        if not self._ensure_nvml():
            return None

        try:
            # Get first GPU (index 0)
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            return GPUMemoryInfo(
                total_mb=mem_info.total // (1024 * 1024),
                used_mb=mem_info.used // (1024 * 1024),
                free_mb=mem_info.free // (1024 * 1024),
            )
        except Exception as e:
            logger.error(f"Failed to get GPU memory info: {e}")
            return None

    async def is_gpu_free(self) -> bool:
        """
        Check if GPU is considered 'free' (VRAM usage below threshold).

        This is used during handoff to verify CUDA has released memory.
        """
        mem_info = await self.get_memory_info()
        if mem_info is None:
            # Can't check - assume not free
            return False

        threshold = self.config.gpu_cleanup_threshold_mb
        return mem_info.used_mb < threshold

    async def wait_for_gpu_cleanup(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for GPU memory to be released.

        Args:
            timeout: Max seconds to wait. Uses config default if None.

        Returns:
            True if GPU is now free, False if timeout occurred.
        """
        if timeout is None:
            timeout = self.config.gpu_cleanup_timeout_seconds

        poll_interval = self.config.gpu_cleanup_poll_interval
        elapsed = 0.0

        logger.info(f"Waiting for GPU cleanup (threshold: {self.config.gpu_cleanup_threshold_mb}MB)...")

        while elapsed < timeout:
            if await self.is_gpu_free():
                logger.info(f"GPU cleanup complete after {elapsed:.1f}s")
                return True

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            # Log progress
            mem_info = await self.get_memory_info()
            if mem_info:
                logger.debug(f"GPU memory: {mem_info.used_mb}MB used, waiting...")

        logger.warning(f"GPU cleanup timeout after {timeout}s")
        return False

    async def request_release_from_core(self) -> bool:
        """
        Request gaia-core to release GPU resources.

        Calls the /gpu/release endpoint on gaia-core.
        """
        import httpx

        url = f"{self.config.core_url}/gpu/release"
        logger.info(f"Requesting GPU release from Core: {url}")

        try:
            async with httpx.AsyncClient(timeout=self.config.http_timeout_seconds) as client:
                response = await client.post(url)

                if response.status_code == 200:
                    logger.info("Core acknowledged GPU release request")
                    return True
                else:
                    logger.error(f"Core GPU release failed: {response.status_code} {response.text}")
                    return False

        except Exception as e:
            logger.error(f"Failed to request GPU release from Core: {e}")
            return False

    async def request_reclaim_by_core(self) -> bool:
        """
        Request gaia-core to reclaim GPU resources.

        Calls the /gpu/reclaim endpoint on gaia-core.
        """
        import httpx

        url = f"{self.config.core_url}/gpu/reclaim"
        logger.info(f"Requesting GPU reclaim by Core: {url}")

        try:
            async with httpx.AsyncClient(timeout=self.config.http_timeout_seconds) as client:
                response = await client.post(url)

                if response.status_code == 200:
                    logger.info("Core acknowledged GPU reclaim request")
                    return True
                else:
                    logger.error(f"Core GPU reclaim failed: {response.status_code} {response.text}")
                    return False

        except Exception as e:
            logger.error(f"Failed to request GPU reclaim by Core: {e}")
            return False

    async def signal_study_gpu_ready(self) -> bool:
        """
        Signal gaia-study that GPU is now available.

        Calls the /study/gpu-ready endpoint.
        """
        import httpx

        url = f"{self.config.study_url}/study/gpu-ready"
        logger.info(f"Signaling Study GPU ready: {url}")

        try:
            async with httpx.AsyncClient(timeout=self.config.http_timeout_seconds) as client:
                response = await client.post(url)

                if response.status_code == 200:
                    logger.info("Study acknowledged GPU ready signal")
                    return True
                else:
                    logger.warning(f"Study GPU ready signal: {response.status_code}")
                    # Not critical if study doesn't respond
                    return True

        except Exception as e:
            logger.warning(f"Failed to signal Study GPU ready: {e}")
            # Not critical
            return True

    async def signal_study_gpu_release(self) -> bool:
        """
        Request gaia-study to release GPU resources.
        """
        import httpx

        url = f"{self.config.study_url}/study/gpu-release"
        logger.info(f"Requesting GPU release from Study: {url}")

        try:
            async with httpx.AsyncClient(timeout=self.config.http_timeout_seconds) as client:
                response = await client.post(url)

                if response.status_code == 200:
                    logger.info("Study acknowledged GPU release request")
                    return True
                else:
                    logger.error(f"Study GPU release failed: {response.status_code}")
                    return False

        except Exception as e:
            logger.error(f"Failed to request GPU release from Study: {e}")
            return False

    def shutdown(self):
        """Shutdown NVML."""
        if self._nvml_initialized and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
                self._nvml_initialized = False
            except Exception:
                pass

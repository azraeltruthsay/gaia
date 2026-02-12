"""
GPU management for GAIA Orchestrator.

Monitors GPU state via pynvml and coordinates GPU ownership
between services. Uses Docker container stop/start for VRAM
release/reclaim since vLLM sleep mode cannot offload weights
with --enforce-eager on Blackwell (sm_120).

Model weights are served from a tmpfs warm pool (/mnt/gaia_warm_pool)
seeded at boot by systemd, so cold starts load from RAM (~37s)
rather than NVMe (~41s). KV cache blocks evicted from GPU are
offloaded to an 8GB CPU RAM buffer (--kv-offloading-backend native).
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

# Try to import docker SDK
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    logger.warning("docker SDK not available - container management will be limited")


class GPUManager:
    """Manages GPU resources and monitors VRAM usage."""

    PRIME_CONTAINER_NAME = "gaia-prime-candidate"

    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.config = get_config()
        self._nvml_initialized = False
        self._docker_client: Optional["docker.DockerClient"] = None

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

        If NVML is available, polls VRAM usage until it drops below threshold.
        If NVML is not available (orchestrator doesn't have NVIDIA runtime),
        falls back to checking whether the prime container is stopped — if the
        container is not running, VRAM is guaranteed released.

        Args:
            timeout: Max seconds to wait. Uses config default if None.

        Returns:
            True if GPU is now free, False if timeout occurred.
        """
        if timeout is None:
            timeout = self.config.gpu_cleanup_timeout_seconds

        poll_interval = self.config.gpu_cleanup_poll_interval

        # Fast path: if NVML is not available, check container status instead
        if not self._ensure_nvml():
            logger.info("NVML not available — checking container status for GPU cleanup")
            try:
                client = self._get_docker_client()
                container = client.containers.get(self.PRIME_CONTAINER_NAME)
                container.reload()  # refresh status
                if container.status != "running":
                    logger.info(f"Prime container is '{container.status}' — GPU cleanup confirmed")
                    return True
                else:
                    logger.warning("Prime container still running but NVML unavailable — cannot verify VRAM")
                    return False
            except Exception as e:
                logger.error(f"Cannot verify GPU cleanup (no NVML, Docker check failed): {e}")
                return False

        # NVML available — poll VRAM usage
        elapsed = 0.0
        logger.info(f"Waiting for GPU cleanup (threshold: {self.config.gpu_cleanup_threshold_mb}MB)...")

        while elapsed < timeout:
            if await self.is_gpu_free():
                logger.info(f"GPU cleanup complete after {elapsed:.1f}s")
                return True

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            mem_info = await self.get_memory_info()
            if mem_info:
                logger.debug(f"GPU memory: {mem_info.used_mb}MB used, waiting...")

        logger.warning(f"GPU cleanup timeout after {timeout}s")
        return False

    def _get_docker_client(self) -> "docker.DockerClient":
        """Get or create Docker client (lazy init)."""
        if self._docker_client is None:
            if not DOCKER_AVAILABLE:
                raise RuntimeError("docker SDK not installed")
            self._docker_client = docker.from_env()
        return self._docker_client

    async def stop_prime_container(self) -> bool:
        """
        Stop the gaia-prime container to fully release VRAM.

        This is used instead of vLLM sleep mode because --enforce-eager
        (required for Blackwell sm_120) prevents CuMemAllocator from
        tracking weight tensors, so sleep only frees KV cache (~1.7GB)
        rather than the full ~10.5GB.

        Container stop releases all VRAM in ~2-3 seconds.
        The 8GB CPU KV offload buffer is also freed with the container.
        """
        try:
            client = self._get_docker_client()
            container = client.containers.get(self.PRIME_CONTAINER_NAME)

            if container.status != "running":
                logger.info(f"Prime container already stopped (status: {container.status})")
                return True

            logger.info(f"Stopping prime container '{self.PRIME_CONTAINER_NAME}'...")
            await asyncio.to_thread(container.stop, timeout=30)
            logger.info("Prime container stopped — VRAM released")
            return True

        except Exception as e:
            logger.error(f"Failed to stop prime container: {e}")
            return False

    async def start_prime_container(self) -> bool:
        """
        Start the gaia-prime container and wait for it to become healthy.

        Cold start takes ~37s from tmpfs warm pool (was ~41s from NVMe).
        """
        try:
            client = self._get_docker_client()
            container = client.containers.get(self.PRIME_CONTAINER_NAME)

            if container.status == "running":
                logger.info("Prime container already running")
                return True

            logger.info(f"Starting prime container '{self.PRIME_CONTAINER_NAME}'...")
            await asyncio.to_thread(container.start)

            # Wait for healthy (model loaded, serving)
            prime_url = getattr(self.config, "prime_url",
                                "http://gaia-prime-candidate:7777")
            logger.info(f"Waiting for prime to become healthy at {prime_url}...")

            import httpx
            max_wait = 120  # seconds
            poll_interval = 3.0
            elapsed = 0.0

            while elapsed < max_wait:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
                try:
                    async with httpx.AsyncClient(timeout=5) as hc:
                        resp = await hc.get(f"{prime_url}/health")
                        if resp.status_code == 200:
                            logger.info(f"Prime healthy after {elapsed:.0f}s")
                            return True
                except Exception:
                    pass
                if elapsed % 15 == 0:
                    logger.debug(f"Still waiting for prime... ({elapsed:.0f}s)")

            logger.error(f"Prime did not become healthy after {max_wait}s")
            return False

        except Exception as e:
            logger.error(f"Failed to start prime container: {e}")
            return False

    async def request_release_from_core(self) -> bool:
        """
        Release GPU: stop the prime container, then tell gaia-core
        to demote gpu_prime from its model pool.
        """
        import httpx

        # Step 1: Stop the prime container (frees all VRAM)
        if not await self.stop_prime_container():
            logger.error("Failed to stop prime container — aborting release")
            return False

        # Step 2: Tell core to update its model pool (demote gpu_prime)
        url = f"{self.config.core_url}/gpu/release"
        logger.info(f"Requesting model pool update from Core: {url}")

        try:
            async with httpx.AsyncClient(timeout=self.config.http_timeout_seconds) as client:
                response = await client.post(url)

                if response.status_code == 200:
                    logger.info("Core acknowledged GPU release — fallback chain active")
                    return True
                else:
                    logger.error(f"Core GPU release failed: {response.status_code} {response.text}")
                    return False

        except Exception as e:
            logger.error(f"Failed to request GPU release from Core: {e}")
            return False

    async def request_reclaim_by_core(self) -> bool:
        """
        Reclaim GPU: start the prime container, wait for healthy,
        then tell gaia-core to restore gpu_prime in its model pool.
        """
        import httpx

        # Step 1: Start the prime container (loads model into VRAM)
        if not await self.start_prime_container():
            logger.error("Failed to start prime container — aborting reclaim")
            return False

        # Step 2: Tell core to restore gpu_prime in model pool
        url = f"{self.config.core_url}/gpu/reclaim"
        logger.info(f"Requesting model pool restore from Core: {url}")

        try:
            async with httpx.AsyncClient(timeout=self.config.http_timeout_seconds) as client:
                response = await client.post(url)

                if response.status_code == 200:
                    logger.info("Core acknowledged GPU reclaim — prime inference restored")
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

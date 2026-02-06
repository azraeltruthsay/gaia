"""
Docker management for GAIA Orchestrator.

Wraps the Docker SDK to provide container lifecycle operations
for live and candidate stacks.
"""

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import docker
from docker.errors import DockerException, NotFound

from .config import get_config
from .state import StateManager
from .models.schemas import ContainerStatus, ServiceHealth, ContainerState

logger = logging.getLogger("GAIA.Orchestrator.Docker")


class DockerManager:
    """Manages Docker container lifecycle for GAIA services."""

    # Service definitions
    LIVE_SERVICES = ["gaia-core", "gaia-web", "gaia-mcp", "gaia-study"]
    CANDIDATE_SERVICES = ["gaia-core-candidate", "gaia-web-candidate", "gaia-mcp-candidate", "gaia-study-candidate"]

    SERVICE_PORTS = {
        "gaia-core": 6415,
        "gaia-web": 6414,
        "gaia-mcp": 8765,
        "gaia-study": 8766,
        "gaia-core-candidate": 6416,
        "gaia-web-candidate": 6417,
        "gaia-mcp-candidate": 8767,
        "gaia-study-candidate": 8768,
    }

    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.config = get_config()
        self._client: Optional[docker.DockerClient] = None

    @property
    def client(self) -> docker.DockerClient:
        """Get or create Docker client."""
        if self._client is None:
            try:
                self._client = docker.from_env()
            except DockerException as e:
                logger.error(f"Failed to connect to Docker: {e}")
                raise
        return self._client

    def _get_container_state(self, container_name: str) -> ContainerState:
        """Get the state of a container by name."""
        try:
            container = self.client.containers.get(container_name)
            status = container.status.lower()

            if status == "running":
                return ContainerState.RUNNING
            elif status in ("exited", "dead"):
                return ContainerState.STOPPED
            elif status == "created":
                return ContainerState.STOPPED
            elif status in ("restarting", "paused"):
                return ContainerState.STARTING
            else:
                return ContainerState.UNKNOWN

        except NotFound:
            return ContainerState.STOPPED
        except Exception as e:
            logger.warning(f"Error getting container state for {container_name}: {e}")
            return ContainerState.UNKNOWN

    async def _check_service_health(self, container_name: str, port: int) -> bool:
        """Check if a service is healthy via its health endpoint."""
        import httpx

        url = f"http://localhost:{port}/health"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                return response.status_code == 200
        except Exception:
            return False

    async def get_status(self) -> ContainerStatus:
        """Get status of all containers."""
        status = ContainerStatus()

        # Check live services
        for service in self.LIVE_SERVICES:
            port = self.SERVICE_PORTS.get(service, 0)
            state = self._get_container_state(service)
            healthy = False
            if state == ContainerState.RUNNING:
                healthy = await self._check_service_health(service, port)

            # Extract short name (without gaia- prefix)
            short_name = service.replace("gaia-", "")
            status.live[short_name] = ServiceHealth(
                name=service,
                state=state,
                port=port,
                healthy=healthy,
            )

        # Check candidate services
        for service in self.CANDIDATE_SERVICES:
            port = self.SERVICE_PORTS.get(service, 0)
            state = self._get_container_state(service)
            healthy = False
            if state == ContainerState.RUNNING:
                healthy = await self._check_service_health(service, port)

            # Extract short name
            short_name = service.replace("gaia-", "").replace("-candidate", "")
            status.candidate[short_name] = ServiceHealth(
                name=service,
                state=state,
                port=port,
                healthy=healthy,
            )

        return status

    async def _run_compose(
        self,
        compose_file: Path,
        command: List[str],
        env: Optional[Dict[str, str]] = None
    ) -> Dict:
        """Run a docker compose command."""
        cmd = ["docker", "compose", "-f", str(compose_file)] + command

        logger.info(f"Running: {' '.join(cmd)}")

        # Run in executor to not block
        loop = asyncio.get_event_loop()

        def run_subprocess():
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env={**dict(subprocess.os.environ), **(env or {})},
                cwd=str(compose_file.parent),
            )
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        result = await loop.run_in_executor(None, run_subprocess)

        if result["returncode"] != 0:
            logger.error(f"Compose command failed: {result['stderr']}")
            raise RuntimeError(f"Compose command failed: {result['stderr']}")

        return result

    async def stop_live(self) -> Dict:
        """Stop the live stack."""
        logger.info("Stopping live stack...")
        return await self._run_compose(
            self.config.compose_file_live,
            ["down"]
        )

    async def start_live(self, gpu_enabled: bool = True) -> Dict:
        """Start the live stack."""
        logger.info(f"Starting live stack (GPU: {gpu_enabled})...")

        env = {}
        if not gpu_enabled:
            env["GAIA_AUTOLOAD_MODELS"] = "0"

        return await self._run_compose(
            self.config.compose_file_live,
            ["up", "-d"],
            env=env
        )

    async def stop_candidate(self) -> Dict:
        """Stop the candidate stack."""
        logger.info("Stopping candidate stack...")
        return await self._run_compose(
            self.config.compose_file_candidate,
            ["--profile", "full", "down"]
        )

    async def start_candidate(self, gpu_enabled: bool = True) -> Dict:
        """Start the candidate stack."""
        logger.info(f"Starting candidate stack (GPU: {gpu_enabled})...")

        env = {}
        if not gpu_enabled:
            env["GAIA_AUTOLOAD_MODELS"] = "0"

        return await self._run_compose(
            self.config.compose_file_candidate,
            ["--profile", "full", "up", "-d"],
            env=env
        )

    async def swap_service(self, service: str, target: str) -> Dict:
        """
        Swap a service between live and candidate.

        This injects a candidate service into the live traffic flow.
        """
        if service not in ("core", "mcp", "study", "web"):
            raise ValueError(f"Unknown service: {service}")

        if target not in ("live", "candidate"):
            raise ValueError(f"Unknown target: {target}")

        logger.info(f"Swapping {service} -> {target}")

        if target == "candidate":
            # Start the candidate service if not running
            candidate_name = f"gaia-{service}-candidate"
            state = self._get_container_state(candidate_name)

            if state != ContainerState.RUNNING:
                logger.info(f"Starting {candidate_name}...")
                await self._run_compose(
                    self.config.compose_file_candidate,
                    ["--profile", service, "up", "-d", candidate_name]
                )
                # Wait for it to be healthy
                await asyncio.sleep(3)

            # Now restart the caller service with the candidate endpoint
            endpoint_map = {
                "mcp": ("MCP_ENDPOINT", f"http://gaia-mcp-candidate:8765/jsonrpc", "gaia-core"),
                "study": ("STUDY_ENDPOINT", f"http://gaia-study-candidate:8766", "gaia-core"),
                "core": ("CORE_ENDPOINT", f"http://gaia-core-candidate:6416", "gaia-web"),
            }

            if service == "web":
                return {
                    "message": "Web doesn't need injection - access candidate at http://localhost:6417"
                }

            env_var, endpoint_url, caller = endpoint_map[service]

            return await self._run_compose(
                self.config.compose_file_live,
                ["up", "-d", caller],
                env={env_var: endpoint_url}
            )

        else:  # target == "live"
            # Restart caller with default (live) endpoint
            caller_map = {
                "mcp": "gaia-core",
                "study": "gaia-core",
                "core": "gaia-web",
                "web": None,
            }

            caller = caller_map[service]
            if caller is None:
                return {"message": "No swap needed for web"}

            return await self._run_compose(
                self.config.compose_file_live,
                ["up", "-d", caller]
            )

    async def stop_container(self, container_name: str) -> bool:
        """Stop a specific container."""
        try:
            container = self.client.containers.get(container_name)
            container.stop(timeout=30)
            logger.info(f"Stopped container: {container_name}")
            return True
        except NotFound:
            logger.warning(f"Container not found: {container_name}")
            return False
        except Exception as e:
            logger.error(f"Error stopping container {container_name}: {e}")
            raise

    async def start_container(self, container_name: str) -> bool:
        """Start a specific container."""
        try:
            container = self.client.containers.get(container_name)
            container.start()
            logger.info(f"Started container: {container_name}")
            return True
        except NotFound:
            logger.warning(f"Container not found: {container_name}")
            return False
        except Exception as e:
            logger.error(f"Error starting container {container_name}: {e}")
            raise

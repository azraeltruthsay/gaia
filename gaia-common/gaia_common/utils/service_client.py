"""
Service Client - HTTP client for inter-service communication in GAIA SOA.

Provides a simple async HTTP client for calling other GAIA services.
"""

import os
import logging
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import httpx

logger = logging.getLogger(__name__)


class ServiceClient:
    """
    HTTP client for making requests to GAIA services.

    Usage:
        client = ServiceClient("gaia-study", default_port=8766)
        result = await client.post("/study/start", {"adapter_name": "test"})
    """

    def __init__(
        self,
        service_name: str,
        default_port: int = 8000,
        timeout: float = 30.0,
        endpoint_env_var: Optional[str] = None
    ):
        """
        Initialize a service client.

        Args:
            service_name: Name of the service (used for default URL construction)
            default_port: Default port if not specified in environment
            timeout: Request timeout in seconds
            endpoint_env_var: Environment variable name for the endpoint URL
        """
        self.service_name = service_name
        self.timeout = timeout

        # Determine base URL from environment or construct default
        if endpoint_env_var:
            self.base_url = os.getenv(endpoint_env_var)
        else:
            env_var = f"{service_name.upper().replace('-', '_')}_ENDPOINT"
            self.base_url = os.getenv(env_var)

        if not self.base_url:
            # Default to Docker network naming convention
            self.base_url = f"http://{service_name}:{default_port}"

        logger.debug(f"ServiceClient for {service_name} initialized with base_url={self.base_url}")

    async def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make a GET request to the service."""
        url = urljoin(self.base_url, path)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, params=params, **kwargs)
            response.raise_for_status()
            return response.json()

    async def post(
        self,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make a POST request to the service."""
        url = urljoin(self.base_url, path)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=data, **kwargs)
            response.raise_for_status()
            return response.json()

    async def delete(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make a DELETE request to the service."""
        url = urljoin(self.base_url, path)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.delete(url, params=params, **kwargs)
            response.raise_for_status()
            return response.json()

    async def health_check(self) -> bool:
        """Check if the service is healthy."""
        try:
            result = await self.get("/health")
            return result.get("status") == "healthy"
        except Exception as e:
            logger.warning(f"Health check failed for {self.service_name}: {e}")
            return False


# Pre-configured clients for GAIA services
def get_study_client() -> ServiceClient:
    """Get a client for the gaia-study service."""
    return ServiceClient("gaia-study", default_port=8766, endpoint_env_var="STUDY_ENDPOINT")


def get_core_client() -> ServiceClient:
    """Get a client for the gaia-core service."""
    return ServiceClient("gaia-core", default_port=6415, endpoint_env_var="CORE_ENDPOINT")


def get_mcp_client() -> ServiceClient:
    """Get a client for the gaia-mcp service."""
    return ServiceClient("gaia-mcp", default_port=8765, endpoint_env_var="MCP_ENDPOINT")

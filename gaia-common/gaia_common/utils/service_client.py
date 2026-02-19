"""
Service Client - HTTP client for inter-service communication in GAIA SOA.

Provides a simple async HTTP client for calling other GAIA services.
Includes automatic retry-with-backoff on transient failures and
optional HA failover to a fallback endpoint.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import httpx

from gaia_common.utils.resilience import async_retry

logger = logging.getLogger(__name__)

# Transient exceptions that warrant a retry (and trigger failover)
_RETRYABLE_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.RemoteProtocolError,
)

# File-based maintenance mode flag (shared Docker volume)
_MAINTENANCE_FLAG = Path("/shared/ha_maintenance")


class ServiceClient:
    """
    HTTP client for making requests to GAIA services.

    Supports optional HA failover: if a ``fallback_base_url`` is provided and
    the primary endpoint fails with a retryable error (after exhausting
    retries), a single attempt is made against the fallback. Failover is
    suppressed when HA maintenance mode is active.

    Usage:
        client = ServiceClient("gaia-study", default_port=8766)
        result = await client.post("/study/start", {"adapter_name": "test"})
    """

    def __init__(
        self,
        service_name: str,
        default_port: int = 8000,
        timeout: float = 30.0,
        endpoint_env_var: Optional[str] = None,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
        fallback_base_url: Optional[str] = None,
    ):
        """
        Initialize a service client.

        Args:
            service_name: Name of the service (used for default URL construction)
            default_port: Default port if not specified in environment
            timeout: Request timeout in seconds
            endpoint_env_var: Environment variable name for the endpoint URL
            max_retries: Maximum retry attempts on transient failures
            retry_base_delay: Base delay between retries (doubles each attempt)
            fallback_base_url: Optional HA fallback URL. On retryable failure
                after primary retries are exhausted, a single request is sent
                here (unless maintenance mode is active).
        """
        self.service_name = service_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.fallback_base_url = fallback_base_url

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
        if self.fallback_base_url:
            logger.debug(f"ServiceClient for {service_name} HA fallback={self.fallback_base_url}")

    @staticmethod
    def _is_maintenance_mode() -> bool:
        """Check if HA maintenance mode is active (file-based flag on shared volume)."""
        return _MAINTENANCE_FLAG.exists()

    def _can_failover(self) -> bool:
        """Return True if fallback is configured and maintenance mode is off."""
        return bool(self.fallback_base_url) and not self._is_maintenance_mode()

    async def _try_fallback(
        self, method: str, path: str, primary_exc: Exception, **request_kwargs
    ) -> Dict[str, Any]:
        """Attempt a single request against the fallback endpoint.

        Args:
            method: HTTP method (GET, POST, DELETE).
            path: URL path.
            primary_exc: The exception from the primary that triggered failover.
            **request_kwargs: Forwarded to httpx (params, json, etc.).

        Returns:
            Parsed JSON response from the fallback.

        Raises:
            The original primary_exc if fallback also fails.
        """
        fallback_url = urljoin(self.fallback_base_url, path)
        logger.warning(
            "Primary %s %s failed (%s), attempting HA fallback to %s",
            method, path, type(primary_exc).__name__, fallback_url,
        )
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await getattr(client, method.lower())(fallback_url, **request_kwargs)
                response.raise_for_status()
                logger.info("HA fallback succeeded: %s %s", method, fallback_url)
                return response.json()
        except Exception as fallback_exc:
            logger.error(
                "HA fallback also failed: %s %s (%s). Raising original error.",
                method, fallback_url, type(fallback_exc).__name__,
            )
            raise primary_exc from fallback_exc

    async def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        retry: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make a GET request to the service (with retry by default)."""
        url = urljoin(self.base_url, path)

        async def _do_get() -> Dict[str, Any]:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params, **kwargs)
                response.raise_for_status()
                return response.json()

        try:
            if retry:
                return await async_retry(
                    _do_get,
                    max_attempts=self.max_retries,
                    base_delay=self.retry_base_delay,
                    retryable_exceptions=_RETRYABLE_EXCEPTIONS,
                )
            return await _do_get()
        except _RETRYABLE_EXCEPTIONS as exc:
            if self._can_failover():
                return await self._try_fallback("GET", path, exc, params=params, **kwargs)
            raise

    async def post(
        self,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        retry: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make a POST request to the service (with retry by default)."""
        url = urljoin(self.base_url, path)

        async def _do_post() -> Dict[str, Any]:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=data, **kwargs)
                response.raise_for_status()
                return response.json()

        try:
            if retry:
                return await async_retry(
                    _do_post,
                    max_attempts=self.max_retries,
                    base_delay=self.retry_base_delay,
                    retryable_exceptions=_RETRYABLE_EXCEPTIONS,
                )
            return await _do_post()
        except _RETRYABLE_EXCEPTIONS as exc:
            if self._can_failover():
                return await self._try_fallback("POST", path, exc, json=data, **kwargs)
            raise

    async def delete(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        retry: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make a DELETE request to the service (with retry by default)."""
        url = urljoin(self.base_url, path)

        async def _do_delete() -> Dict[str, Any]:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.delete(url, params=params, **kwargs)
                response.raise_for_status()
                return response.json()

        try:
            if retry:
                return await async_retry(
                    _do_delete,
                    max_attempts=self.max_retries,
                    base_delay=self.retry_base_delay,
                    retryable_exceptions=_RETRYABLE_EXCEPTIONS,
                )
            return await _do_delete()
        except _RETRYABLE_EXCEPTIONS as exc:
            if self._can_failover():
                return await self._try_fallback("DELETE", path, exc, params=params, **kwargs)
            raise

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


def get_core_client(
    fallback_base_url: Optional[str] = None,
) -> ServiceClient:
    """Get a client for the gaia-core service."""
    fallback = fallback_base_url or os.getenv("CORE_FALLBACK_ENDPOINT")
    return ServiceClient(
        "gaia-core", default_port=6415, endpoint_env_var="CORE_ENDPOINT",
        fallback_base_url=fallback,
    )


def get_mcp_client(
    fallback_base_url: Optional[str] = None,
) -> ServiceClient:
    """Get a client for the gaia-mcp service."""
    fallback = fallback_base_url or os.getenv("MCP_FALLBACK_ENDPOINT")
    return ServiceClient(
        "gaia-mcp", default_port=8765, endpoint_env_var="MCP_ENDPOINT",
        fallback_base_url=fallback,
    )


def get_orchestrator_client() -> ServiceClient:
    """Get a client for the gaia-orchestrator service."""
    return ServiceClient("gaia-orchestrator", default_port=6410, endpoint_env_var="ORCHESTRATOR_ENDPOINT")

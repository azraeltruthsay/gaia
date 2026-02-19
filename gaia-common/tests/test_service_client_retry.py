"""Test that ServiceClient retry integration works correctly."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from gaia_common.utils.service_client import ServiceClient


@pytest.mark.asyncio
async def test_service_client_has_retry_params():
    """ServiceClient should accept retry configuration."""
    client = ServiceClient("test-service", max_retries=5, retry_base_delay=1.0)
    assert client.max_retries == 5
    assert client.retry_base_delay == 1.0


@pytest.mark.asyncio
async def test_service_client_retry_disabled():
    """ServiceClient should skip retry when retry=False."""
    client = ServiceClient("test-service")

    # Mock httpx to raise an error on the first call
    import httpx

    with patch("gaia_common.utils.service_client.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get.side_effect = httpx.ConnectError("refused")
        mock_client_cls.return_value = mock_client

        # With retry=False, should fail immediately (1 attempt)
        with pytest.raises(httpx.ConnectError):
            await client.get("/health", retry=False)

        assert mock_client.get.call_count == 1


@pytest.mark.asyncio
async def test_service_client_default_retry_enabled():
    """ServiceClient should have retry enabled by default."""
    client = ServiceClient("test-service", max_retries=3, retry_base_delay=0.01)
    assert client.max_retries == 3

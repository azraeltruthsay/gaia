"""Shared fixtures for gaia-mcp tests."""

import pytest
from gaia_mcp.approval import ApprovalStore


@pytest.fixture
def approval_store():
    """Create a fresh ApprovalStore with a short TTL for testing."""
    return ApprovalStore(ttl_seconds=10)

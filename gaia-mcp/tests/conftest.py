"""Shared fixtures for gaia-mcp tests."""

import pyotp
import pytest
from gaia_common.utils.totp import TotpVerifier
from gaia_mcp.approval import ApprovalStore


@pytest.fixture
def totp_secret() -> str:
    """A known TOTP secret for tests -- never the real enrolled one."""
    return pyotp.random_base32()


@pytest.fixture
def approval_store(totp_secret):
    """A fresh ApprovalStore with a short TTL and a known TOTP secret
    (GAIA_Project-96xm): the real constructor loads the secret from
    /run/secrets, which won't exist in tests, so inject a verifier
    directly rather than relying on the filesystem.
    """
    store = ApprovalStore(ttl_seconds=10)
    store._totp_verifier = TotpVerifier(totp_secret)
    return store


def totp_code(secret: str) -> str:
    """Helper: the current valid code for a test secret."""
    return pyotp.TOTP(secret).now()

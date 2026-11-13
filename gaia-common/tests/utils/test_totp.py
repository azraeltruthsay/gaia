"""Tests for TOTP approval verification (GAIA_Project-96xm)."""

from __future__ import annotations

import time
from pathlib import Path

import pyotp
import pytest

from gaia_common.utils.totp import TotpVerifier, enroll, load_totp_verifier


@pytest.fixture
def secret() -> str:
    return pyotp.random_base32()


class TestTotpVerifier:
    def test_correct_code_verifies(self, secret):
        verifier = TotpVerifier(secret)
        code = pyotp.TOTP(secret).now()
        assert verifier.verify(code) is True

    def test_wrong_code_rejected(self, secret):
        verifier = TotpVerifier(secret)
        other_secret = pyotp.random_base32()
        wrong_code = pyotp.TOTP(other_secret).now()
        assert verifier.verify(wrong_code) is False

    def test_empty_code_rejected(self, secret):
        verifier = TotpVerifier(secret)
        assert verifier.verify("") is False
        assert verifier.verify(None) is False

    def test_replayed_code_rejected_within_reuse_window(self, secret):
        verifier = TotpVerifier(secret, reuse_window_s=90)
        code = pyotp.TOTP(secret).now()
        assert verifier.verify(code) is True
        # Same code, submitted again immediately -- must not approve twice.
        assert verifier.verify(code) is False

    def test_replayed_code_allowed_after_reuse_window_expires(self, secret):
        verifier = TotpVerifier(secret, reuse_window_s=0)
        code = pyotp.TOTP(secret).now()
        assert verifier.verify(code) is True
        # reuse_window_s=0 means no replay protection window to wait out.
        assert verifier.verify(code) is True

    def test_different_valid_codes_both_accepted(self, secret):
        verifier = TotpVerifier(secret)
        totp = pyotp.TOTP(secret)
        code_now = totp.now()
        assert verifier.verify(code_now) is True
        # A different (still-valid) code for an adjacent time step isn't a
        # replay of the first and should still be accepted.
        code_next = totp.at(time.time() + 30)
        if code_next != code_now:
            assert verifier.verify(code_next) is True


class TestLoadTotpVerifier:
    def test_loads_from_first_existing_path(self, tmp_path: Path, secret):
        secret_file = tmp_path / "totp_secret"
        secret_file.write_text(secret)
        verifier = load_totp_verifier(str(tmp_path / "missing"), str(secret_file))
        assert verifier is not None
        assert verifier.verify(pyotp.TOTP(secret).now()) is True

    def test_falls_back_to_env_var_path(self, tmp_path: Path, secret, monkeypatch):
        secret_file = tmp_path / "totp_secret"
        secret_file.write_text(secret)
        monkeypatch.setenv("GAIA_TEST_TOTP_PATH", str(secret_file))
        verifier = load_totp_verifier(str(tmp_path / "missing"), env_var="GAIA_TEST_TOTP_PATH")
        assert verifier is not None

    def test_returns_none_when_nothing_found(self, tmp_path: Path):
        verifier = load_totp_verifier(str(tmp_path / "missing"))
        assert verifier is None

    def test_returns_none_for_empty_secret_file(self, tmp_path: Path):
        secret_file = tmp_path / "totp_secret"
        secret_file.write_text("")
        verifier = load_totp_verifier(str(secret_file))
        assert verifier is None


class TestEnroll:
    def test_writes_secret_with_restricted_permissions(self, tmp_path: Path):
        target = tmp_path / "secrets" / "gaia_mcp_totp_secret"
        uri = enroll(target, name="gaia-mcp", issuer="GAIA MCP Approval")
        assert target.exists()
        assert oct(target.stat().st_mode)[-3:] == "600"
        assert uri.startswith("otpauth://totp/")
        assert "GAIA%20MCP%20Approval" in uri or "GAIA+MCP+Approval" in uri or "GAIA MCP Approval" in uri

    def test_enrolled_secret_verifies_real_codes(self, tmp_path: Path):
        target = tmp_path / "gaia_mcp_totp_secret"
        enroll(target, name="gaia-mcp", issuer="GAIA MCP Approval")
        secret_value = target.read_text().strip()
        verifier = TotpVerifier(secret_value)
        assert verifier.verify(pyotp.TOTP(secret_value).now()) is True

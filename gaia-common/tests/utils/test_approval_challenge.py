"""Tests for the TOTP-based approval challenge (GAIA_Project-96xm)."""

from __future__ import annotations

from pathlib import Path

import pyotp
import pytest

from gaia_common.utils import approval_challenge as ac


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ac, "_CHALLENGES_FILE", tmp_path / "approval_challenges.json")
    monkeypatch.setattr(ac, "_totp_verifier", None)
    yield
    monkeypatch.setattr(ac, "_totp_verifier", None)


@pytest.fixture
def secret() -> str:
    return pyotp.random_base32()


@pytest.fixture
def enrolled(tmp_path: Path, secret, monkeypatch):
    secret_file = tmp_path / "gaia_mcp_totp_secret"
    secret_file.write_text(secret)
    monkeypatch.setenv("GAIA_MCP_TOTP_SECRET_FILE", str(secret_file))
    return secret


class TestCreateChallenge:
    def test_creates_pending_entry_without_a_server_code(self):
        challenge = ac.create_challenge(action="write_candidates", context="4 files")
        assert challenge["action"] == "write_candidates"
        assert "code" not in challenge
        assert "reverse" not in challenge
        assert challenge["verified"] is False


class TestVerifyChallenge:
    def test_correct_totp_code_approves(self, enrolled):
        challenge = ac.create_challenge(action="promote")
        code = pyotp.TOTP(enrolled).now()
        result = ac.verify_challenge(challenge["challenge_id"], code)
        assert result["approved"] is True
        assert result["action"] == "promote"

    def test_wrong_code_rejected(self, enrolled):
        challenge = ac.create_challenge(action="promote")
        result = ac.verify_challenge(challenge["challenge_id"], "000000")
        assert result["approved"] is False

    def test_unknown_challenge_id_rejected(self, enrolled):
        result = ac.verify_challenge("not-a-real-id", pyotp.TOTP(enrolled).now())
        assert result["approved"] is False
        assert "not found" in result["reason"].lower()

    def test_already_verified_challenge_rejected_on_reuse(self, enrolled):
        challenge = ac.create_challenge(action="promote")
        code = pyotp.TOTP(enrolled).now()
        first = ac.verify_challenge(challenge["challenge_id"], code)
        assert first["approved"] is True
        second = ac.verify_challenge(challenge["challenge_id"], code)
        assert second["approved"] is False
        assert "already used" in second["reason"].lower()

    def test_fails_closed_when_no_secret_enrolled(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GAIA_MCP_TOTP_SECRET_FILE", raising=False)
        challenge = ac.create_challenge(action="promote")
        result = ac.verify_challenge(challenge["challenge_id"], "123456")
        assert result["approved"] is False


class TestFormatChallengePrompt:
    def test_prompt_asks_for_authenticator_code_not_a_reversed_string(self):
        challenge = ac.create_challenge(action="promote", context="2 files")
        text = ac.format_challenge_prompt(challenge)
        assert "authenticator code" in text
        assert "reverse" not in text.lower()

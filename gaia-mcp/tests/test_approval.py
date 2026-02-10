"""Tests for the ApprovalStore challenge-response system."""

import time
import pytest
from gaia_mcp.approval import ApprovalStore


class TestApprovalStoreCreate:
    """Test creating pending actions."""

    def test_create_pending_returns_tuple(self, approval_store):
        action_id, challenge, created_at, expiry = approval_store.create_pending(
            method="ai_write",
            params={"path": "/sandbox/test.txt", "content": "hello"},
        )
        assert isinstance(action_id, str)
        assert len(challenge) == 5
        assert challenge.isalpha()
        assert challenge.isupper()
        assert expiry > created_at

    def test_create_pending_with_proposal(self, approval_store):
        action_id, challenge, _, _ = approval_store.create_pending(
            method="run_shell",
            params={"cmd": "ls"},
            proposal="List directory contents",
        )
        pending = approval_store.list_pending()
        assert len(pending) == 1
        assert pending[0]["proposal"] == "List directory contents"

    def test_create_multiple_pending(self, approval_store):
        approval_store.create_pending(method="tool_a", params={})
        approval_store.create_pending(method="tool_b", params={})
        assert len(approval_store.list_pending()) == 2


class TestApprovalStoreApprove:
    """Test the approval challenge-response flow."""

    def test_approve_with_reversed_challenge(self, approval_store):
        action_id, challenge, _, _ = approval_store.create_pending(
            method="ai_write", params={"path": "/sandbox/f.txt"}
        )
        reversed_challenge = challenge[::-1]
        result = approval_store.approve(action_id, reversed_challenge)
        assert result["method"] == "ai_write"
        assert result["params"]["path"] == "/sandbox/f.txt"

    def test_approve_removes_from_pending(self, approval_store):
        action_id, challenge, _, _ = approval_store.create_pending(
            method="test", params={}
        )
        approval_store.approve(action_id, challenge[::-1])
        assert len(approval_store.list_pending()) == 0

    def test_approve_wrong_challenge_raises(self, approval_store):
        action_id, _, _, _ = approval_store.create_pending(
            method="test", params={}
        )
        with pytest.raises(ValueError, match="invalid approval challenge"):
            approval_store.approve(action_id, "WRONG")

    def test_approve_unknown_id_raises(self, approval_store):
        with pytest.raises(KeyError):
            approval_store.approve("nonexistent-id", "ABCDE")

    def test_approve_expired_raises(self):
        store = ApprovalStore(ttl_seconds=0)  # Expire immediately
        action_id, challenge, _, _ = store.create_pending(
            method="test", params={}
        )
        time.sleep(0.01)
        with pytest.raises(KeyError, match="expired"):
            store.approve(action_id, challenge[::-1])


class TestApprovalStoreCancel:
    """Test cancellation of pending actions."""

    def test_cancel_existing(self, approval_store):
        action_id, _, _, _ = approval_store.create_pending(
            method="test", params={}
        )
        assert approval_store.cancel(action_id) is True
        assert len(approval_store.list_pending()) == 0

    def test_cancel_nonexistent(self, approval_store):
        assert approval_store.cancel("no-such-id") is False


class TestApprovalStoreCleanup:
    """Test expired action cleanup."""

    def test_cleanup_removes_expired(self):
        store = ApprovalStore(ttl_seconds=0)
        store.create_pending(method="a", params={})
        store.create_pending(method="b", params={})
        time.sleep(0.01)
        removed = store.cleanup_expired()
        assert removed == 2
        assert len(store.list_pending()) == 0

    def test_cleanup_preserves_valid(self, approval_store):
        approval_store.create_pending(method="valid", params={})
        removed = approval_store.cleanup_expired()
        assert removed == 0
        assert len(approval_store.list_pending()) == 1


class TestApprovalStoreListPending:
    """Test listing pending actions."""

    def test_list_truncates_long_proposals(self, approval_store):
        long_proposal = "X" * 3000
        approval_store.create_pending(
            method="test", params={}, proposal=long_proposal
        )
        pending = approval_store.list_pending()
        assert len(pending[0]["proposal"]) < 3000
        assert pending[0]["proposal"].endswith("[truncated]")

    def test_list_includes_required_fields(self, approval_store):
        approval_store.create_pending(method="ai_write", params={"path": "/f"})
        pending = approval_store.list_pending()
        item = pending[0]
        assert "action_id" in item
        assert "method" in item
        assert "created_at" in item
        assert "expiry" in item
        assert "proposal" in item

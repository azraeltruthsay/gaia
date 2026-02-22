"""
Approval Store for MCP sensitive actions.

Manages pending actions that require human approval before execution.
"""

import random
import string
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from gaia_common.utils import get_logger

logger = get_logger(__name__)


class ApprovalStore:
    """
    In-memory store for pending actions that require human approval.

    Each pending action is stored as:
        action_id -> {
            "method": str,
            "params": dict,
            "challenge": str,
            "created_at": float,
            "expiry": float,
            "proposal": str,
        }

    The approval workflow:
    1. Action is created with a challenge code (e.g., "ABCDE")
    2. Human reviews the proposal and provides the reversed challenge ("EDCBA")
    3. If challenge matches, action is approved and executed
    """

    def __init__(self, ttl_seconds: int = 900):
        """
        Initialize the approval store.

        Args:
            ttl_seconds: Time-to-live for pending actions (default: 15 minutes)
        """
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def _gen_challenge(self) -> str:
        """Generate a 5-character alphabetic challenge code."""
        return ''.join(random.choice(string.ascii_uppercase) for _ in range(5))

    def create_pending(
        self,
        method: str,
        params: Dict[str, Any],
        proposal: Optional[str] = None,
        allow_pending: bool = False # This param is from request_approval, not here. Removing it.
    ) -> Tuple[str, str, float, float]:
        """
        Create a pending action awaiting approval.

        Args:
            method: The tool method name
            params: The tool parameters
            proposal: Optional human-readable description of the action

        Returns:
            Tuple of (action_id, challenge, created_at, expiry)
        """
        with self._lock:
            action_id = str(uuid.uuid4())
            challenge = self._gen_challenge()
            now = time.time()
            expiry = now + self._ttl

            # Generate proposal if not provided
            if proposal is None:
                try:
                    import json
                    proposal = json.dumps(params or {}, indent=2, ensure_ascii=False)
                except Exception:
                    proposal = str(params)

            self._store[action_id] = {
                "method": method,
                "params": params,
                "challenge": challenge,
                "created_at": now,
                "expiry": expiry,
                "proposal": proposal,
            }

            logger.info(
                f"Created pending action {action_id} method={method} "
                f"challenge={challenge} expiry={datetime.utcfromtimestamp(expiry).isoformat()}"
            )

            return action_id, challenge, now, expiry

    def list_pending(self) -> List[Dict[str, Any]]:
        """
        List all pending actions.

        Returns:
            List of pending action summaries
        """
        with self._lock:
            now = time.time()
            result = []

            for action_id, item in list(self._store.items()):
                # Skip expired
                if now > item["expiry"]:
                    del self._store[action_id]
                    continue

                # Truncate long proposals
                proposal = item.get("proposal", "")
                if len(proposal) > 2000:
                    proposal = proposal[:2000] + "\n... [truncated]"

                result.append({
                    "action_id": action_id,
                    "method": item["method"],
                    "created_at": datetime.utcfromtimestamp(item["created_at"]).isoformat(),
                    "expiry": datetime.utcfromtimestamp(item["expiry"]).isoformat(),
                    "proposal": proposal,
                })

            return result

    def approve(self, action_id: str, provided_challenge: str) -> Dict[str, Any]:
        """
        Approve a pending action.

        Args:
            action_id: The action ID to approve
            provided_challenge: The reversed challenge code

        Returns:
            The approved action details

        Raises:
            KeyError: If action not found or expired
            ValueError: If challenge is invalid
        """
        with self._lock:
            item = self._store.get(action_id)
            if not item:
                raise KeyError("action_id not found or expired")

            if time.time() > item["expiry"]:
                del self._store[action_id]
                raise KeyError("action expired")

            # Expected: reversed challenge
            expected = item["challenge"][::-1]
            if provided_challenge != expected:
                raise ValueError("invalid approval challenge")

            # Remove from pending
            payload = {
                "method": item["method"],
                "params": item["params"],
                "created_at": item["created_at"],
            }
            del self._store[action_id]

            logger.info(f"Approved action {action_id} method={item['method']}")
            return payload

    def cancel(self, action_id: str) -> bool:
        """
        Cancel a pending action.

        Args:
            action_id: The action ID to cancel

        Returns:
            True if cancelled, False if not found
        """
        with self._lock:
            if action_id in self._store:
                del self._store[action_id]
                logger.info(f"Cancelled action {action_id}")
                return True
            return False

    def cleanup_expired(self) -> int:
        """
        Remove all expired actions.

        Returns:
            Number of expired actions removed
        """
        with self._lock:
            now = time.time()
            expired = [
                aid for aid, item in self._store.items()
                if now > item["expiry"]
            ]
            for aid in expired:
                del self._store[aid]
            if expired:
                logger.info(f"Cleaned up {len(expired)} expired actions")
            return len(expired)

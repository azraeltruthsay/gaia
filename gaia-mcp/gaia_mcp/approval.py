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

    def validate_against_blast_shield(self, method: str, params: Dict[str, Any]):
        """
        VouchCore Pattern: Deterministic pre-flight safety check.
        Raises ValueError if action is forbidden regardless of LLM reasoning.

        Hardened v2 (2026-04-09): Uses shlex tokenization and regex word
        boundaries instead of substring matching. Resolves symlinks and
        normalizes paths before checking blocked list.
        """
        if method == "run_shell":
            cmd = str(params.get("command", ""))
            self._validate_shell_command(cmd)

        if method in ("write_file", "ai_write", "replace"):
            path = str(params.get("path", ""))
            self._validate_path(path)

        # Also validate paths in shell commands
        if method == "run_shell":
            cmd = str(params.get("command", ""))
            self._validate_shell_paths(cmd)

    def _validate_shell_command(self, cmd: str):
        """Validate shell commands using tokenization, not substring matching."""
        import re

        # Normalize: collapse whitespace, strip
        normalized = " ".join(cmd.split()).lower()

        # Check for command chaining/injection operators
        chain_operators = ["&&", "||", ";", "|", "$(", "`"]
        for op in chain_operators:
            if op in normalized:
                # Split on the operator and validate each part
                parts = re.split(r'[;&|`$()]+', normalized)
                for part in parts:
                    part = part.strip()
                    if part:
                        self._check_forbidden_command(part)
                return

        self._check_forbidden_command(normalized)

    def _check_forbidden_command(self, cmd: str):
        """Check a single (non-chained) command against forbidden patterns."""
        import re

        # Tokenize safely — fall back to split on failure
        try:
            import shlex
            tokens = shlex.split(cmd)
        except ValueError:
            tokens = cmd.split()

        if not tokens:
            return

        base_cmd = tokens[0]
        flags = set(tokens[1:])

        # Destructive file operations (word-boundary matching)
        if base_cmd == "rm" and ({"-rf", "-fr", "-r", "--recursive"} & flags):
            self._block(cmd, "rm with recursive flag")
        if base_cmd in ("rmdir",) and "/" in " ".join(tokens[1:]):
            self._block(cmd, "rmdir on root path")

        # Privilege escalation
        if base_cmd in ("sudo", "su", "doas", "pkexec"):
            self._block(cmd, f"privilege escalation ({base_cmd})")

        # Disk/filesystem destruction
        if base_cmd in ("mkfs", "mkfs.ext4", "mkfs.xfs", "wipefs", "fdisk", "parted"):
            self._block(cmd, f"disk operation ({base_cmd})")
        if base_cmd == "dd" and any("of=/dev/" in t for t in tokens):
            self._block(cmd, "dd to device")

        # Permission escalation
        if base_cmd in ("chmod", "chown", "setfacl"):
            if base_cmd == "chmod" and any(re.match(r'^[0-7]*[4267][0-7]*$', t) for t in tokens[1:]):
                self._block(cmd, "chmod with dangerous permissions (setuid/setgid/world-writable)")
            if base_cmd == "chmod" and ({"+s", "u+s", "g+s", "a+rwx"} & flags):
                self._block(cmd, "chmod setuid/setgid/world-writable")
            if base_cmd == "chown" and "root" in " ".join(tokens[1:]):
                self._block(cmd, "chown to root")

        # Container/namespace escape
        if base_cmd in ("nsenter", "chroot", "unshare", "pivot_root"):
            self._block(cmd, f"container escape ({base_cmd})")

        # Remote code execution
        if base_cmd in ("curl", "wget") and any(p in cmd for p in ["| bash", "|bash", "| sh", "|sh"]):
            self._block(cmd, "remote code execution via pipe")

        # Credential access
        if base_cmd in ("cat", "head", "tail", "less", "more"):
            sensitive_files = ["/etc/shadow", "/etc/passwd", "/run/secrets", "/.ssh/"]
            for sf in sensitive_files:
                if any(sf in t for t in tokens[1:]):
                    self._block(cmd, f"credential access ({sf})")

        # Network listeners
        if base_cmd in ("nc", "ncat", "netcat", "socat") and any(l in flags for l in ["-l", "-lp", "--listen"]):
            self._block(cmd, "network listener")

        # Arbitrary code execution (bypasses whitelist)
        if base_cmd in ("python", "python3", "node", "ruby", "perl") and ({"-c", "-e"} & flags):
            self._block(cmd, f"arbitrary code execution ({base_cmd})")

        # Find with exec
        if base_cmd == "find" and ({"-exec", "-execdir", "-delete"} & flags):
            self._block(cmd, "find with exec/delete")

    def _validate_path(self, path: str):
        """Validate file paths using normalization and symlink resolution."""
        import os

        if not path:
            return

        # Block relative traversal
        if ".." in path:
            self._block(path, "path traversal (..)")

        # Normalize and resolve symlinks
        try:
            resolved = os.path.realpath(path)
        except (OSError, ValueError):
            resolved = os.path.normpath(path)

        # Also normalize the raw path (without symlink resolution)
        normed = os.path.normpath(path)

        blocked_paths = [
            "/etc", "/boot", "/.ssh", "/run/secrets",
            "/proc", "/sys", "/dev", "/root",
            "/sys/fs/cgroup", "/dev/shm", "/var/spool",
        ]
        for bp in blocked_paths:
            if resolved.startswith(bp) or normed.startswith(bp):
                logger.critical("BLAST SHIELD: Blocked path (resolved=%s, raw=%s, blocked=%s)", resolved, path, bp)
                raise ValueError(f"Blast Shield: Writing to system/sensitive path '{bp}' is forbidden.")

    def _validate_shell_paths(self, cmd: str):
        """Extract and validate paths from shell commands."""
        import re
        # Find anything that looks like an absolute path
        paths = re.findall(r'(/[a-zA-Z0-9_./-]+)', cmd)
        for p in paths:
            try:
                self._validate_path(p)
            except ValueError:
                raise

    def _block(self, cmd: str, reason: str):
        """Log and raise for a blocked command."""
        logger.critical("BLAST SHIELD: Blocked command — reason=%s cmd=%s", reason, cmd[:200])
        raise ValueError(f"Blast Shield: {reason}")

    def create_pending(
        self,
        method: str,
        params: Dict[str, Any],
        proposal: Optional[str] = None,
        allow_pending: bool = False
    ) -> Tuple[str, str, float, float]:
        """
        Create a pending action awaiting approval.
        """
        # ── 🛡️ BLAST SHIELD CHECK ──
        self.validate_against_blast_shield(method, params)

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
                "allow_pending": allow_pending
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

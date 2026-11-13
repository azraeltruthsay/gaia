"""
Approval Challenge -- TOTP-based human verification (GAIA_Project-96xm).

Used to generate a random string and ask the human to provide the reverse
before approving a destructive or significant action (writing to
candidates, promoting code, executing plans). That confirmed nothing
beyond "can read this channel" -- one factor, not two, since anyone who
can read the prompt can also reverse a 5-character string.

The human now confirms with their own authenticator app's current code
instead -- the same mechanism, and the same enrolled secret, as gaia-mcp's
ApprovalStore (both protect GAIA's own sensitive actions; a separate,
deliberately independent secret protects a different thing entirely --
gaia-ops-broker's gate on Claude Code's own host privilege).

Usage:
    from gaia_common.utils.approval_challenge import (
        create_challenge, verify_challenge, get_pending_challenge
    )

    # Create a pending approval
    challenge = create_challenge(action="write_candidates", context="4 files")
    # challenge = {"challenge_id": "abc12", "action": ..., "context": ..., ...}

    # Present to user: format_challenge_prompt(challenge)

    # Verify the human's current authenticator code
    result = verify_challenge("abc12", user_input="482913")
    # result = {"approved": True, ...}
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

from gaia_common.utils.totp import load_totp_verifier

logger = logging.getLogger("GAIA.ApprovalChallenge")

_CHALLENGES_FILE = Path(os.environ.get("SHARED_DIR", "/shared")) / "approval_challenges.json"
_CHALLENGE_TTL = 300  # 5 minutes

_totp_verifier = None  # lazily loaded, module-level cache


def _get_verifier():
    global _totp_verifier
    if _totp_verifier is None:
        _totp_verifier = load_totp_verifier(
            "/run/secrets/gaia_mcp_totp_secret",
            env_var="GAIA_MCP_TOTP_SECRET_FILE",
        )
    return _totp_verifier


def create_challenge(
    action: str,
    context: str = "",
) -> Dict:
    """
    Create a pending approval awaiting the human's TOTP code.

    Args:
        action: What's being approved (e.g., "write_candidates", "promote")
        context: Additional context (e.g., "4 files to candidates/")

    Returns:
        Challenge dict with metadata (no code/reverse -- there's nothing
        server-generated to reveal anymore).
    """
    challenge = {
        "challenge_id": uuid.uuid4().hex[:10],
        "action": action,
        "context": context,
        "created_at": time.time(),
        "expires_at": time.time() + _CHALLENGE_TTL,
        "verified": False,
    }

    _save_challenge(challenge)
    logger.info("Approval challenge created: %s for action=%s", challenge["challenge_id"], action)

    return challenge


def verify_challenge(challenge_id: str, user_input: str) -> Dict:
    """
    Verify the human's current TOTP code against a pending approval.

    Args:
        challenge_id: The challenge ID
        user_input: The human's current 6-digit authenticator code

    Returns:
        {"approved": bool, "reason": str}
    """
    challenges = _load_challenges()
    challenge = challenges.get(challenge_id)

    if not challenge:
        return {"approved": False, "reason": "Challenge not found or expired"}

    if time.time() > challenge.get("expires_at", 0):
        _remove_challenge(challenge_id)
        return {"approved": False, "reason": "Challenge expired (5 minute limit)"}

    if challenge.get("verified"):
        return {"approved": False, "reason": "Challenge already used"}

    verifier = _get_verifier()
    if not verifier or not verifier.verify((user_input or "").strip()):
        logger.warning("Approval challenge %s failed TOTP check for action=%s",
                       challenge_id, challenge["action"])
        return {"approved": False, "reason": "Invalid or expired authenticator code"}

    challenge["verified"] = True
    _save_challenge(challenge)
    logger.info("Approval challenge %s verified for action=%s", challenge_id, challenge["action"])
    return {"approved": True, "action": challenge["action"], "context": challenge["context"]}


def get_pending_challenge(action: Optional[str] = None) -> Optional[Dict]:
    """Get the most recent pending (unverified, unexpired) challenge."""
    challenges = _load_challenges()
    now = time.time()
    for cid, c in sorted(challenges.items(), key=lambda x: x[1].get("created_at", 0), reverse=True):
        if c.get("verified") or now > c.get("expires_at", 0):
            continue
        if action and c.get("action") != action:
            continue
        return c
    return None


def format_challenge_prompt(challenge: Dict) -> str:
    """Format a pending approval for display to the user."""
    action = challenge.get("action", "this action")
    context = challenge.get("context", "")

    lines = [
        f"**Approval required for: {action}**",
    ]
    if context:
        lines.append(f"*{context}*")
    lines.append("")
    lines.append("To approve, reply with your current 6-digit authenticator code.")
    lines.append("(Request expires in 5 minutes)")

    return "\n".join(lines)


def cleanup_expired():
    """Remove expired challenges."""
    challenges = _load_challenges()
    now = time.time()
    expired = [cid for cid, c in challenges.items() if now > c.get("expires_at", 0)]
    for cid in expired:
        del challenges[cid]
    if expired:
        _save_challenges(challenges)
        logger.debug("Cleaned up %d expired approval challenges", len(expired))


# -- Persistence --

def _load_challenges() -> Dict:
    try:
        if _CHALLENGES_FILE.exists():
            return json.loads(_CHALLENGES_FILE.read_text())
    except Exception:
        pass
    return {}


def _save_challenge(challenge: Dict):
    challenges = _load_challenges()
    challenges[challenge["challenge_id"]] = challenge
    _save_challenges(challenges)


def _save_challenges(challenges: Dict):
    _CHALLENGES_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CHALLENGES_FILE.write_text(json.dumps(challenges, indent=2))


def _remove_challenge(challenge_id: str):
    challenges = _load_challenges()
    challenges.pop(challenge_id, None)
    _save_challenges(challenges)

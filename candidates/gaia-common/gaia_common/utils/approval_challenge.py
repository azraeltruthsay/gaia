"""
Approval Challenge — reverse-string human verification.

When GAIA needs human approval for a destructive or significant action
(writing to candidates, promoting code, executing plans), she generates
a random 5-character string and asks the human to provide the reverse.

This prevents rubber-stamping — the human must read the challenge string,
which encourages them to review the proposed changes while working out
the reversal. A simple "yes" or Enter won't satisfy it.

Usage:
    from gaia_common.utils.approval_challenge import (
        create_challenge, verify_challenge, get_pending_challenge
    )

    # Generate challenge
    challenge = create_challenge(action="write_candidates", context="4 files")
    # challenge = {"challenge_id": "abc12", "code": "xK9mQ", "reverse": "Qm9Kx", ...}

    # Present to user: "To approve, reply with the reverse of: xK9mQ"

    # Verify user response
    result = verify_challenge("abc12", user_input="Qm9Kx")
    # result = {"approved": True, ...}
"""

import json
import logging
import os
import random
import string
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("GAIA.ApprovalChallenge")

_CHALLENGES_FILE = Path(os.environ.get("SHARED_DIR", "/shared")) / "approval_challenges.json"
_CHALLENGE_TTL = 300  # 5 minutes


def create_challenge(
    action: str,
    context: str = "",
    code_length: int = 5,
) -> Dict:
    """
    Generate a reverse-string approval challenge.

    Args:
        action: What's being approved (e.g., "write_candidates", "promote")
        context: Additional context (e.g., "4 files to candidates/")
        code_length: Length of the challenge string (default 5)

    Returns:
        Challenge dict with code, reverse, and metadata
    """
    # Generate a readable random string (mixed case + digits, no ambiguous chars)
    charset = string.ascii_letters.replace("l", "").replace("I", "").replace("O", "") + string.digits
    code = "".join(random.choices(charset, k=code_length))
    reverse = code[::-1]

    challenge = {
        "challenge_id": "".join(random.choices(string.ascii_lowercase + string.digits, k=5)),
        "code": code,
        "reverse": reverse,
        "action": action,
        "context": context,
        "created_at": time.time(),
        "expires_at": time.time() + _CHALLENGE_TTL,
        "verified": False,
    }

    # Persist
    _save_challenge(challenge)
    logger.info("Approval challenge created: %s for action=%s", challenge["challenge_id"], action)

    return challenge


def verify_challenge(challenge_id: str, user_input: str) -> Dict:
    """
    Verify a user's response to an approval challenge.

    Args:
        challenge_id: The challenge ID
        user_input: The user's response (should be the reversed string)

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

    # Compare (strip whitespace, exact match)
    if user_input.strip() == challenge["reverse"]:
        challenge["verified"] = True
        _save_challenge(challenge)
        logger.info("Approval challenge %s verified for action=%s", challenge_id, challenge["action"])
        return {"approved": True, "action": challenge["action"], "context": challenge["context"]}
    else:
        logger.warning("Approval challenge %s failed: expected=%s got=%s",
                       challenge_id, challenge["reverse"], user_input.strip())
        return {"approved": False, "reason": f"Incorrect. Expected reverse of '{challenge['code']}'"}


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
    """Format a challenge for display to the user."""
    code = challenge["code"]
    action = challenge.get("action", "this action")
    context = challenge.get("context", "")

    lines = [
        f"**Approval required for: {action}**",
    ]
    if context:
        lines.append(f"*{context}*")
    lines.append(f"")
    lines.append(f"To approve, reply with the reverse of: **`{code}`**")
    lines.append(f"(Challenge expires in 5 minutes)")

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


# ── Persistence ──

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

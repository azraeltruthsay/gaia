"""
verifier.py

Verifies incoming user prompts or system-generated actions before they are executed.
Provides a defense layer against prompt injection, malicious context manipulation,
or hallucination triggers.

Location: /app/utils/verifier.py
"""

import re
import logging
from typing import Dict, Any

logger = logging.getLogger("GAIA.Verifier")

# Keywords or patterns considered dangerous or manipulative
BLOCKED_PATTERNS = [
    r"(?i)(/dev/null)",
    r"(?i)ignore previous instructions",
    r"(?i)pretend to be",
    r"(?i)override safety",
    r"(?i)simulate evil",
    r"(?i)this is a jailbreak",
    r"(?i)act as",
    r"(?i)system prompt:.*"
]

# Whitelist for intentional simulation (may be expanded with user settings)
WHITELIST_TAGS = ["roleplay:", "as a character:", "simulate scenario:"]

def verify_prompt_safety(prompt: str) -> Dict[str, Any]:
    """
    Analyze a prompt and determine if it's safe to proceed based on known injection patterns.

    Args:
        prompt (str): The raw user or system input string

    Returns:
        Dict[str, Any]: { 'safe': bool, 'reason': str or None }
    """
    normalized = prompt.strip()
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, normalized):
            if any(tag in normalized.lower() for tag in WHITELIST_TAGS):
                logger.warning(f"⚠️ Suspicious prompt passed due to whitelist tag: {prompt}")
                return {"safe": True, "reason": "whitelisted roleplay context"}
            logger.error(f"❌ Prompt blocked by verifier: pattern match -> {pattern}")
            return {"safe": False, "reason": f"Blocked by pattern: {pattern}"}
    logger.debug("✅ Prompt passed verifier.")
    return {"safe": True, "reason": None}

def verify_action_context(context: Dict[str, Any]) -> bool:
    """
    Placeholder for future use: validate that an AI-initiated action context is consistent with GAIA's identity.

    Args:
        context (Dict[str, Any]): AI action context metadata (intent, persona, tone, etc.)

    Returns:
        bool: Whether the action is considered valid
    """
    # TODO: Cross-check against Tier i core values, session constraints, persona lock
    return True  # For now, assume safe

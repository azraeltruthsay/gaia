"""
History Review — Pre-injection audit of conversation history.

Before history is injected into the LLM prompt, each assistant message
is checked for epistemic violations:
  - Fabricated file paths (citing files that were never read via tool call)
  - Fabricated quotes (blockquote formatting claiming document sources)
  - Confidently wrong factual claims flagged by prior user corrections
  - Hallucinated URLs/links with no real source

Messages that fail are either:
  - REDACTED: replaced with a short note explaining why
  - EDITED: trimmed to remove the offending section with a parenthetical

This is a fast, rule-based pass (no LLM call). It runs on the history
list before it enters the CognitionPacket, so all downstream consumers
(sliding window, session RAG, prompt builder) see clean history.

Config key: HISTORY_REVIEW in gaia_constants.json
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("GAIA.HistoryReview")

# ─── Patterns that indicate fabricated content ───────────────────────────

# File paths that look like citations but weren't tool-called
_FAKE_PATH_RE = re.compile(
    r"(?:read_file|from|source|verified in|confirmed in|see)\s*[:>]?\s*"
    r"[`\[]*(?:knowledge|/knowledge|/code|/app|/logs|/sandbox)"
    r"[/\w._-]+\.(?:md|json|pdf|txt|py|yaml|yml)\b",
    re.IGNORECASE,
)

# Blockquotes claiming to be from a document
_FAKE_QUOTE_RE = re.compile(
    r'>\s*[*"].*?[*"]\s*$',
    re.MULTILINE,
)

# Markdown links to fabricated knowledge base URLs
_FAKE_LINK_RE = re.compile(
    r"\[.*?\]\(https?://knowledge\.base/.*?\)",
    re.IGNORECASE,
)

# Section references like "[Section 3.1](https://...)" or "(Line 42, ...)"
_FAKE_SECTION_REF_RE = re.compile(
    r"\[(?:Section|Line|Page)\s+\d+[^]]*\]\(https?://[^)]+\)",
    re.IGNORECASE,
)

# Claims of verifying/confirming from files with specific line numbers
_FAKE_VERIFICATION_RE = re.compile(
    r"(?:verified|confirmed|source|evidence)\s+(?:in|from|via)\s+\[?"
    r"(?:`[^`]+`|knowledge/\S+)",
    re.IGNORECASE,
)

# User correction patterns — the message AFTER these is probably an
# acknowledgment of error; we should keep those (they show self-correction).
_USER_CORRECTION_RE = re.compile(
    r"(?:you'?re\s+(?:wrong|incorrect|mixing|confusing|hallucinating|confabulating))"
    r"|(?:that'?s\s+(?:not\s+(?:true|correct|right)|wrong|incorrect))"
    r"|(?:I\s+don'?t\s+think\s+(?:that'?s|the)\s+)"
    r"|(?:actually,?\s+(?:that'?s|it'?s)\s+not)",
    re.IGNORECASE,
)


def _count_violations(text: str) -> Tuple[int, List[str]]:
    """Count epistemic violations in an assistant message.

    Returns (violation_count, list_of_reasons).
    """
    violations = []

    # Check for fabricated file path citations
    fake_paths = _FAKE_PATH_RE.findall(text)
    if fake_paths:
        violations.append(f"cited {len(fake_paths)} unverified file path(s)")

    # Check for fabricated blockquotes
    fake_quotes = _FAKE_QUOTE_RE.findall(text)
    if fake_quotes:
        violations.append(f"contains {len(fake_quotes)} fabricated blockquote(s)")

    # Check for fake knowledge base links
    fake_links = _FAKE_LINK_RE.findall(text)
    if fake_links:
        violations.append(f"contains {len(fake_links)} fabricated link(s)")

    # Check for fake section references
    fake_sections = _FAKE_SECTION_REF_RE.findall(text)
    if fake_sections:
        violations.append(f"contains {len(fake_sections)} fabricated section ref(s)")

    # Check for ungrounded verification claims
    fake_verifications = _FAKE_VERIFICATION_RE.findall(text)
    if len(fake_verifications) > 1:
        violations.append(f"makes {len(fake_verifications)} unverified source claims")

    return len(violations), violations


def _is_user_correction(text: str) -> bool:
    """Check if a user message is correcting the assistant."""
    return bool(_USER_CORRECTION_RE.search(text))


def _redact_message(original: str, reasons: List[str]) -> str:
    """Replace an assistant message with a redaction note."""
    reason_str = "; ".join(reasons)
    return (
        f"(This earlier response was redacted during history review "
        f"because it {reason_str}. The information was unreliable.)"
    )


def _build_correction_summary(
    user_msg: str, assistant_msg: str, reasons: List[str]
) -> Optional[str]:
    """When a user corrected the assistant and the assistant acknowledged,
    compress the pair into a brief correction note.
    """
    # Keep the assistant's acknowledgment but trim the fabricated details
    # Look for the acknowledgment pattern
    ack_patterns = [
        r"(?:you'?re\s+(?:absolutely\s+)?right|I\s+(?:made|had)\s+(?:a\s+)?(?:critical\s+)?error"
        r"|that\s+was\s+my\s+mistake|I\s+apologize|I\s+sincerely\s+appreciate)",
    ]
    for pat in ack_patterns:
        m = re.search(pat, assistant_msg, re.IGNORECASE)
        if m:
            # Found acknowledgment — return a compact summary
            return (
                f"(Earlier in this conversation, the assistant made errors "
                f"that {'; '.join(reasons)}. The user corrected these errors "
                f"and the assistant acknowledged the mistakes.)"
            )
    return None


def review_history(
    history: List[Dict[str, str]],
    config: Optional[dict] = None,
    session_id: str = "",
) -> List[Dict[str, str]]:
    """Review and clean conversation history before prompt injection.

    Args:
        history: List of {"role": str, "content": str} messages.
        config: HISTORY_REVIEW config dict from gaia_constants.json.
        session_id: For logging.

    Returns:
        Cleaned history list (same format, potentially with redacted messages).
    """
    cfg = config or {}
    if not cfg.get("enabled", True):
        return history

    violation_threshold = cfg.get("violation_threshold", 2)
    max_history_len = cfg.get("max_messages", 20)

    if not history:
        return history

    cleaned: List[Dict[str, str]] = []
    total_redacted = 0
    total_compressed = 0
    i = 0

    while i < len(history):
        msg = history[i]
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role != "assistant":
            # Check if this user message is a correction
            if role == "user" and _is_user_correction(content):
                # Look ahead: if the next message is an assistant acknowledgment
                # with violations, compress the pair
                if i + 1 < len(history) and history[i + 1].get("role") == "assistant":
                    next_content = history[i + 1].get("content", "")
                    v_count, v_reasons = _count_violations(next_content)
                    if v_count >= 1:
                        summary = _build_correction_summary(content, next_content, v_reasons)
                        if summary:
                            # Replace the pair with a single assistant note
                            cleaned.append({"role": "user", "content": content})
                            cleaned.append({"role": "assistant", "content": summary})
                            total_compressed += 1
                            i += 2  # skip both messages
                            continue

            cleaned.append(msg)
            i += 1
            continue

        # Assistant message — check for violations
        v_count, v_reasons = _count_violations(content)

        if v_count >= violation_threshold:
            # Heavily violated — redact entirely
            redacted = _redact_message(content, v_reasons)
            cleaned.append({"role": role, "content": redacted})
            total_redacted += 1
            logger.warning(
                "HistoryReview[%s]: redacted message %d/%d (%d violations: %s)",
                session_id, i, len(history), v_count, "; ".join(v_reasons),
            )
        elif v_count >= 1:
            # Minor violations — keep but add a caveat
            caveat = (
                f"\n\n(Note: this earlier response may contain unverified claims. "
                f"Detected issues: {'; '.join(v_reasons)}.)"
            )
            cleaned.append({"role": role, "content": content + caveat})
            logger.info(
                "HistoryReview[%s]: annotated message %d/%d (%d violations)",
                session_id, i, len(history), v_count,
            )
        else:
            # Clean message — pass through
            cleaned.append(msg)

        i += 1

    # Trim to max length (keep most recent)
    if len(cleaned) > max_history_len:
        cleaned = cleaned[-max_history_len:]

    if total_redacted or total_compressed:
        logger.warning(
            "HistoryReview[%s]: %d redacted, %d compressed out of %d messages",
            session_id, total_redacted, total_compressed, len(history),
        )

    return cleaned

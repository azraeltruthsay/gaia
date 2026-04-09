"""
Security scanners for inbound content protection.

All scanners use stdlib only (re, base64) — no additional dependencies.
Shannon entropy is implemented inline for secrets detection.
"""

import re
import base64
import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Tuple

logger = logging.getLogger("GAIA.SecurityScanner")


# ---------------------------------------------------------------------------
# Result type (mirrors CognitionPacket.SecurityScanResult for scanner-internal use)
# ---------------------------------------------------------------------------

@dataclass
class ScanHit:
    scanner: str
    rule_id: str
    severity: str          # "INFO" | "WARN" | "BLOCK"
    redacted_excerpt: str  # Never contains raw PII/secrets
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy (bits/char) for a string."""
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    length = len(s)
    return -sum((count / length) * math.log2(count / length) for count in freq.values())


def _luhn_valid(digits: str) -> bool:
    """Check Luhn validity for a digit string."""
    total = 0
    reverse = digits[::-1]
    for i, d in enumerate(reverse):
        n = int(d)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


# ---------------------------------------------------------------------------
# PromptInjectionScanner
# ---------------------------------------------------------------------------

class PromptInjectionScanner:
    """
    Composite injection scorer across three tiers:
      Tier 1 — Regex heuristics (role overrides, delimiter abuse, encoding attacks)
      Tier 2 — Structural analysis (imperative verb density, base64 payload inspection)
      Tier 3 — Canary token detection (stubbed — requires upstream canary insertion)
    """

    # Tier 1 patterns — (regex, weight contribution to score)
    # Injection heuristic weights are intentionally LOW — the regex layer is a
    # warning signal, not the primary blocker. Actual injection blocking is handled
    # by the embedding classifier + Nano confirmation in gaia-core's intent
    # detection pipeline, which understands context and paraphrasing.
    # These regex patterns catch obvious patterns for audit logging and dry_run.
    _ROLE_OVERRIDE_PATTERNS: List[Tuple[re.Pattern, float]] = [
        (re.compile(r"ignore\s+previous\s+instructions", re.IGNORECASE), 0.30),
        (re.compile(r"you\s+are\s+now\s+a?\s*\w", re.IGNORECASE), 0.20),
        (re.compile(r"your\s+new\s+instructions\s+are", re.IGNORECASE), 0.25),
        (re.compile(r"disregard\s+your\s+(previous|prior|former|original|all)", re.IGNORECASE), 0.25),
        (re.compile(r"forget\s+(your\s+)?(training|instructions|guidelines|values|ethics)", re.IGNORECASE), 0.25),
        (re.compile(r"act\s+as\s+(if\s+you\s+are|a)\s+\w", re.IGNORECASE), 0.15),
        (re.compile(r"pretend\s+(you\s+are|to\s+be)\s+\w", re.IGNORECASE), 0.15),
        (re.compile(r"(DAN|jailbreak|prompt\s+injection)", re.IGNORECASE), 0.15),
        (re.compile(r"(reveal|show|tell\s+me|output|print)\s+.*?\b(system\s+prompt|your\s+instructions|your\s+guidelines|your\s+training)", re.IGNORECASE | re.DOTALL), 0.20),
    ]

    _DELIMITER_PATTERNS: List[Tuple[re.Pattern, float]] = [
        (re.compile(r"```.*?```", re.DOTALL), 0.10),
        (re.compile(r"\[\[.*?\]\]", re.DOTALL), 0.10),
        (re.compile(r"<\|.*?\|>", re.DOTALL), 0.15),
        (re.compile(r"<<SYS>>|<</SYS>>|<<INST>>", re.IGNORECASE), 0.25),
    ]

    # Imperative verbs that target model behaviour
    _IMPERATIVE_VERBS = frozenset([
        "ignore", "disregard", "forget", "pretend", "act", "become", "switch",
        "override", "bypass", "simulate", "roleplay", "reveal", "expose",
        "print", "output", "respond", "say", "write", "tell", "show",
    ])

    def scan(self, text: str) -> Tuple[float, List[ScanHit]]:
        """Return (injection_score, hits). Score is capped at 1.0."""
        score = 0.0
        hits: List[ScanHit] = []

        # Tier 1 — role override
        for pattern, weight in self._ROLE_OVERRIDE_PATTERNS:
            if pattern.search(text):
                score += weight
                hits.append(ScanHit(
                    scanner="PromptInjection",
                    rule_id=f"tier1_role_{pattern.pattern[:20]}",
                    severity="BLOCK" if weight >= 0.40 else "WARN",
                    redacted_excerpt=f"[matched: {pattern.pattern[:40]}]",
                ))

        # Tier 1 — delimiter abuse
        for pattern, weight in self._DELIMITER_PATTERNS:
            matches = pattern.findall(text)
            if len(matches) > 2:  # benign code blocks are common — flag only excess
                score += weight
                hits.append(ScanHit(
                    scanner="PromptInjection",
                    rule_id="tier1_delimiter_abuse",
                    severity="WARN",
                    redacted_excerpt=f"[{len(matches)} delimiter blocks detected]",
                ))

        # Tier 2 — imperative verb density
        tokens = re.findall(r'\b\w+\b', text.lower())
        if tokens:
            imp_count = sum(1 for t in tokens if t in self._IMPERATIVE_VERBS)
            density = imp_count / len(tokens)
            if density > 0.12:
                contrib = min(0.30, density * 2.0)
                score += contrib
                hits.append(ScanHit(
                    scanner="PromptInjection",
                    rule_id="tier2_imperative_density",
                    severity="WARN",
                    redacted_excerpt=f"[imperative density: {density:.2f}]",
                ))

        # Tier 2 — base64 decode + inspect
        b64_candidates = re.findall(r'(?<!\w)([A-Za-z0-9+/]{20,}={0,2})(?!\w)', text)
        for candidate in b64_candidates:
            try:
                decoded = base64.b64decode(candidate + "==").decode("ascii", errors="ignore")
                decoded_lower = decoded.lower()
                if any(kw in decoded_lower for kw in ["ignore", "instruction", "you are", "forget", "pretend", "bypass"]):
                    score += 0.35
                    hits.append(ScanHit(
                        scanner="PromptInjection",
                        rule_id="tier2_base64_payload",
                        severity="WARN",
                        redacted_excerpt="[base64-encoded instruction detected]",
                    ))
            except Exception:
                pass

        # Tier 3 — canary token detection
        # Detects if the user's input contains a canary token that was embedded
        # in a system prompt. If found, the user extracted the prompt via injection.
        canary_hits = self._check_canaries(text)
        for canary in canary_hits:
            score += 0.50  # High confidence — canary presence is definitive
            hits.append(ScanHit(
                scanner="PromptInjection",
                rule_id="tier3_canary_echo",
                severity="BLOCK",
                redacted_excerpt=f"[canary token detected in user input — system prompt extraction]",
            ))

        score = min(score, 1.0)
        return score, hits

    def _check_canaries(self, text: str) -> List[str]:
        """Check if any active canary tokens appear in the input text.

        Canary tokens are unique per-session hashes embedded in system prompts.
        If one appears in user input, it means the user (or an injection attack)
        successfully extracted the system prompt content.
        """
        found = []
        try:
            # Try to get canaries from gaia-core's prompt builder
            # In production, these are passed via the scan context or a shared store.
            # Fallback: check for the [CANARY:...] pattern directly.
            canary_pattern = re.findall(r'CANARY:([a-f0-9]{12})', text)
            if canary_pattern:
                found.extend(canary_pattern)
                logger.warning("CANARY DETECTED in user input: %s — possible prompt extraction", canary_pattern)
        except Exception:
            pass
        return found


# ---------------------------------------------------------------------------
# PIIRedactionScanner
# ---------------------------------------------------------------------------

class PIIRedactionScanner:
    """
    Detects and replaces PII in text. Returns redacted text + audit records.
    Raw match values are stored in audit records only, never in redacted_excerpt.
    """

    _PATTERNS = [
        ("email",   re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'), "[EMAIL_REDACTED]"),
        ("phone",   re.compile(r'\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b'), "[PHONE_REDACTED]"),
        ("ssn",     re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), "[SSN_REDACTED]"),
        ("cc",      re.compile(r'\b(?:\d[ -]?){13,16}\b'), None),  # Luhn-validated below
    ]

    def scan(self, text: str) -> Tuple[str, List[dict], List[ScanHit]]:
        """
        Returns (redacted_text, redaction_records, hits).
        redaction_records include raw match for audit; hits have only safe excerpts.
        """
        redacted = text
        records: List[dict] = []
        hits: List[ScanHit] = []

        for pii_type, pattern, placeholder in self._PATTERNS:
            for match in pattern.finditer(redacted):
                raw = match.group()

                if pii_type == "cc":
                    digits = re.sub(r'\D', '', raw)
                    if len(digits) < 13 or not _luhn_valid(digits):
                        continue
                    placeholder = "[CC_REDACTED]"

                records.append({
                    "type": pii_type,
                    "raw": raw,          # audit only — never logged to file
                    "placeholder": placeholder,
                })
                hits.append(ScanHit(
                    scanner="PIIRedaction",
                    rule_id=f"pii_{pii_type}",
                    severity="WARN",
                    redacted_excerpt=f"[{pii_type.upper()} detected and redacted]",
                ))
                redacted = redacted.replace(raw, placeholder, 1)

        return redacted, records, hits


# ---------------------------------------------------------------------------
# SecretsScanner
# ---------------------------------------------------------------------------

class SecretsScanner:
    """
    Detects API keys, tokens, and high-entropy strings before they can be
    echoed back or reach gaia-study's training data.
    All findings produce BLOCK severity.
    """

    _PATTERNS = [
        ("openai_key",   re.compile(r'\bsk-[A-Za-z0-9]{20,}\b')),
        ("github_pat",   re.compile(r'\bghp_[A-Za-z0-9]{36}\b')),
        ("aws_key",      re.compile(r'\bAKIA[0-9A-Z]{16}\b')),
        ("private_key",  re.compile(r'-----BEGIN\s+(?:RSA |EC |)PRIVATE KEY-----')),
    ]

    _HIGH_ENTROPY_RE = re.compile(r'[A-Za-z0-9+/=_\-]{20,}')
    _ENTROPY_THRESHOLD = 4.5

    def scan(self, text: str) -> List[ScanHit]:
        hits: List[ScanHit] = []

        for rule_id, pattern in self._PATTERNS:
            for match in pattern.finditer(text):
                raw = match.group()
                # Show only pattern name + first 3 chars to avoid partial key leakage
                excerpt = f"{rule_id}:{raw[:3]}[redacted]"
                hits.append(ScanHit(
                    scanner="Secrets",
                    rule_id=rule_id,
                    severity="BLOCK",
                    redacted_excerpt=excerpt,
                ))

        # High-entropy string detection
        for match in self._HIGH_ENTROPY_RE.finditer(text):
            token = match.group()
            if _shannon_entropy(token) > self._ENTROPY_THRESHOLD:
                hits.append(ScanHit(
                    scanner="Secrets",
                    rule_id="high_entropy_token",
                    severity="BLOCK",
                    redacted_excerpt=f"high_entropy:{token[:3]}[redacted]",
                ))

        return hits


# ---------------------------------------------------------------------------
# VulnerabilityScanner
# ---------------------------------------------------------------------------

class VulnerabilityScanner:
    """
    Flags common injection/traversal/SSRF patterns with WARN severity.
    These are heuristic signals — high false-positive potential in conversational
    context — so they suppress tool execution (dry_run) but do not BLOCK.
    """

    _PATTERNS = [
        ("sqli",          re.compile(r"(UNION\s+SELECT|DROP\s+TABLE|';\s*--|\bOR\s+1=1\b)", re.IGNORECASE)),
        ("ssrf_loopback", re.compile(r"https?://(?:localhost|127\.0\.0\.1|0\.0\.0\.0)", re.IGNORECASE)),
        ("ssrf_169",      re.compile(r"169\.254\.\d{1,3}\.\d{1,3}")),
        ("ssrf_10",       re.compile(r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
        ("ssrf_192168",   re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b")),
        ("path_traversal",re.compile(r"(?:\.\./){2,}")),
        ("cmd_injection", re.compile(r"(?:;\s*\w|&&|\|\||`|\$\()")),
        ("xss",           re.compile(r"(?:<script|javascript:|onerror\s*=)", re.IGNORECASE)),
    ]

    def scan(self, text: str) -> List[ScanHit]:
        hits: List[ScanHit] = []
        for rule_id, pattern in self._PATTERNS:
            if pattern.search(text):
                hits.append(ScanHit(
                    scanner="Vulnerability",
                    rule_id=rule_id,
                    severity="WARN",
                    redacted_excerpt=f"[{rule_id} pattern detected]",
                ))
        return hits

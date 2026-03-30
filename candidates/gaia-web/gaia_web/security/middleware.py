"""
SecurityScanMiddleware — inbound content protection for gaia-web.

Runs after a CognitionPacket is constructed but before dispatch to gaia-core.
Hardened against prompt injection, PII leakage, secret exposure, and common
web vulnerability patterns. All scanners use stdlib only.

Integration pattern (all callers):
    packet, should_block = _security_middleware.scan_packet(packet)
    if should_block:
        # abort — do NOT dispatch to gaia-core
        ...

Audit log: /logs/security_audit.jsonl (JSONL, append mode)
  - Raw matches are NEVER written to this file.
  - Each line contains safe redacted_excerpt values only.
"""

import logging
import json
from datetime import datetime, timezone
from typing import Tuple

from gaia_common.protocols.cognition_packet import (
    CognitionPacket,
    SecurityScan,
    SecurityScanResult,
)
from gaia_web.security.scanner import (
    PromptInjectionScanner,
    PIIRedactionScanner,
    SecretsScanner,
    VulnerabilityScanner,
)

logger = logging.getLogger("GAIA.SecurityScanner")

# Dedicated audit file logger — raw matches NEVER written here
_audit_logger = logging.getLogger("GAIA.SecurityAudit")
_audit_handler = logging.FileHandler("/logs/security_audit.jsonl", mode="a", encoding="utf-8")
_audit_handler.setFormatter(logging.Formatter("%(message)s"))
_audit_logger.addHandler(_audit_handler)
_audit_logger.setLevel(logging.INFO)
_audit_logger.propagate = False  # Don't bleed into main log


def _load_config() -> dict:
    """Load SECURITY_SCAN block from gaia_constants.json via Config singleton if available."""
    try:
        from gaia_common.config import Config
        cfg = Config()
        return cfg.get("SECURITY_SCAN", {})
    except Exception:
        pass
    # Fallback defaults
    return {
        "enabled": True,
        "injection_block_threshold": 0.85,
        "injection_warn_threshold": 0.50,
        "redact_pii": True,
        "scan_secrets": True,
    }


class SecurityScanMiddleware:
    """
    Inbound security scanner singleton.

    Instantiate once at module level; call scan_packet() or scan_text() per request.
    If config.enabled is False, both methods are no-ops.
    """

    def __init__(self) -> None:
        self._cfg = _load_config()
        self._enabled = self._cfg.get("enabled", True)
        self._block_threshold = float(self._cfg.get("injection_block_threshold", 0.85))
        self._warn_threshold = float(self._cfg.get("injection_warn_threshold", 0.50))
        self._redact_pii = self._cfg.get("redact_pii", True)
        self._scan_secrets = self._cfg.get("scan_secrets", True)

        self._injection_scanner = PromptInjectionScanner()
        self._pii_scanner = PIIRedactionScanner()
        self._secrets_scanner = SecretsScanner()
        self._vuln_scanner = VulnerabilityScanner()

        logger.info(
            "SecurityScanMiddleware initialised (enabled=%s, block=%.2f, warn=%.2f)",
            self._enabled, self._block_threshold, self._warn_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_packet(self, packet: CognitionPacket) -> Tuple[CognitionPacket, bool]:
        """
        Scan a CognitionPacket's original_prompt.

        Returns (mutated_packet, should_block).
        Mutates packet.content.original_prompt if PII was redacted.
        Writes scan results into packet.governance.security_scan.
        """
        if not self._enabled:
            return packet, False

        prompt = getattr(packet.content, "original_prompt", "") or ""
        packet_id = getattr(packet.header, "packet_id", "unknown")
        session_id = getattr(packet.header, "session_id", "unknown")

        redacted_prompt, scan, should_block = self.scan_text(prompt, packet_id, session_id)

        # Mutate prompt if redacted
        if redacted_prompt != prompt:
            packet.content.original_prompt = redacted_prompt

        # Write scan results into governance
        packet.governance.security_scan = scan

        # WARN behaviour: enable dry_run to suppress tool execution
        if scan.scan_results and not should_block:
            if not packet.governance.safety.dry_run:
                packet.governance.safety.dry_run = True
                logger.info("SecurityScan WARN: dry_run enabled for packet %s", packet_id)

        return packet, should_block

    def scan_text(self, text: str, packet_id: str, session_id: str) -> Tuple[str, SecurityScan, bool]:
        """
        Scan raw text (for dict-based paths such as main.py).

        Returns (redacted_text, SecurityScan, should_block).
        """
        if not self._enabled:
            return text, SecurityScan(ran=False, passed=True), False

        all_hits = []
        should_block = False
        redacted = text

        # --- Injection scan ---
        injection_score, inj_hits = self._injection_scanner.scan(text)
        all_hits.extend(inj_hits)

        # --- PII redaction ---
        if self._redact_pii:
            redacted, redaction_records, pii_hits = self._pii_scanner.scan(redacted)
            all_hits.extend(pii_hits)

        # --- Secrets scan ---
        if self._scan_secrets:
            secret_hits = self._secrets_scanner.scan(redacted)
            all_hits.extend(secret_hits)

        # --- Vulnerability scan ---
        vuln_hits = self._vuln_scanner.scan(text)
        all_hits.extend(vuln_hits)

        # --- Determine block ---
        has_block_hit = any(h.severity == "BLOCK" for h in all_hits)
        if injection_score >= self._block_threshold or has_block_hit:
            should_block = True

        # --- Build SecurityScan ---
        scan_results = [
            SecurityScanResult(
                scanner=h.scanner,
                rule_id=h.rule_id,
                severity=h.severity,
                redacted_excerpt=h.redacted_excerpt,
                timestamp=h.timestamp,
            )
            for h in all_hits
        ]

        passed = not should_block
        scan = SecurityScan(
            ran=True,
            passed=passed,
            scan_results=scan_results,
            injection_score=injection_score,
        )

        # --- Audit log (safe values only — raw matches never logged) ---
        now = datetime.now(timezone.utc).isoformat()
        for hit in all_hits:
            entry = {
                "packet_id": packet_id,
                "session_id": session_id,
                "source": "SecurityScanMiddleware",
                "timestamp": now,
                "scanner": hit.scanner,
                "rule_id": hit.rule_id,
                "severity": hit.severity,
                "redacted_excerpt": hit.redacted_excerpt,
            }
            _audit_logger.info(json.dumps(entry))

        if all_hits:
            logger.info(
                "SecurityScan packet=%s session=%s injection_score=%.2f hits=%d block=%s",
                packet_id, session_id, injection_score, len(all_hits), should_block,
            )
        elif logger.isEnabledFor(logging.DEBUG):
            logger.debug("SecurityScan packet=%s clean", packet_id)

        return redacted, scan, should_block

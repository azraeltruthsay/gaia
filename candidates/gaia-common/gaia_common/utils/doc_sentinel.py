"""
DocSentinel — Automated Living Documentation (Phase 5-C, Proposal 05)

Three-stage pipeline that keeps GAIA's documentation alive:

  1. Incident Reports  — real-time, triggered by GaiaVitals state changes
  2. Glossary Miner    — background, extracts terms from proposals/chamber
  3. Capability Catalog — automatic, scans CapabilityEngine for active limbs

All output lands in the wiki's auto/ directory (mounted from gaia-instance).

Usage:
    from gaia_common.utils.doc_sentinel import DocSentinel

    sentinel = DocSentinel()
    sentinel.record_event("IRRITATED", "Loop counter exceeded tier 1", trace="...")
    sentinel.mine_glossary()
    sentinel.generate_capability_catalog(engine)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("GAIA.DocSentinel")

# ── Configuration ──────────────────────────────────────────────────────

# Wiki auto-generated docs directory (mounted volume in gaia-wiki container)
WIKI_AUTO_DIR = Path(os.environ.get(
    "WIKI_AUTO_DIR",
    os.environ.get("KNOWLEDGE_DIR", "/knowledge") + "/wiki_auto"
))

# Incident reports subdirectory
INCIDENTS_DIR = WIKI_AUTO_DIR / "incidents"

# Source directories for glossary mining
PROJECT_ROOT = Path(os.environ.get("GAIA_PROJECT_ROOT", "/gaia/GAIA_Project"))
PROPOSALS_DIR = PROJECT_ROOT / "proposals"
COUNCIL_CHAMBER = PROJECT_ROOT / "COUNCIL_CHAMBER.md"

# Known GAIA terms that should always appear in the glossary
_SEED_TERMS = {
    "AAAK": "Azrael's Autonomous Architecture of Knowledge — GAIA's compression format for context transfer.",
    "CPR Loop": "Cognitive Pulse Resuscitation — tiered self-recovery from reasoning loops (Breath/Diagnosis/Intubation).",
    "Sovereign Shield": "GAIA's unified security system (Blast Shield + Approval Gate + Force Field).",
    "Blast Shield": "Deterministic pre-flight safety check that blocks dangerous commands before execution.",
    "Force Field": "Adversarial translation system — sanitizes injection payloads while allowing GAIA to respond.",
    "NeuralRouter": "Unified 6-stage intent + engine routing pipeline (reflex/embed/weighted/heuristic/matrix/nano).",
    "GaiaVitals": "Unified sovereign health monitor — 4 pulses (biological/structural/cognitive/security).",
    "CapabilityEngine": "Unified tools + skills registry — every GAIA action is a 'Limb'.",
    "Limb": "A capability (tool or skill) registered in the CapabilityEngine.",
    "Memento Skill": "Hot-reloadable Python module in gaia-mcp/skills/ that extends GAIA's capabilities.",
    "MemPalace": "GAIA's persistent memory system — palace/wing/hall/drawer/artifact hierarchy.",
    "Cognition Packet": "Structured data envelope (v0.3) that flows through the cognitive pipeline.",
    "Heartbeat": "Background canary that validates the inference chain by asking Nano the time.",
    "Irritation Score": "Weighted aggregate (0-100) of all GaiaVitals pulse domains.",
    "Sovereign Status": "Overall system health: STABLE/IRRITATED/RECOVERING/CRITICAL/LOCKED.",
}


# ── DocSentinel ────────────────────────────────────────────────────────

class DocSentinel:
    """Automated documentation maintainer for GAIA.

    Stateless — can be instantiated fresh per call or cached as singleton.
    All output is file-based (wiki auto/ directory on shared volume).
    """

    def __init__(self, auto_dir: Optional[Path] = None):
        self.auto_dir = auto_dir or WIKI_AUTO_DIR
        self._last_status: Optional[str] = None

    # ── Stage 1: Incident Reports ──────────────────────────────────────

    def record_event(
        self,
        status: str,
        reason: str,
        trace: str = "",
        previous_status: str = "",
        vitals_snapshot: Optional[Dict] = None,
    ) -> Optional[Path]:
        """Record a sovereign health incident as a Markdown report.

        Called by GaiaVitals or any monitoring system when a state
        change is detected (e.g., STABLE -> IRRITATED).

        Args:
            status: Current sovereign status (e.g., "IRRITATED")
            reason: Human-readable reason for the state change
            trace: Optional loop trace or error fragment
            previous_status: What the status was before
            vitals_snapshot: Optional full vitals report dict

        Returns:
            Path to the generated incident file, or None on failure.
        """
        try:
            incidents_dir = self.auto_dir / "incidents"
            incidents_dir.mkdir(parents=True, exist_ok=True)

            now = datetime.now(timezone.utc)
            slug = re.sub(r'[^a-z0-9]+', '-', reason.lower()[:40]).strip('-')
            filename = f"{now.strftime('%Y-%m-%d_%H%M')}_{slug}.md"
            filepath = incidents_dir / filename

            lines = [
                f"# Incident: {status}",
                "",
                f"**Time**: {now.isoformat()}",
                f"**Status**: {previous_status} -> {status}" if previous_status else f"**Status**: {status}",
                f"**Reason**: {reason}",
                "",
            ]

            if trace:
                lines.extend([
                    "## Trace",
                    "```",
                    trace[:2000],
                    "```",
                    "",
                ])

            if vitals_snapshot:
                lines.extend([
                    "## Vitals Snapshot",
                    "```json",
                    json.dumps(vitals_snapshot, indent=2, default=str)[:3000],
                    "```",
                    "",
                ])

            lines.extend([
                "## Resolution",
                "",
                "_Pending — document resolution steps here._",
                "",
                f"---",
                f"*Auto-generated by DocSentinel at {now.strftime('%H:%M UTC')}*",
            ])

            filepath.write_text("\n".join(lines))
            logger.info("DocSentinel: incident recorded -> %s", filepath.name)
            return filepath

        except Exception:
            logger.debug("DocSentinel: failed to record incident", exc_info=True)
            return None

    def check_and_record(self, vitals_report: Dict[str, Any]) -> Optional[Path]:
        """Check for status change and auto-record if needed.

        Tracks the last known status and fires record_event only on
        transitions (not every poll cycle).
        """
        current = vitals_report.get("sovereign_status", "STABLE")

        if self._last_status is not None and current != self._last_status:
            # State change detected
            if current not in ("STABLE",):  # don't record recovery-to-stable
                path = self.record_event(
                    status=current,
                    reason=f"State transition: {self._last_status} -> {current}",
                    previous_status=self._last_status,
                    vitals_snapshot=vitals_report,
                )
                self._last_status = current
                return path

        self._last_status = current
        return None

    # ── Stage 2: Glossary Miner ────────────────────────────────────────

    def mine_glossary(self, extra_sources: Optional[List[Path]] = None) -> Path:
        """Extract technical terms and generate auto/glossary.md.

        Scans:
          - proposals/*.md for defined terms
          - COUNCIL_CHAMBER.md for strategic vocabulary
          - Seed terms from _SEED_TERMS

        Returns path to generated glossary file.
        """
        terms: Dict[str, str] = dict(_SEED_TERMS)

        # Mine proposals
        sources = list((extra_sources or []))
        if PROPOSALS_DIR.exists():
            sources.extend(sorted(PROPOSALS_DIR.glob("*.md")))
        if COUNCIL_CHAMBER.exists():
            sources.append(COUNCIL_CHAMBER)

        for source_path in sources:
            try:
                text = source_path.read_text(errors="replace")
                mined = self._extract_terms_from_markdown(text, source_path.name)
                terms.update(mined)
            except Exception:
                logger.debug("DocSentinel: failed to mine %s", source_path, exc_info=True)

        # Generate glossary markdown
        glossary_path = self.auto_dir / "glossary.md"
        self.auto_dir.mkdir(parents=True, exist_ok=True)

        lines = [
            "# GAIA Glossary",
            "",
            f"> Auto-generated by DocSentinel | Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            f"**{len(terms)} terms** across the GAIA architecture.",
            "",
        ]

        for term in sorted(terms.keys(), key=str.lower):
            definition = terms[term]
            lines.append(f"**{term}**")
            lines.append(f": {definition}")
            lines.append("")

        glossary_path.write_text("\n".join(lines))
        logger.info("DocSentinel: glossary generated with %d terms -> %s", len(terms), glossary_path)
        return glossary_path

    @staticmethod
    def _extract_terms_from_markdown(text: str, source_name: str = "") -> Dict[str, str]:
        """Extract defined terms from Markdown text.

        Looks for patterns like:
          - **Term**: Definition
          - `Term` — Description
          - | Term | Description |  (table rows)
          - # Proposal NN: Title (extract title as term)
        """
        terms: Dict[str, str] = {}

        # Pattern: **Term**: Definition or **Term** — Definition
        for match in re.finditer(
            r'\*\*([A-Z][A-Za-z0-9_ -]+)\*\*\s*[:—–-]\s*(.+?)(?:\n|$)',
            text
        ):
            term = match.group(1).strip()
            definition = match.group(2).strip()
            if len(term) >= 3 and len(definition) >= 10:
                terms[term] = definition[:200]

        # Pattern: `Term` — Description
        for match in re.finditer(
            r'`([A-Za-z][A-Za-z0-9_.-]+)`\s*[—–-]\s*(.+?)(?:\n|$)',
            text
        ):
            term = match.group(1).strip()
            definition = match.group(2).strip()
            if len(term) >= 3 and len(definition) >= 10:
                terms[term] = definition[:200]

        # Pattern: Proposal title
        for match in re.finditer(
            r'^#\s+Proposal\s+\d+:\s+(.+?)(?:\s*\(|$)',
            text, re.MULTILINE
        ):
            title = match.group(1).strip()
            terms[title] = f"GAIA consolidation proposal (source: {source_name})"

        return terms

    # ── Stage 3: Capability Catalog ────────────────────────────────────

    def generate_capability_catalog(
        self,
        limbs: Optional[List[Dict[str, Any]]] = None,
    ) -> Path:
        """Generate auto/capabilities.md from the CapabilityEngine.

        Args:
            limbs: Output from CapabilityEngine.list_limbs().
                   If None, generates a stub.

        Returns path to generated capabilities file.
        """
        catalog_path = self.auto_dir / "capabilities.md"
        self.auto_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

        lines = [
            "# GAIA Capability Catalog",
            "",
            f"> Auto-generated by DocSentinel | Last updated: {now}",
            "",
        ]

        if not limbs:
            lines.extend([
                "No capability data available. Run with CapabilityEngine.list_limbs().",
                "",
            ])
            catalog_path.write_text("\n".join(lines))
            return catalog_path

        # Group by domain
        domains: Dict[str, List[Dict]] = {}
        for limb in limbs:
            domain = limb.get("domain", "unknown")
            domains.setdefault(domain, []).append(limb)

        static_count = sum(1 for l in limbs if l.get("source") == "static")
        dynamic_count = sum(1 for l in limbs if l.get("source") == "dynamic")
        lines.extend([
            f"**{len(limbs)} capabilities** across **{len(domains)} domains** "
            f"({static_count} static, {dynamic_count} dynamic)",
            "",
        ])

        for domain in sorted(domains.keys()):
            actions = domains[domain]
            lines.append(f"## {domain}")
            lines.append("")
            lines.append("| Action | Source | Sensitive | Description |")
            lines.append("|--------|--------|-----------|-------------|")
            for a in sorted(actions, key=lambda x: x.get("action", "")):
                sensitive = "Yes" if a.get("sensitive") else ""
                desc = a.get("description", "")[:60]
                source = a.get("source", "")
                if source == "dynamic":
                    source = "dynamic (override)" if a.get("legacy_name") else "dynamic"
                lines.append(f"| {a.get('action', '?')} | {source} | {sensitive} | {desc} |")
            lines.append("")

        lines.extend([
            "---",
            f"*Auto-generated by DocSentinel at {now}*",
        ])

        catalog_path.write_text("\n".join(lines))
        logger.info(
            "DocSentinel: capability catalog generated (%d limbs, %d domains) -> %s",
            len(limbs), len(domains), catalog_path,
        )
        return catalog_path


# ── Module-level helpers ───────────────────────────────────────────────

_singleton: Optional[DocSentinel] = None


def get_doc_sentinel() -> DocSentinel:
    """Get or create the DocSentinel singleton."""
    global _singleton
    if _singleton is None:
        _singleton = DocSentinel()
    return _singleton

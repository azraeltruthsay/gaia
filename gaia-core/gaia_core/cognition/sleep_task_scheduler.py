"""
Sleep Task Scheduler — orchestrates autonomous maintenance during SLEEPING state.

Registered tasks are executed one-at-a-time in priority order (lowest number = highest
priority), with least-recently-run selection among tasks of equal priority.

All task handlers are plain synchronous functions — the sleep cycle loop runs in a
daemon thread, not an asyncio event loop.
"""

from __future__ import annotations

import json as _json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("GAIA.SleepTaskScheduler")


@dataclass
class SleepTask:
    """A single registerable sleep-time task."""

    task_id: str
    task_type: str
    priority: int  # 1 = highest
    interruptible: bool
    estimated_duration_seconds: int
    handler: Callable[[], Any]
    last_run: Optional[datetime] = None
    run_count: int = 0
    last_error: Optional[str] = None


class SleepTaskScheduler:
    """Priority-based scheduler for sleep-time maintenance tasks."""

    def __init__(
        self,
        config,
        model_pool=None,
        agent_core=None,
        timeline_store=None,
    ) -> None:
        self.config = config
        self.model_pool = model_pool
        self.agent_core = agent_core
        self._timeline = timeline_store
        self._tasks: List[SleepTask] = []

        self._register_default_tasks()

    # ------------------------------------------------------------------
    # Task registration
    # ------------------------------------------------------------------

    def register_task(self, task: SleepTask) -> None:
        self._tasks.append(task)
        logger.info("Registered sleep task: %s (P%d)", task.task_id, task.priority)

    def _register_default_tasks(self) -> None:
        """Register built-in maintenance tasks."""

        self.register_task(SleepTask(
            task_id="conversation_curation",
            task_type="conversation_curation",
            priority=1,
            interruptible=True,
            estimated_duration_seconds=60,
            handler=self._run_conversation_curation,
        ))

        self.register_task(SleepTask(
            task_id="samvega_introspection",
            task_type="REFLECTIVE_MEMORY",
            priority=2,
            interruptible=True,
            estimated_duration_seconds=120,
            handler=self._run_samvega_introspection,
        ))

        self.register_task(SleepTask(
            task_id="blueprint_validation",
            task_type="blueprint_validation",
            priority=3,
            interruptible=True,
            estimated_duration_seconds=420,
            handler=self._run_blueprint_validation,
        ))

        self.register_task(SleepTask(
            task_id="code_evolution",
            task_type="code_evolution",
            priority=3,
            interruptible=True,
            estimated_duration_seconds=30,
            handler=self._run_code_evolution,
        ))

        self.register_task(SleepTask(
            task_id="promotion_readiness",
            task_type="PROMOTION_READINESS",
            priority=3,
            interruptible=True,
            estimated_duration_seconds=90,
            handler=self._run_promotion_readiness,
        ))

        self.register_task(SleepTask(
            task_id="code_review",
            task_type="SELF_MODEL_UPDATE",
            priority=4,
            interruptible=True,
            estimated_duration_seconds=120,
            handler=self._run_code_review,
        ))

        self.register_task(SleepTask(
            task_id="knowledge_research",
            task_type="KNOWLEDGE_ACQUISITION",
            priority=4,
            interruptible=True,
            estimated_duration_seconds=180,
            handler=self._run_knowledge_research,
        ))

        self.register_task(SleepTask(
            task_id="wiki_doc_regen",
            task_type="DOC_GENERATION",
            priority=5,
            interruptible=True,
            estimated_duration_seconds=30,
            handler=self._run_wiki_doc_regen,
        ))

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def get_next_task(self) -> Optional[SleepTask]:
        """Return the highest-priority, least-recently-run task."""
        if not self._tasks:
            return None

        # Sort by (priority ASC, last_run ASC nulls-first)
        epoch = datetime.min.replace(tzinfo=timezone.utc)
        candidates = sorted(
            self._tasks,
            key=lambda t: (t.priority, t.last_run or epoch),
        )
        return candidates[0] if candidates else None

    def execute_task(self, task: SleepTask) -> bool:
        """Execute a task handler. Returns True on success."""
        logger.info("Starting sleep task: %s", task.task_id)
        start = time.monotonic()
        try:
            task.handler()
            elapsed = time.monotonic() - start
            task.last_run = datetime.now(timezone.utc)
            task.run_count += 1
            task.last_error = None
            logger.info("Completed %s in %.1fs (run #%d)", task.task_id, elapsed, task.run_count)
            self._emit_task_exec(task.task_id, task.task_type, elapsed, True)
            return True
        except Exception as exc:
            elapsed = time.monotonic() - start
            task.last_run = datetime.now(timezone.utc)
            task.last_error = str(exc)
            logger.error("Task %s failed after %.1fs: %s", task.task_id, elapsed, exc, exc_info=True)
            self._emit_task_exec(task.task_id, task.task_type, elapsed, False, str(exc))
            return False

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> List[Dict[str, Any]]:
        return [
            {
                "task_id": t.task_id,
                "task_type": t.task_type,
                "priority": t.priority,
                "interruptible": t.interruptible,
                "run_count": t.run_count,
                "last_run": t.last_run.isoformat() if t.last_run else None,
                "last_error": t.last_error,
            }
            for t in self._tasks
        ]

    # ------------------------------------------------------------------
    # Built-in task handlers
    # ------------------------------------------------------------------

    def _run_conversation_curation(self) -> None:
        """Curate recent session conversations for the knowledge base."""
        from gaia_core.cognition.conversation_curator import ConversationCurator
        from gaia_core.memory.session_manager import SessionManager

        curator = ConversationCurator()
        session_manager = SessionManager(self.config)

        # Curate all active sessions that have enough messages
        curated = 0
        for sid, session in session_manager.sessions.items():
            if session.history:
                if curator.curate(sid, session.history):
                    curated += 1

        logger.info("Conversation curation: %d sessions curated", curated)

    # Blueprint-to-source mapping for validation
    _BLUEPRINT_SOURCES: Dict[str, List[str]] = {
        "GAIA_SLEEP_CYCLE.md": [
            "gaia_core/cognition/sleep_wake_manager.py",
            "gaia_core/cognition/sleep_cycle_loop.py",
            "gaia_core/api/sleep_endpoints.py",
            "gaia_core/utils/resource_monitor.py",
        ],
        "GAIA_CORE.md": [
            "gaia_core/cognition/agent_core.py",
            "gaia_core/main.py",
        ],
        "GAIA_ORCHESTRATOR.md": [
            "gaia_orchestrator/handoff_manager.py",
            "gaia_orchestrator/gpu_manager.py",
        ],
    }

    # Service IDs with known YAML blueprints that should use the structured
    # pre-check path. The .md legacy path remains for blueprints without YAML.
    _YAML_BLUEPRINT_SERVICES: List[str] = [
        "gaia-core", "gaia-web", "gaia-mcp", "gaia-orchestrator",
        "gaia-prime", "gaia-study", "gaia-audio",
    ]

    # Map YAML service IDs to their source directories (candidate preferred)
    _SERVICE_SOURCE_DIRS: Dict[str, str] = {
        "gaia-core": "/gaia/GAIA_Project/candidates/gaia-core/gaia_core",
        "gaia-web": "/gaia/GAIA_Project/candidates/gaia-web/gaia_web",
        "gaia-mcp": "/gaia/GAIA_Project/candidates/gaia-mcp/gaia_mcp",
        "gaia-orchestrator": "/gaia/GAIA_Project/candidates/gaia-orchestrator/gaia_orchestrator",
        "gaia-study": "/gaia/GAIA_Project/candidates/gaia-study/gaia_study",
        "gaia-audio": "/gaia/GAIA_Project/candidates/gaia-audio/gaia_audio",
    }

    def _run_promotion_readiness(self) -> None:
        """Assess promotion readiness for candidate services.

        For each service in _YAML_BLUEPRINT_SERVICES:
        - If candidate source exists but no live directory → assess readiness
        - If candidate source is newer than live → assess readiness
        - Auto-generate blueprints for services missing them
        - Write reports and council notes for promotable services
        """
        try:
            from gaia_common.utils.promotion_readiness import assess_promotion_readiness
            from gaia_common.utils.blueprint_generator import generate_candidate_blueprint
            from gaia_common.utils.blueprint_io import load_blueprint, save_blueprint
            from gaia_common.utils.promotion_request import (
                create_promotion_request,
                load_pending_request,
            )
        except ImportError:
            logger.debug("Promotion readiness modules not available, skipping")
            return

        project_root = "/gaia/GAIA_Project"
        reports_dir = Path(project_root) / "knowledge" / "promotion_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        for service_id in self._YAML_BLUEPRINT_SERVICES:
            candidate_dir = Path(project_root) / "candidates" / service_id
            live_dir = Path(project_root) / service_id

            # Only assess services with candidates that differ from live
            if not candidate_dir.exists():
                continue
            if live_dir.exists():
                # Skip if live directory exists (already promoted)
                # Future: compare mtimes for re-promotion
                continue

            logger.info("Assessing promotion readiness for %s", service_id)

            # Auto-generate blueprint if missing
            bp = load_blueprint(service_id, candidate=True)
            if bp is None:
                bp = load_blueprint(service_id, candidate=False)
            if bp is None:
                source_dir = self._SERVICE_SOURCE_DIRS.get(service_id)
                if source_dir and Path(source_dir).exists():
                    try:
                        bp = generate_candidate_blueprint(service_id, source_dir)
                        save_blueprint(bp, candidate=True)
                        logger.info("Auto-generated blueprint for %s", service_id)
                    except Exception:
                        logger.warning("Blueprint generation failed for %s", service_id, exc_info=True)

            # Run readiness assessment
            try:
                report = assess_promotion_readiness(service_id, project_root)
            except Exception:
                logger.warning("Readiness assessment failed for %s", service_id, exc_info=True)
                continue

            # Save report
            import json as _json_mod
            report_path = reports_dir / f"{service_id}.json"
            report_path.write_text(
                _json_mod.dumps(report.to_dict(), indent=2),
                encoding="utf-8",
            )
            logger.info(
                "Promotion readiness for %s: %s (%d/%d checks pass)",
                service_id, report.verdict,
                report.pass_count, len(report.checks),
            )

            # Create promotion request if service is ready and no pending request exists
            if report.verdict in ("ready", "ready_with_warnings"):
                existing = load_pending_request(service_id)
                if existing is None:
                    try:
                        req = create_promotion_request(
                            service_id=service_id,
                            verdict=report.verdict,
                            recommendation=report.recommendation,
                            pipeline_cmd=report.pipeline_cmd,
                            check_summary=report.to_markdown(),
                        )
                        logger.info("Created promotion request %s for %s", req.request_id, service_id)

                        # Write council note
                        try:
                            from gaia_core.cognition.council_notes import CouncilNoteManager
                            cn = CouncilNoteManager(self.config)
                            cn.write_note(
                                user_prompt=f"[System] Promotion readiness: {service_id}",
                                lite_response=report.to_markdown(),
                                escalation_reason=f"Service {service_id} is {report.verdict} for promotion",
                                session_id="sleep-promotion-readiness",
                            )
                        except Exception:
                            logger.debug("Could not write council note for promotion readiness", exc_info=True)
                    except Exception:
                        logger.warning("Could not create promotion request for %s", service_id, exc_info=True)

    def _run_blueprint_validation(self) -> None:
        """Scan blueprints against source files and flag stale content.

        Uses two paths:
        1. YAML blueprints → structured pre-check via blueprint_precheck module
        2. Legacy .md blueprints → inline regex extraction (fallback)
        3. Code-architect corpus readiness check
        """
        total_mismatches = 0

        # ── Path 1: YAML blueprint pre-check (structured) ────────────────
        total_mismatches += self._validate_yaml_blueprints()

        # ── Path 2: Legacy .md blueprint validation (fallback) ────────────
        total_mismatches += self._validate_legacy_blueprints()

        # ── Path 3: Code-architect corpus readiness ──────────────────────
        self._check_code_architect_corpus()

        logger.info(
            "Blueprint validation complete: %d total mismatches",
            total_mismatches,
        )
        self._rebuild_blueprint_embeddings()

    def _validate_yaml_blueprints(self) -> int:
        """Run structured pre-check on YAML blueprints. Returns mismatch count."""
        try:
            from gaia_common.utils.blueprint_io import load_blueprint
            from gaia_common.utils.blueprint_precheck import run_blueprint_precheck
        except ImportError:
            logger.debug("blueprint_precheck not available, skipping YAML validation")
            return 0

        mismatches = 0
        for service_id in self._YAML_BLUEPRINT_SERVICES:
            bp = load_blueprint(service_id)
            if bp is None:
                logger.debug("No YAML blueprint for %s, skipping", service_id)
                continue

            source_dir = self._SERVICE_SOURCE_DIRS.get(service_id)
            if source_dir is None or not Path(source_dir).exists():
                # Try production path
                source_dir = f"/gaia/GAIA_Project/{service_id.replace('-', '_')}"
                if not Path(source_dir).exists():
                    logger.debug("No source dir for %s, skipping", service_id)
                    continue

            result = run_blueprint_precheck(bp, source_dir)
            missing_items = [i for i in result.items if i.status == "missing"]

            if missing_items:
                mismatches += len(missing_items)
                missing_strs = [
                    f"{i.category}:{i.blueprint_claim}" for i in missing_items
                ]
                logger.warning(
                    "Blueprint %s.yaml has %d missing items: %s",
                    service_id, len(missing_items), missing_strs,
                )
            else:
                logger.info(
                    "Blueprint %s.yaml pre-check passed (%d/%d found)",
                    service_id, result.summary.found, result.summary.total,
                )

        return mismatches

    def _validate_legacy_blueprints(self) -> int:
        """Run legacy .md blueprint validation. Returns mismatch count."""
        blueprints_dir = Path("/gaia/GAIA_Project/knowledge/blueprints")
        source_roots = [
            Path("/gaia/GAIA_Project/candidates/gaia-core"),
            Path("/gaia/GAIA_Project/candidates/gaia-orchestrator"),
            Path("/gaia/GAIA_Project/gaia-core"),
            Path("/gaia/GAIA_Project/gaia-orchestrator"),
        ]

        mismatches = 0
        for bp_name, source_files in self._BLUEPRINT_SOURCES.items():
            bp_path = blueprints_dir / bp_name
            if not bp_path.exists():
                logger.debug("Blueprint %s not found, skipping", bp_name)
                continue

            bp_text = bp_path.read_text(encoding="utf-8")
            facts = self._extract_facts(source_files, source_roots)
            missing = self._check_facts(facts, bp_text)

            if missing:
                mismatches += len(missing)
                logger.warning(
                    "Blueprint %s has %d stale references: %s",
                    bp_name, len(missing), missing,
                )
                self._append_update_notes(bp_path, bp_text, missing)
            else:
                logger.info("Blueprint %s is up-to-date", bp_name)

        return mismatches

    @staticmethod
    def _extract_facts(
        source_files: List[str],
        source_roots: List[Path],
    ) -> Dict[str, List[str]]:
        """Extract enum members, endpoints, and key constants from source files.

        Returns dict of {category: [fact_string, ...]}.
        """
        facts: Dict[str, List[str]] = {
            "enums": [],
            "endpoints": [],
            "constants": [],
        }

        re_enum_class = re.compile(r"^class\s+(\w+)\(.*Enum.*\):", re.MULTILINE)
        re_enum_member = re.compile(r"^\s+(\w+)\s*=\s*", re.MULTILINE)
        re_endpoint = re.compile(
            r'@router\.(get|post|put|delete|patch)\(\s*["\']([^"\']+)["\']',
            re.MULTILINE,
        )
        re_constant = re.compile(r"^([A-Z][A-Z_]{2,})\s*=\s*", re.MULTILINE)

        for rel_path in source_files:
            src_path = None
            for root in source_roots:
                candidate = root / rel_path
                if candidate.exists():
                    src_path = candidate
                    break
            if src_path is None:
                continue

            text = src_path.read_text(encoding="utf-8")

            # Extract enum members
            for class_match in re_enum_class.finditer(text):
                class_name = class_match.group(1)
                # Find members in the lines following the class definition
                class_start = class_match.end()
                # Read until next class/def at column 0
                block_end = len(text)
                for m in re.finditer(r"^(?:class |def )", text[class_start:], re.MULTILINE):
                    block_end = class_start + m.start()
                    break
                block = text[class_start:block_end]
                for member_match in re_enum_member.finditer(block):
                    member = member_match.group(1)
                    if not member.startswith("_"):
                        facts["enums"].append(f"{class_name}.{member}")

            # Extract endpoints
            for ep_match in re_endpoint.finditer(text):
                method = ep_match.group(1).upper()
                path = ep_match.group(2)
                facts["endpoints"].append(f"{method} {path}")

            # Extract top-level constants
            for const_match in re_constant.finditer(text):
                facts["constants"].append(const_match.group(1))

        return facts

    @staticmethod
    def _check_facts(
        facts: Dict[str, List[str]],
        bp_text: str,
    ) -> List[str]:
        """Return list of facts not found in blueprint text."""
        missing: List[str] = []
        for category, items in facts.items():
            for item in items:
                # For enum members like GaiaState.ACTIVE, check the member name
                if category == "enums":
                    member = item.split(".")[-1]
                    if member not in bp_text:
                        missing.append(f"enum:{item}")
                elif category == "endpoints":
                    # Check for the path portion (e.g., "/sleep/study-handoff")
                    path = item.split(" ", 1)[1]
                    if path not in bp_text:
                        missing.append(f"endpoint:{item}")
                elif category == "constants":
                    if item not in bp_text:
                        missing.append(f"constant:{item}")
        return missing

    @staticmethod
    def _append_update_notes(
        bp_path: Path,
        bp_text: str,
        missing: List[str],
    ) -> None:
        """Append a timestamped 'Recent Implementation Updates' note to a blueprint."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        header = "\n## Recent Implementation Updates\n"

        # Check if section already exists
        if "## Recent Implementation Updates" in bp_text:
            # Append to existing section
            note = f"\n### {timestamp}\n\nDetected {len(missing)} item(s) not reflected in blueprint:\n"
        else:
            note = f"{header}\n### {timestamp}\n\nDetected {len(missing)} item(s) not reflected in blueprint:\n"

        for item in missing:
            note += f"- `{item}`\n"
        note += "\n*Auto-detected by blueprint_validation sleep task.*\n"

        with open(bp_path, "a", encoding="utf-8") as f:
            f.write(note)

    def _rebuild_blueprint_embeddings(self) -> None:
        """Rebuild the vector index for all blueprint documents."""
        try:
            from gaia_common.utils.vector_indexer import VectorIndexer

            VectorIndexer._instances.pop("blueprints", None)
            indexer = VectorIndexer.instance("blueprints")
            indexer.build_index_from_docs(chunk_size=1024, chunk_overlap=128)
            logger.info(
                "Blueprint embeddings rebuilt: %d chunks indexed",
                len(indexer.index.get("docs", [])),
            )
        except Exception:
            logger.error("Failed to rebuild blueprint embeddings", exc_info=True)

    @property
    def _CORPUS_DIR(self) -> str:
        return str(Path(self.config.KNOWLEDGE_DIR) / "curricula" / "code-architect")

    @property
    def _PRIME_MD(self) -> str:
        return str(Path(self.config.SHARED_DIR) / "self_model" / "prime.md")

    _MIN_CORPUS_SIZE = 50
    _MIN_FORWARD_RATIO = 0.15

    def _check_code_architect_corpus(self) -> None:
        """Check code-architect training corpus readiness and log to prime.md."""
        try:
            pairs_dir = Path(self._CORPUS_DIR) / "pairs"
            if not pairs_dir.exists():
                return

            pair_files = list(pairs_dir.glob("*.json"))
            total = len(pair_files)
            if total == 0:
                return

            forward_count = 0
            for pf in pair_files:
                try:
                    data = _json.loads(pf.read_text(encoding="utf-8"))
                    if data.get("pair_type") == "forward":
                        forward_count += 1
                except Exception:
                    continue

            forward_ratio = forward_count / total if total else 0.0

            # Check if adapter already exists
            adapter_dir = Path("/shared/adapters/code-architect")
            adapter_exists = adapter_dir.exists() and any(adapter_dir.iterdir()) if adapter_dir.exists() else False

            if total >= self._MIN_CORPUS_SIZE and forward_ratio >= self._MIN_FORWARD_RATIO:
                if not adapter_exists:
                    note = (
                        f"code-architect corpus has reached training threshold "
                        f"({total} pairs: {total - forward_count} retroactive, {forward_count} forward). "
                        f"Forward pair ratio: {forward_ratio:.0%}. "
                        f"Recommend triggering training via promote_pipeline.sh --qlora --adapter code-architect"
                    )
                    self._append_prime_note(note, high_priority=True)
                    logger.info("Code-architect corpus ready: %s", note)
            elif total >= self._MIN_CORPUS_SIZE and forward_ratio < self._MIN_FORWARD_RATIO:
                needed = max(1, int(self._MIN_CORPUS_SIZE * self._MIN_FORWARD_RATIO) - forward_count)
                note = (
                    f"code-architect corpus size sufficient ({total} pairs) but forward pair ratio "
                    f"too low ({forward_ratio:.0%} < 15%). Need {needed} more forward pairs before training."
                )
                self._append_prime_note(note, high_priority=False)
                logger.info("Code-architect corpus: %s", note)
            else:
                logger.debug(
                    "Code-architect corpus: %d/%d pairs (%d forward)",
                    total, self._MIN_CORPUS_SIZE, forward_count,
                )

        except Exception:
            logger.debug("Code-architect corpus check skipped", exc_info=True)

    def _append_prime_note(self, note: str, high_priority: bool = False) -> None:
        """Append a timestamped note to prime.md (best-effort)."""
        try:
            prime_path = Path(self._PRIME_MD)
            if not prime_path.parent.exists():
                return

            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            prefix = "[HIGH PRIORITY] " if high_priority else ""
            entry = f"\n\n### {prefix}Sleep Cycle Note ({timestamp})\n{note}\n"

            with open(prime_path, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception:
            logger.debug("Failed to append to prime.md", exc_info=True)

    # ------------------------------------------------------------------
    # code_review (SELF_MODEL_UPDATE) — autonomous blueprint fidelity review
    # ------------------------------------------------------------------

    _CODE_REVIEW_ADAPTER = "code-architect"
    
    @property
    def _PRIME_ENDPOINT(self) -> str:
        return self.config.get_endpoint("prime")

    @property
    def _REVIEW_QUEUE_PATH(self) -> str:
        return str(Path(self.config.KNOWLEDGE_DIR) / "curricula" / "code-architect" / "review_queue.json")

    def _run_code_review(self) -> None:
        """
        Autonomous code review using the code-architect adapter.

        For each live service blueprint:
        1. Run AST summarizer on live source files
        2. Run mechanical pre-check
        3. Build review prompt
        4. Query gaia-prime with code-architect adapter
        5. Parse ReviewResult and surface discrepancies
        """
        try:
            from gaia_common.utils.ast_summarizer import summarize_file
            from gaia_common.utils.blueprint_io import load_blueprint, save_blueprint
            from gaia_common.utils.blueprint_precheck import run_blueprint_precheck
            from gaia_common.utils.review_prompt_builder import ReviewResult, build_review_prompt
        except ImportError:
            logger.debug("Code review dependencies not available, skipping")
            return

        # Check if adapter exists (no point running review without it)
        if not self._adapter_available():
            logger.debug("code-architect adapter not available, skipping code review")
            return

        all_discrepancies: list[dict] = []
        services_reviewed = 0
        review_queue_items: list[dict] = []

        for service_id in self._YAML_BLUEPRINT_SERVICES:
            bp = load_blueprint(service_id)
            if bp is None:
                continue

            source_dir = self._SERVICE_SOURCE_DIRS.get(service_id)
            if source_dir is None or not Path(source_dir).exists():
                source_dir = f"/app/{service_id.replace('-', '_')}"
                if not Path(source_dir).exists():
                    continue

            try:
                result = self._review_service(service_id, bp, source_dir)
                if result is None:
                    continue

                services_reviewed += 1

                # Process discrepancies
                critical_major = [
                    d for d in result.discrepancies
                    if d.severity in ("critical", "major")
                ]

                if critical_major:
                    # Append to blueprint open_questions
                    self._append_review_findings(service_id, bp, critical_major)

                    # Collect for queue
                    for d in critical_major:
                        item = {
                            "service_id": service_id,
                            "dimension": d.dimension,
                            "severity": d.severity,
                            "blueprint_claim": d.blueprint_claim,
                            "code_evidence": d.code_evidence,
                            "recommendation": d.recommendation,
                            "affected_file": d.affected_file,
                            "review_timestamp": result.review_timestamp.isoformat(),
                        }
                        review_queue_items.append(item)
                        all_discrepancies.append(item)

                logger.info(
                    "Code review %s: fidelity=%.0f%% discrepancies=%d (critical/major=%d)",
                    service_id, result.overall_fidelity_score * 100,
                    len(result.discrepancies), len(critical_major),
                )

            except Exception:
                logger.warning("Code review failed for %s", service_id, exc_info=True)

        # Write review queue for Web UI consumption
        if review_queue_items:
            self._write_review_queue(review_queue_items)

        # Summary to prime.md
        if services_reviewed > 0:
            disc_summary = ""
            if all_discrepancies:
                # Group by service for summary
                by_svc: dict[str, list[str]] = {}
                for d in all_discrepancies:
                    by_svc.setdefault(d["service_id"], []).append(
                        f"[{d['severity']}] {d['blueprint_claim']}"
                    )
                parts = [f"{svc}: {items[0]}" for svc, items in by_svc.items()]
                disc_summary = " " + "; ".join(parts[:3])

            note = (
                f"Code review cycle complete. {len(all_discrepancies)} discrepancies found "
                f"across {services_reviewed} services.{disc_summary}"
            )
            self._append_prime_note(note, high_priority=bool(all_discrepancies))

        logger.info(
            "Code review complete: %d services, %d critical/major discrepancies",
            services_reviewed, len(all_discrepancies),
        )

    def _adapter_available(self) -> bool:
        """Check if the code-architect adapter is available."""
        # Check via model pool if available
        if self.model_pool is not None:
            model = getattr(self.model_pool, "_primary_model", None)
            if model is not None and hasattr(model, "health_check"):
                try:
                    return model.health_check()
                except Exception:
                    pass

        # Fallback: check if adapter directory exists
        adapter_dir = Path("/shared/adapters/code-architect")
        return adapter_dir.exists() and any(adapter_dir.iterdir()) if adapter_dir.exists() else False

    def _review_service(self, service_id: str, bp, source_dir: str):
        """Run a full review cycle for one service. Returns ReviewResult or None."""

        from gaia_common.utils.ast_summarizer import summarize_file
        from gaia_common.utils.blueprint_precheck import run_blueprint_precheck
        from gaia_common.utils.review_prompt_builder import ReviewResult, build_review_prompt

        # Step 1: AST summaries
        source_path = Path(source_dir)
        ast_summaries = {}
        for py_file in sorted(source_path.rglob("*.py")):
            if py_file.name.startswith("_") and py_file.name != "__init__.py":
                continue
            try:
                summary = summarize_file(py_file.read_text(), filename=str(py_file))
                rel_name = str(py_file.relative_to(source_path))
                ast_summaries[rel_name] = summary
            except Exception:
                continue

        if not ast_summaries:
            return None

        # Step 2: Pre-check
        precheck_result = run_blueprint_precheck(bp, source_dir)

        # Step 3: Build prompt
        prompt = build_review_prompt(
            bp, ast_summaries, precheck_result,
            review_direction="forward",
            max_prompt_tokens=8000,  # conservative for sleep-cycle review
        )

        # Step 4: Call gaia-prime with adapter
        response_text = self._call_prime_with_adapter(prompt)
        if not response_text:
            return None

        # Step 5: Parse ReviewResult
        try:
            # Extract JSON from response (may be wrapped in markdown)
            json_text = response_text.strip()
            if json_text.startswith("```"):
                lines = json_text.split("\n")
                lines = [ln for ln in lines if not ln.strip().startswith("```")]
                json_text = "\n".join(lines).strip()

            data = _json.loads(json_text)
            return ReviewResult.model_validate(data)
        except Exception:
            logger.warning("Failed to parse review result for %s", service_id, exc_info=True)
            return None

    def _call_prime_with_adapter(self, prompt: str) -> str:
        """Call gaia-prime with the code-architect adapter. Returns response text."""
        # Try model pool first
        if self.model_pool is not None:
            model = getattr(self.model_pool, "_primary_model", None)
            if model is not None and hasattr(model, "create_chat_completion_with_adapter"):
                try:
                    result = model.create_chat_completion_with_adapter(
                        adapter_name=self._CODE_REVIEW_ADAPTER,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=2048,
                        temperature=0.0,
                    )
                    return result.get("choices", [{}])[0].get("message", {}).get("content", "")
                except Exception:
                    logger.warning("Model pool adapter call failed, trying direct HTTP", exc_info=True)

        # Fallback: direct HTTP call
        try:
            import requests

            endpoint = os.getenv("PRIME_ENDPOINT", self._PRIME_ENDPOINT)
            resp = requests.post(
                f"{endpoint}/v1/chat/completions",
                json={
                    "model": self._CODE_REVIEW_ADAPTER,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                    "temperature": 0.0,
                },
                timeout=90,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception:
            logger.warning("Direct HTTP adapter call failed", exc_info=True)
            return ""

    def _append_review_findings(self, service_id: str, bp, discrepancies: list) -> None:
        """Append critical/major review findings to a service blueprint's open_questions."""
        try:
            from gaia_common.utils.blueprint_io import load_blueprint, save_blueprint
            from gaia_common.models.blueprint import Intent

            # Load fresh copy (may have been updated)
            fresh_bp = load_blueprint(service_id, candidate=False)
            if fresh_bp is None:
                fresh_bp = load_blueprint(service_id, candidate=True)
            if fresh_bp is None:
                return

            # Ensure intent exists
            if fresh_bp.intent is None:
                fresh_bp.intent = Intent(purpose="(auto-populated by code review)")

            # Append findings as open questions
            existing = set(fresh_bp.intent.open_questions or [])
            for d in discrepancies:
                question = f"[{d.severity}] {d.dimension}: {d.blueprint_claim} — {d.recommendation}"
                if question not in existing:
                    fresh_bp.intent.open_questions.append(question)

            # Save as candidate (sleep cycle cannot write to live)
            save_blueprint(fresh_bp, candidate=True)
            logger.info("Appended %d findings to %s blueprint open_questions", len(discrepancies), service_id)

        except Exception:
            logger.debug("Failed to append review findings for %s", service_id, exc_info=True)

    def _write_review_queue(self, items: list) -> None:
        """Write review queue items for Web UI consumption."""

        queue_path = Path(self._REVIEW_QUEUE_PATH)
        try:
            # Merge with existing queue
            existing: list = []
            if queue_path.exists():
                existing = _json.loads(queue_path.read_text(encoding="utf-8"))

            # Deduplicate by (service_id, blueprint_claim)
            seen = {(e["service_id"], e["blueprint_claim"]) for e in existing}
            for item in items:
                key = (item["service_id"], item["blueprint_claim"])
                if key not in seen:
                    existing.append(item)
                    seen.add(key)

            queue_path.parent.mkdir(parents=True, exist_ok=True)
            queue_path.write_text(
                _json.dumps(existing, indent=2, default=str),
                encoding="utf-8",
            )
            logger.info("Review queue updated: %d items", len(existing))
        except Exception:
            logger.debug("Failed to write review queue", exc_info=True)

    # ------------------------------------------------------------------
    # wiki_doc_regen (DOC_GENERATION) — blueprint YAML → wiki markdown
    # ------------------------------------------------------------------

    @property
    def _BLUEPRINTS_DIR(self) -> str:
        return str(Path(self.config.KNOWLEDGE_DIR) / "blueprints")

    @property
    def _WIKI_AUTO_DIR(self) -> str:
        return str(Path(self.config.KNOWLEDGE_DIR) / "wiki_auto")

    @property
    def _REGEN_MANIFEST(self) -> str:
        return str(Path(self.config.KNOWLEDGE_DIR) / "wiki_auto" / "_last_regen_manifest.json")

    def _run_wiki_doc_regen(self) -> None:
        """Generate wiki markdown pages from blueprint YAML files.

        Pure YAML → Markdown transformation. No LLM inference required.
        Skips unchanged blueprints (mtime tracking via manifest).
        Writes atomically (tmp + os.replace) to prevent partial files.
        """
        import yaml

        bp_dir = Path(self._BLUEPRINTS_DIR)
        out_dir = Path(self._WIKI_AUTO_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load manifest of last-processed mtimes
        manifest = self._load_regen_manifest()

        index_rows: list[dict] = []
        generated = 0
        skipped = 0

        for bp_path in sorted(bp_dir.glob("*.yaml")):
            service_id = bp_path.stem
            mtime = bp_path.stat().st_mtime

            # Skip if unchanged since last regen
            if manifest.get(service_id) == mtime:
                # Still need index row from cached output
                out_path = out_dir / f"{service_id}.md"
                if out_path.exists():
                    try:
                        data = yaml.safe_load(bp_path.read_text(encoding="utf-8"))
                        if data:
                            index_rows.append(self._index_row_from_data(service_id, data))
                    except Exception:
                        pass
                skipped += 1
                continue

            try:
                raw = bp_path.read_text(encoding="utf-8")
                data = yaml.safe_load(raw)
                if not data or not isinstance(data, dict):
                    logger.debug("Skipping %s: empty or non-dict YAML", bp_path.name)
                    continue
            except Exception:
                logger.warning("Skipping %s: malformed YAML", bp_path.name, exc_info=True)
                continue

            # Render and write
            page = self._render_service_wiki_page(service_id, data)
            self._atomic_write(out_dir / f"{service_id}.md", page)
            index_rows.append(self._index_row_from_data(service_id, data))
            manifest[service_id] = mtime
            generated += 1

        # Write index page
        if index_rows:
            index_page = self._render_wiki_index(index_rows)
            self._atomic_write(out_dir / "index.md", index_page)

        # Persist manifest
        self._save_regen_manifest(manifest)

        logger.info(
            "Wiki doc regen: %d generated, %d skipped (unchanged)",
            generated, skipped,
        )

    @staticmethod
    def _render_service_wiki_page(service_id: str, data: dict) -> str:
        """Render a single service's blueprint data into a wiki markdown page."""
        lines: list[str] = []
        role = data.get("role", service_id)
        lines.append(f"# {service_id}")
        lines.append(f"**Role:** {role}")
        lines.append("")

        # Purpose
        intent = data.get("intent", {})
        if isinstance(intent, dict) and intent.get("purpose"):
            lines.append("## Purpose")
            lines.append("")
            lines.append(str(intent["purpose"]).strip())
            lines.append("")

        # Design Decisions
        decisions = intent.get("design_decisions", []) if isinstance(intent, dict) else []
        if decisions:
            lines.append("## Design Decisions")
            lines.append("")
            for d in decisions:
                lines.append(f"- {d}")
            lines.append("")

        # Runtime
        runtime = data.get("runtime", {})
        if runtime and isinstance(runtime, dict):
            lines.append("## Runtime")
            lines.append("")
            lines.append("| Property | Value |")
            lines.append("|----------|-------|")
            for key in ["port", "base_image", "gpu", "startup_cmd", "health_check", "dockerfile"]:
                val = runtime.get(key)
                if val is not None:
                    lines.append(f"| {key} | `{val}` |")
            lines.append("")

        # Interfaces — Inbound
        interfaces = data.get("interfaces", [])
        if interfaces and isinstance(interfaces, list):
            inbound = [i for i in interfaces if isinstance(i, dict) and i.get("direction") == "inbound"]
            outbound = [i for i in interfaces if isinstance(i, dict) and i.get("direction") == "outbound"]

            if inbound:
                lines.append("## Inbound Endpoints")
                lines.append("")
                lines.append("| ID | Method | Path | Description |")
                lines.append("|----|--------|------|-------------|")
                for ep in inbound:
                    transport = ep.get("transport", {})
                    method = transport.get("method", "—") if isinstance(transport, dict) else "—"
                    path = transport.get("path", "—") if isinstance(transport, dict) else "—"
                    desc = ep.get("description", "")
                    lines.append(f"| {ep.get('id', '—')} | {method} | `{path}` | {desc} |")
                lines.append("")

            if outbound:
                lines.append("## Outbound Connections")
                lines.append("")
                lines.append("| ID | Transport | Target | Description |")
                lines.append("|----|-----------|--------|-------------|")
                for ep in outbound:
                    transport = ep.get("transport", {})
                    if isinstance(transport, dict):
                        t_type = transport.get("type", "—")
                        target = transport.get("target_service", transport.get("path", "—"))
                    else:
                        t_type = "—"
                        target = "—"
                    desc = ep.get("description", "")
                    lines.append(f"| {ep.get('id', '—')} | {t_type} | {target} | {desc} |")
                lines.append("")

        # Dependencies — Services
        deps = data.get("dependencies", {})
        if isinstance(deps, dict):
            svc_deps = deps.get("services", [])
            if svc_deps and isinstance(svc_deps, list):
                lines.append("## Service Dependencies")
                lines.append("")
                lines.append("| Service | Role | Required | Fallback |")
                lines.append("|---------|------|----------|----------|")
                for s in svc_deps:
                    if isinstance(s, dict):
                        lines.append(
                            f"| {s.get('id', '—')} | {s.get('role', '—')} "
                            f"| {s.get('required', '—')} | {s.get('fallback', 'none')} |"
                        )
                lines.append("")

            # Volumes
            volumes = deps.get("volumes", [])
            if volumes and isinstance(volumes, list):
                lines.append("## Volume Mounts")
                lines.append("")
                lines.append("| Name | Access | Mount Path | Purpose |")
                lines.append("|------|--------|------------|---------|")
                for v in volumes:
                    if isinstance(v, dict):
                        lines.append(
                            f"| {v.get('name', '—')} | {v.get('access', '—')} "
                            f"| `{v.get('mount_path', '—')}` | {v.get('purpose', '—')} |"
                        )
                lines.append("")

            # External APIs
            ext_apis = deps.get("external_apis", [])
            if ext_apis and isinstance(ext_apis, list):
                lines.append("## External APIs")
                lines.append("")
                lines.append("| Name | Purpose | Required |")
                lines.append("|------|---------|----------|")
                for api in ext_apis:
                    if isinstance(api, dict):
                        lines.append(
                            f"| {api.get('name', '—')} | {api.get('purpose', '—')} "
                            f"| {api.get('required', '—')} |"
                        )
                lines.append("")

        # Failure Modes
        failures = data.get("failure_modes", [])
        if failures and isinstance(failures, list):
            lines.append("## Failure Modes")
            lines.append("")
            for fm in failures:
                if not isinstance(fm, dict):
                    continue
                severity = fm.get("severity", "unknown")
                condition = fm.get("condition", "Unknown condition")
                response = fm.get("response", "No response defined")
                auto = fm.get("auto_recovers", False)

                admonition = "warning" if severity in ("degraded", "partial") else "danger"
                lines.append(f'!!! {admonition} "{condition}"')
                lines.append(f"    **Severity:** {severity} | **Auto-recovers:** {'yes' if auto else 'no'}")
                lines.append(f"    {response}")
                lines.append("")

        # Footer
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines.append("---")
        lines.append(f"*Auto-generated from `{service_id}.yaml` by wiki_doc_regen sleep task at {timestamp}.*")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _render_wiki_index(rows: list[dict]) -> str:
        """Render the auto-generated service map index page."""
        lines: list[str] = []
        lines.append("# Auto-Generated Service Map")
        lines.append("")
        lines.append("This page is regenerated automatically from blueprint YAML files")
        lines.append("during GAIA's sleep cycle. Do not edit manually.")
        lines.append("")
        lines.append("| Service | Role | Port | GPU | Status |")
        lines.append("|---------|------|------|-----|--------|")
        for row in sorted(rows, key=lambda r: r.get("service_id", "")):
            sid = row.get("service_id", "—")
            lines.append(
                f"| [{sid}]({sid}.md) | {row.get('role', '—')} "
                f"| {row.get('port', '—')} | {row.get('gpu', '—')} "
                f"| {row.get('status', '—')} |"
            )
        lines.append("")

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines.append("---")
        lines.append(f"*Auto-generated by wiki_doc_regen sleep task at {timestamp}.*")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _index_row_from_data(service_id: str, data: dict) -> dict:
        """Extract a summary row from blueprint data for the index page."""
        runtime = data.get("runtime", {}) if isinstance(data.get("runtime"), dict) else {}
        return {
            "service_id": service_id,
            "role": data.get("role", "—"),
            "port": runtime.get("port", "—"),
            "gpu": runtime.get("gpu", "—"),
            "status": data.get("service_status", "—"),
        }

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        """Write content atomically via tmp file + os.replace."""
        tmp_path = path.with_suffix(".tmp")
        try:
            tmp_path.write_text(content, encoding="utf-8")
            os.replace(str(tmp_path), str(path))
        except Exception:
            # Clean up tmp on failure
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

    def _load_regen_manifest(self) -> dict:
        """Load the mtime manifest for incremental regen."""
        manifest_path = Path(self._REGEN_MANIFEST)
        if manifest_path.exists():
            try:
                return _json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_regen_manifest(self, manifest: dict) -> None:
        """Save the mtime manifest for incremental regen."""
        manifest_path = Path(self._REGEN_MANIFEST)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(manifest_path, _json.dumps(manifest, indent=2))

    # ------------------------------------------------------------------
    # knowledge_research (KNOWLEDGE_ACQUISITION) — fill knowledge gaps
    # ------------------------------------------------------------------

    _MCP_ENDPOINT = "http://gaia-mcp:8765/jsonrpc"

    def _run_knowledge_research(self) -> None:
        """Research knowledge gaps identified by thought seeds.

        Reads unreviewed seeds with seed_type == "knowledge_gap", uses MCP
        web_search + web_fetch to gather information, saves results to
        /knowledge/research/, indexes them with confidence_tier="researched",
        and archives the seed.

        No GPU needed — only HTTP calls and file I/O.
        """
        from gaia_core.cognition.thought_seed import (
            list_unreviewed_seeds,
            archive_seed,
        )

        ep_config = self.config.constants.get("EPISTEMIC_DRIVE", {})
        if not ep_config.get("enabled", True):
            logger.debug("Epistemic drive disabled, skipping knowledge research")
            return

        max_per_cycle = ep_config.get("max_research_per_cycle", 3)
        trusted_only = ep_config.get("research_trusted_domains_only", True)

        # Gather knowledge gap seeds
        all_seeds = list_unreviewed_seeds()
        gap_seeds = [
            (path, data) for path, data in all_seeds
            if data.get("seed_type") == "knowledge_gap"
        ]

        if not gap_seeds:
            logger.debug("No knowledge gap seeds to research")
            return

        logger.info("Knowledge research: %d gap seeds found, processing up to %d",
                     len(gap_seeds), max_per_cycle)

        researched = 0
        research_dir = Path("/knowledge/research")
        research_dir.mkdir(parents=True, exist_ok=True)

        for seed_path, seed_data in gap_seeds[:max_per_cycle]:
            seed_text = seed_data.get("seed", "")
            # Extract the topic from "Knowledge gap — <topic>" pattern
            topic = seed_text
            if "—" in topic:
                topic = topic.split("—", 1)[1].strip()
            elif "-" in topic and topic.lower().startswith("knowledge gap"):
                topic = topic.split("-", 1)[1].strip()

            # Strip trailing periods and common suffixes
            topic = topic.rstrip(". ")
            if not topic:
                archive_seed(seed_path.name)
                continue

            try:
                content = self._research_topic(topic, trusted_only)
                if content:
                    # Save to research directory
                    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")[:60]
                    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
                    filename = f"{timestamp}_{slug}.md"
                    filepath = research_dir / filename

                    filepath.write_text(content, encoding="utf-8")
                    logger.info("Knowledge research: saved %s", filename)

                    # Index with confidence_tier="researched"
                    self._index_research_document(str(filepath))
                    researched += 1

                archive_seed(seed_path.name)

            except Exception:
                logger.warning("Knowledge research failed for topic: %s",
                               topic, exc_info=True)

        if researched > 0:
            # Write council note so Prime knows about new knowledge
            try:
                from gaia_core.cognition.council_notes import CouncilNoteManager
                cn = CouncilNoteManager(self.config)
                cn.write_note(
                    user_prompt="[System] Knowledge research complete",
                    lite_response=(
                        f"Researched {researched} knowledge gap(s) during sleep. "
                        f"New documents saved to /knowledge/research/ with "
                        f"confidence_tier='researched'."
                    ),
                    escalation_reason="New auto-researched knowledge available",
                    session_id="sleep-knowledge-research",
                )
            except Exception:
                logger.debug("Could not write council note for knowledge research",
                             exc_info=True)

        logger.info("Knowledge research complete: %d topics researched", researched)

    def _research_topic(self, topic: str, trusted_only: bool = True) -> str:
        """Research a topic via MCP web_search + web_fetch. Returns markdown content."""
        import requests

        mcp_endpoint = os.getenv("MCP_ENDPOINT", self._MCP_ENDPOINT)

        # Step 1: web_search
        search_payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "web_search",
                "arguments": {"query": topic},
            },
            "id": f"research_search_{int(time.time())}",
        }

        try:
            resp = requests.post(mcp_endpoint, json=search_payload, timeout=15)
            resp.raise_for_status()
            search_result = resp.json().get("result", {})
        except Exception:
            logger.warning("Knowledge research: web_search failed for '%s'", topic)
            return ""

        # Extract best URL from search results
        content_items = search_result.get("content", [])
        if not content_items:
            return ""

        # Parse the search output to find URLs
        search_text = ""
        for item in content_items:
            if isinstance(item, dict) and item.get("type") == "text":
                search_text = item.get("text", "")
                break

        if not search_text:
            return ""

        # Extract first URL from search results
        url_match = re.search(r"https?://[^\s\)\"']+", search_text)
        if not url_match:
            return f"# {topic}\n\nSearch results (no fetchable URL found):\n\n{search_text[:2000]}"

        fetch_url = url_match.group(0)

        # If trusted_only, check domain against trusted/reliable lists
        if trusted_only:
            web_cfg = self.config.constants.get("WEB_RESEARCH", {})
            trusted = set(web_cfg.get("trusted_domains", []))
            reliable = set(web_cfg.get("reliable_domains", []))
            allowed = trusted | reliable

            from urllib.parse import urlparse
            domain = urlparse(fetch_url).netloc.lstrip("www.")
            if not any(domain.endswith(d) for d in allowed):
                # Return just the search summary without fetching
                return (
                    f"# {topic}\n\n"
                    f"*Auto-researched (search only — domain not in trusted list)*\n\n"
                    f"{search_text[:2000]}"
                )

        # Step 2: web_fetch
        fetch_payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "web_fetch",
                "arguments": {"url": fetch_url},
            },
            "id": f"research_fetch_{int(time.time())}",
        }

        try:
            resp = requests.post(mcp_endpoint, json=fetch_payload, timeout=20)
            resp.raise_for_status()
            fetch_result = resp.json().get("result", {})
        except Exception:
            logger.warning("Knowledge research: web_fetch failed for '%s'", fetch_url)
            return (
                f"# {topic}\n\n"
                f"*Auto-researched (fetch failed)*\n\n"
                f"Source: {fetch_url}\n\n"
                f"Search summary:\n{search_text[:2000]}"
            )

        # Extract fetched content
        fetch_text = ""
        for item in fetch_result.get("content", []):
            if isinstance(item, dict) and item.get("type") == "text":
                fetch_text = item.get("text", "")
                break

        # Compose final document
        # Truncate to reasonable size
        if len(fetch_text) > 8000:
            fetch_text = fetch_text[:8000] + "\n\n[...truncated]"

        return (
            f"# {topic}\n\n"
            f"*Auto-researched during sleep cycle — confidence tier: researched*\n\n"
            f"Source: {fetch_url}\n\n"
            f"---\n\n"
            f"{fetch_text}"
        )

    def _index_research_document(self, filepath: str) -> None:
        """Index a research document with confidence_tier='researched'."""
        try:
            from gaia_common.utils.vector_indexer import VectorIndexer

            # Use "system" knowledge base but tag as "researched" tier
            indexer = VectorIndexer.instance("system")
            indexer.add_document(filepath, confidence_tier="researched")
            logger.info("Indexed research document: %s", filepath)
        except Exception:
            logger.warning("Failed to index research document: %s",
                           filepath, exc_info=True)

    def _run_code_evolution(self) -> None:
        """Generate code evolution snapshot for temporal self-awareness."""
        try:
            from gaia_common.utils.code_evolution import generate_code_evolution_snapshot

            shared_dir = os.getenv("SHARED_DIR", "/shared")
            output_path = os.path.join(shared_dir, "self_model", "code_evolution.md")
            generate_code_evolution_snapshot(
                project_root="/gaia/GAIA_Project",
                output_path=output_path,
            )
            logger.info("Code evolution snapshot generated at %s", output_path)
        except Exception:
            logger.error("Code evolution snapshot failed", exc_info=True)
            raise

    def _emit_task_exec(
        self, task_id: str, task_type: str, elapsed: float, success: bool, error: str = "",
    ) -> None:
        """Emit a task_exec event to the timeline store (best-effort)."""
        if self._timeline is not None:
            try:
                data = {
                    "task_id": task_id,
                    "task_type": task_type,
                    "duration_s": round(elapsed, 1),
                    "success": success,
                }
                if error:
                    data["error"] = error[:200]
                self._timeline.append("task_exec", data)
            except Exception:
                logger.debug("Timeline task_exec emit failed", exc_info=True)

    # ------------------------------------------------------------------
    # Samvega Introspection
    # ------------------------------------------------------------------

    def _run_samvega_introspection(self) -> None:
        """Review unreviewed samvega artifacts, boost repeated patterns, flag tier-5."""
        from gaia_core.cognition.samvega import (
            list_unreviewed_artifacts,
            update_artifact,
        )
        from datetime import datetime, timezone
        from collections import defaultdict

        artifacts = list_unreviewed_artifacts()
        if not artifacts:
            logger.info("Samvega introspection: no unreviewed artifacts")
            return

        samvega_cfg = self.config.constants.get("SAMVEGA", {})
        tier5_threshold = samvega_cfg.get("tier5_promotion_threshold", 0.7)
        repeated_boost = samvega_cfg.get("weight_multipliers", {}).get("repeated_domain", 1.3)

        # Group by root_cause for cluster detection
        clusters: dict[str, list[tuple]] = defaultdict(list)
        for path, data in artifacts:
            rc = data.get("root_cause", "").strip().lower()
            clusters[rc].append((path, data))

        reviewed = 0
        promoted = 0
        now_iso = datetime.now(timezone.utc).isoformat()

        for root_cause, group in clusters.items():
            is_cluster = len(group) >= 2
            for path, data in group:
                if is_cluster:
                    boosted = min(1.0, data.get("weight", 0) * repeated_boost)
                    data["weight"] = boosted

                data["reviewed"] = True
                data["reviewed_at"] = now_iso

                if data.get("weight", 0) >= tier5_threshold:
                    data["promoted_to_tier5"] = True
                    promoted += 1

                update_artifact(path.name, data)
                reviewed += 1

        logger.info(
            "Samvega introspection: reviewed %d artifacts, %d promoted to tier-5, %d clusters",
            reviewed, promoted, sum(1 for g in clusters.values() if len(g) >= 2),
        )

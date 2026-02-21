"""
gaia_common/utils/promotion_readiness.py

Structured promotion readiness assessment for GAIA candidate services.

Runs a multi-check validation suite and produces a PromotionReadinessReport
that GAIA (or a human) can use to decide whether a service is ready to promote.

Usage:
    from gaia_common.utils.promotion_readiness import assess_promotion_readiness
    report = assess_promotion_readiness("gaia-audio")
    print(report.to_markdown())

Consumers:
  - sleep_task_scheduler: promotion_readiness task (auto-assess during sleep)
  - MCP tools: on-demand assessment during conversation
  - gaia_promote_executor.py: validates before executing promotion
"""

from __future__ import annotations

import filecmp
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

logger = logging.getLogger("GAIA.PromotionReadiness")


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class ReadinessCheck:
    """Result of a single readiness check."""
    name: str
    status: Literal["pass", "fail", "warn", "skip"]
    detail: str
    blocking: bool  # If True, a "fail" status blocks promotion


@dataclass
class PromotionReadinessReport:
    """Full promotion readiness assessment for a candidate service."""
    service_id: str
    timestamp: datetime
    verdict: Literal["ready", "ready_with_warnings", "not_ready"]
    checks: List[ReadinessCheck]
    recommendation: str
    pipeline_cmd: str = ""

    def to_markdown(self) -> str:
        """Render as human-readable markdown."""
        status_icon = {"pass": "+", "fail": "x", "warn": "!", "skip": "-"}
        lines = [
            f"# Promotion Readiness: {self.service_id}",
            f"**Assessed:** {self.timestamp.isoformat()}",
            f"**Verdict:** {self.verdict.upper()}",
            "",
            "## Checks",
            "",
        ]
        for check in self.checks:
            icon = status_icon.get(check.status, "?")
            blocking = " (blocking)" if check.blocking and check.status == "fail" else ""
            lines.append(f"[{icon}] {check.name}: {check.detail}{blocking}")
        lines.extend([
            "",
            "## Recommendation",
            self.recommendation,
        ])
        if self.pipeline_cmd:
            lines.extend([
                "",
                "## Promotion Command",
                f"```bash",
                self.pipeline_cmd,
                "```",
            ])
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "service_id": self.service_id,
            "timestamp": self.timestamp.isoformat(),
            "verdict": self.verdict,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "detail": c.detail,
                    "blocking": c.blocking,
                }
                for c in self.checks
            ],
            "recommendation": self.recommendation,
            "pipeline_cmd": self.pipeline_cmd,
        }

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "pass")

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "fail")

    @property
    def warn_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "warn")

    @property
    def blocking_failures(self) -> List[ReadinessCheck]:
        return [c for c in self.checks if c.status == "fail" and c.blocking]


# ── Public API ───────────────────────────────────────────────────────────────


def assess_promotion_readiness(
    service_id: str,
    project_root: str = "/gaia/GAIA_Project",
) -> PromotionReadinessReport:
    """Run all promotion readiness checks for a candidate service.

    Args:
        service_id: Service to assess (e.g. "gaia-audio")
        project_root: Root of the GAIA project directory

    Returns:
        PromotionReadinessReport with verdict and individual check results
    """
    root = Path(project_root)
    checks: List[ReadinessCheck] = []

    # 1. Candidate directory exists
    checks.append(_check_candidate_dir(service_id, root))

    # 2. Blueprint exists
    checks.append(_check_blueprint_exists(service_id, root))

    # 3. Blueprint schema validates
    checks.append(_check_blueprint_validates(service_id))

    # 4. Blueprint pre-check
    checks.append(_check_blueprint_precheck(service_id, root))

    # 5. gaia-common sync
    checks.append(_check_common_sync(service_id, root))

    # 6. Dockerfile exists with HEALTHCHECK
    checks.append(_check_dockerfile(service_id, root))

    # 7. Compose service defined
    checks.append(_check_compose_service(service_id, root))

    # 8. Source files lint-clean
    checks.append(_check_lint(service_id, root))

    # 9. Test files exist
    checks.append(_check_tests_exist(service_id, root))

    # Determine verdict
    blocking_fails = [c for c in checks if c.status == "fail" and c.blocking]
    warnings = [c for c in checks if c.status == "warn"]

    if blocking_fails:
        verdict = "not_ready"
        recommendation = (
            f"{len(blocking_fails)} blocking issue(s) must be resolved: "
            + "; ".join(c.name for c in blocking_fails)
        )
    elif warnings:
        verdict = "ready_with_warnings"
        recommendation = (
            f"Service is promotable with {len(warnings)} warning(s). "
            "Review warnings before proceeding."
        )
    else:
        verdict = "ready"
        recommendation = "All checks pass. Service is ready for promotion."

    pipeline_cmd = f"./scripts/promote_pipeline.sh --services {service_id}"

    return PromotionReadinessReport(
        service_id=service_id,
        timestamp=datetime.now(timezone.utc),
        verdict=verdict,
        checks=checks,
        recommendation=recommendation,
        pipeline_cmd=pipeline_cmd,
    )


# ── Individual checks ────────────────────────────────────────────────────────


def _check_candidate_dir(service_id: str, root: Path) -> ReadinessCheck:
    """Check that the candidate directory exists with source files."""
    candidate_dir = root / "candidates" / service_id
    if not candidate_dir.exists():
        return ReadinessCheck(
            name="candidate_directory",
            status="fail",
            detail=f"Candidate directory not found: {candidate_dir}",
            blocking=True,
        )

    py_files = list(candidate_dir.rglob("*.py"))
    if not py_files:
        return ReadinessCheck(
            name="candidate_directory",
            status="fail",
            detail=f"Candidate directory exists but contains no Python files",
            blocking=True,
        )

    return ReadinessCheck(
        name="candidate_directory",
        status="pass",
        detail=f"Found {len(py_files)} Python files in {candidate_dir}",
        blocking=True,
    )


def _check_blueprint_exists(service_id: str, root: Path) -> ReadinessCheck:
    """Check that a blueprint exists (candidate or live)."""
    candidate_bp = root / "knowledge" / "blueprints" / "candidates" / f"{service_id}.yaml"
    live_bp = root / "knowledge" / "blueprints" / f"{service_id}.yaml"

    if candidate_bp.exists():
        return ReadinessCheck(
            name="blueprint_exists",
            status="pass",
            detail=f"Candidate blueprint found at {candidate_bp.name}",
            blocking=False,
        )
    elif live_bp.exists():
        return ReadinessCheck(
            name="blueprint_exists",
            status="pass",
            detail=f"Live blueprint found at {live_bp.name}",
            blocking=False,
        )
    else:
        return ReadinessCheck(
            name="blueprint_exists",
            status="warn",
            detail="No blueprint found. Generate one with blueprint_generator before promotion.",
            blocking=False,
        )


def _check_blueprint_validates(service_id: str) -> ReadinessCheck:
    """Check that the blueprint passes schema validation."""
    try:
        from gaia_common.utils.blueprint_io import (
            load_blueprint,
            validate_candidate_blueprint,
        )
    except ImportError:
        return ReadinessCheck(
            name="blueprint_validates",
            status="skip",
            detail="blueprint_io not available",
            blocking=False,
        )

    bp = load_blueprint(service_id, candidate=True)
    if bp is None:
        bp = load_blueprint(service_id, candidate=False)
    if bp is None:
        return ReadinessCheck(
            name="blueprint_validates",
            status="skip",
            detail="No blueprint to validate",
            blocking=False,
        )

    try:
        result = validate_candidate_blueprint(service_id)
        if result.passed:
            return ReadinessCheck(
                name="blueprint_validates",
                status="pass",
                detail="Blueprint schema validation passed",
                blocking=False,
            )
        else:
            issues = "; ".join(result.issues[:3]) if hasattr(result, "issues") else "validation failed"
            return ReadinessCheck(
                name="blueprint_validates",
                status="warn",
                detail=f"Blueprint validation issues: {issues}",
                blocking=False,
            )
    except Exception as exc:
        return ReadinessCheck(
            name="blueprint_validates",
            status="warn",
            detail=f"Blueprint validation error: {exc}",
            blocking=False,
        )


def _check_blueprint_precheck(service_id: str, root: Path) -> ReadinessCheck:
    """Run mechanical pre-check of blueprint claims against source code."""
    try:
        from gaia_common.utils.blueprint_io import load_blueprint
        from gaia_common.utils.blueprint_precheck import run_blueprint_precheck
    except ImportError:
        return ReadinessCheck(
            name="blueprint_precheck",
            status="skip",
            detail="blueprint_precheck not available",
            blocking=False,
        )

    bp = load_blueprint(service_id, candidate=True)
    if bp is None:
        bp = load_blueprint(service_id, candidate=False)
    if bp is None:
        return ReadinessCheck(
            name="blueprint_precheck",
            status="skip",
            detail="No blueprint to pre-check",
            blocking=False,
        )

    # Find source directory
    pkg_name = service_id.replace("-", "_")
    source_dir = root / "candidates" / service_id / pkg_name
    if not source_dir.exists():
        source_dir = root / service_id / pkg_name
    if not source_dir.exists():
        return ReadinessCheck(
            name="blueprint_precheck",
            status="skip",
            detail=f"Source directory not found for pre-check",
            blocking=False,
        )

    result = run_blueprint_precheck(bp, str(source_dir))
    missing = [i for i in result.items if i.status == "missing"]
    total = len(result.items)
    found = sum(1 for i in result.items if i.status == "found")

    if not missing:
        return ReadinessCheck(
            name="blueprint_precheck",
            status="pass",
            detail=f"All {total} blueprint claims verified in source",
            blocking=False,
        )
    else:
        return ReadinessCheck(
            name="blueprint_precheck",
            status="warn",
            detail=f"{found}/{total} claims verified; {len(missing)} missing: "
                   + ", ".join(m.blueprint_claim[:50] for m in missing[:3]),
            blocking=False,
        )


def _check_common_sync(service_id: str, root: Path) -> ReadinessCheck:
    """Check that gaia-common is in sync between candidate and production."""
    candidate_cp = root / "candidates" / "gaia-common" / "gaia_common" / "protocols" / "cognition_packet.py"
    live_cp = root / "gaia-common" / "gaia_common" / "protocols" / "cognition_packet.py"

    if not candidate_cp.exists() or not live_cp.exists():
        return ReadinessCheck(
            name="common_sync",
            status="skip",
            detail="Cannot compare gaia-common (files missing)",
            blocking=True,
        )

    if filecmp.cmp(str(candidate_cp), str(live_cp), shallow=False):
        return ReadinessCheck(
            name="common_sync",
            status="pass",
            detail="CognitionPacket in sync between candidate and production",
            blocking=True,
        )
    else:
        return ReadinessCheck(
            name="common_sync",
            status="fail",
            detail="CognitionPacket differs between candidate and production. Promote gaia-common first.",
            blocking=True,
        )


def _check_dockerfile(service_id: str, root: Path) -> ReadinessCheck:
    """Check that a Dockerfile exists and has a HEALTHCHECK."""
    dockerfile = root / "candidates" / service_id / "Dockerfile"
    if not dockerfile.exists():
        return ReadinessCheck(
            name="dockerfile",
            status="fail",
            detail=f"No Dockerfile found at {dockerfile}",
            blocking=True,
        )

    content = dockerfile.read_text(encoding="utf-8")
    has_healthcheck = "HEALTHCHECK" in content

    if has_healthcheck:
        return ReadinessCheck(
            name="dockerfile",
            status="pass",
            detail="Dockerfile found with HEALTHCHECK",
            blocking=True,
        )
    else:
        return ReadinessCheck(
            name="dockerfile",
            status="warn",
            detail="Dockerfile found but missing HEALTHCHECK directive",
            blocking=True,
        )


def _check_compose_service(service_id: str, root: Path) -> ReadinessCheck:
    """Check that the service is defined in a compose file."""
    # Check candidate compose
    candidate_compose = root / "docker-compose.candidate.yml"
    live_compose = root / "docker-compose.yml"

    candidate_name = f"{service_id}-candidate"
    found_in = []

    for compose_file, search_names in [
        (candidate_compose, [candidate_name, service_id]),
        (live_compose, [service_id]),
    ]:
        if compose_file.exists():
            content = compose_file.read_text(encoding="utf-8")
            for name in search_names:
                # Look for the service key in YAML (indented as a top-level service)
                if f"  {name}:" in content or f"\n  {name}:" in content:
                    found_in.append(f"{compose_file.name} ({name})")

    if not found_in:
        return ReadinessCheck(
            name="compose_service",
            status="warn",
            detail="Service not found in any compose file. Add to docker-compose.yml before promotion.",
            blocking=False,
        )

    return ReadinessCheck(
        name="compose_service",
        status="pass",
        detail=f"Found in: {', '.join(found_in)}",
        blocking=False,
    )


def _check_lint(service_id: str, root: Path) -> ReadinessCheck:
    """Check if source files pass ruff linting (best-effort)."""
    pkg_name = service_id.replace("-", "_")
    source_dir = root / "candidates" / service_id / pkg_name

    if not source_dir.exists():
        return ReadinessCheck(
            name="lint_clean",
            status="skip",
            detail="Source directory not found",
            blocking=False,
        )

    try:
        result = subprocess.run(
            ["ruff", "check", str(source_dir), "--select", "E,F,W"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        issue_lines = [
            ln for ln in result.stdout.strip().splitlines()
            if ln.strip() and not ln.startswith("Found") and not ln.startswith("All checks")
        ]
        issue_count = len(issue_lines)
        if result.returncode == 0 or issue_count == 0:
            return ReadinessCheck(
                name="lint_clean",
                status="pass",
                detail="Source passes ruff lint (E,F,W rules)",
                blocking=False,
            )
        else:
            return ReadinessCheck(
                name="lint_clean",
                status="warn",
                detail=f"Ruff found {issue_count} issue(s)",
                blocking=False,
            )
    except FileNotFoundError:
        return ReadinessCheck(
            name="lint_clean",
            status="skip",
            detail="ruff not installed on this host",
            blocking=False,
        )
    except Exception as exc:
        return ReadinessCheck(
            name="lint_clean",
            status="skip",
            detail=f"Lint check failed: {exc}",
            blocking=False,
        )


def _check_tests_exist(service_id: str, root: Path) -> ReadinessCheck:
    """Check that test files exist for the service."""
    candidate_dir = root / "candidates" / service_id

    test_files = (
        list(candidate_dir.rglob("test_*.py"))
        + list(candidate_dir.rglob("*_test.py"))
    )
    # Deduplicate
    test_files = list({str(f) for f in test_files})

    if not test_files:
        return ReadinessCheck(
            name="tests_exist",
            status="warn",
            detail="No test files found. Consider adding tests before promotion.",
            blocking=False,
        )

    return ReadinessCheck(
        name="tests_exist",
        status="pass",
        detail=f"Found {len(test_files)} test file(s)",
        blocking=False,
    )

"""
Tests for gaia_common.utils.review_prompt_builder

Tests prompt assembly from mock blueprint + AST summaries + pre-check results,
ReviewResult schema validation, and token budget truncation.
"""

import json
from datetime import datetime, timezone

import pytest

from gaia_common.models.blueprint import (
    BlueprintMeta,
    BlueprintModel,
    BlueprintStatus,
    Dependencies,
    FailureMode,
    GeneratedBy,
    HttpRestInterface,
    Intent,
    Interface,
    InterfaceDirection,
    InterfaceStatus,
    Runtime,
    ServiceDependency,
    ServiceStatus,
    Severity,
    WebSocketInterface,
)
from gaia_common.utils.ast_summarizer import (
    ASTSummary,
    ClassInfo,
    ConstantInfo,
    EndpointInfo,
    ErrorHandlerInfo,
    FunctionInfo,
    HttpCallInfo,
)
from gaia_common.utils.blueprint_precheck import PreCheckItem, PreCheckResult, PreCheckSummary
from gaia_common.utils.review_prompt_builder import (
    DiscrepancyItem,
    OpenQuestionUpdate,
    ReviewResult,
    build_review_prompt,
)


# ── Test fixtures ────────────────────────────────────────────────────────────

def _make_blueprint() -> BlueprintModel:
    return BlueprintModel(
        id="test-service",
        version="0.1.0",
        role="Test Service",
        service_status=ServiceStatus.CANDIDATE,
        runtime=Runtime(port=8000),
        interfaces=[
            Interface(
                id="health",
                direction=InterfaceDirection.INBOUND,
                transport=HttpRestInterface(path="/health", method="GET"),
                description="Health check",
                status=InterfaceStatus.ACTIVE,
            ),
            Interface(
                id="process",
                direction=InterfaceDirection.INBOUND,
                transport=HttpRestInterface(
                    path="/process", method="POST",
                    input_schema="ProcessRequest", output_schema="ProcessResponse",
                ),
                description="Process a request",
                status=InterfaceStatus.ACTIVE,
            ),
        ],
        dependencies=Dependencies(
            services=[
                ServiceDependency(id="gaia-core", role="brain", required=True),
            ],
        ),
        failure_modes=[
            FailureMode(
                condition="core unavailable",
                response="return 503",
                severity=Severity.DEGRADED,
            ),
        ],
        intent=Intent(
            purpose="Test service for review prompt builder testing",
            cognitive_role="Tester",
            design_decisions=["Keep it simple", "Use FastAPI"],
            open_questions=["Should we add caching?"],
        ),
        meta=BlueprintMeta(
            status=BlueprintStatus.CANDIDATE,
            generated_by=GeneratedBy.MANUAL_SEED,
        ),
    )


def _make_ast_summary() -> ASTSummary:
    return ASTSummary(
        module_docstring="Test service main module",
        classes=[
            ClassInfo(
                name="ProcessRequest",
                bases=["BaseModel"],
                docstring="Request model",
                methods=[],
                line=10,
            ),
        ],
        functions=[
            FunctionInfo(
                name="health_check",
                params=[],
                return_type="dict",
                decorators=['router.get("/health")'],
                is_async=True,
                line=20,
            ),
            FunctionInfo(
                name="process",
                params=["request: ProcessRequest"],
                return_type="ProcessResponse",
                decorators=['router.post("/process")'],
                is_async=True,
                line=25,
            ),
        ],
        endpoints=[
            EndpointInfo(method="GET", path="/health", function_name="health_check", line=20),
            EndpointInfo(method="POST", path="/process", function_name="process", line=25),
        ],
        enums=[],
        constants=[
            ConstantInfo(name="DEFAULT_TIMEOUT", value="5000", line=5),
        ],
        gaia_imports=["from gaia_common.models import BaseRequest"],
        error_handlers=[
            ErrorHandlerInfo(
                exception_types=["httpx.ConnectError"],
                status_code=503,
                enclosing_function="process",
                line=30,
            ),
        ],
        http_calls=[
            HttpCallInfo(
                call_method="post",
                url_or_path="http://gaia-core:8000/turn",
                enclosing_function="process",
                line=28,
            ),
        ],
        filename="main.py",
    )


def _make_precheck_result() -> PreCheckResult:
    return PreCheckResult(
        service_id="test-service",
        timestamp=datetime.now(timezone.utc),
        items=[
            PreCheckItem(
                category="endpoint",
                blueprint_claim="GET /health",
                status="found",
                source_file="main.py",
                detail="→ main.py:20",
            ),
            PreCheckItem(
                category="endpoint",
                blueprint_claim="POST /process",
                status="found",
                source_file="main.py",
                detail="→ main.py:25",
            ),
            PreCheckItem(
                category="dependency",
                blueprint_claim="gaia-core (brain)",
                status="found",
                source_file="main.py",
                detail="→ main.py:28",
            ),
            PreCheckItem(
                category="failure_mode",
                blueprint_claim="core unavailable",
                status="found",
                source_file="main.py",
                detail="→ ConnectError handler in main.py:30",
            ),
        ],
        summary=PreCheckSummary(total=4, found=4, missing=0, diverged=0),
    )


# ── Tests ────────────────────────────────────────────────────────────────────

class TestBuildReviewPrompt:
    """Test the main prompt assembly."""

    def test_prompt_contains_system_section(self):
        bp = _make_blueprint()
        summaries = {"main.py": _make_ast_summary()}
        precheck = _make_precheck_result()
        prompt = build_review_prompt(bp, summaries, precheck)

        assert "SYSTEM:" in prompt
        assert "code reviewer for the GAIA AI system" in prompt

    def test_prompt_contains_user_section(self):
        bp = _make_blueprint()
        summaries = {"main.py": _make_ast_summary()}
        precheck = _make_precheck_result()
        prompt = build_review_prompt(bp, summaries, precheck)

        assert "USER:" in prompt

    def test_prompt_contains_blueprint_header(self):
        bp = _make_blueprint()
        summaries = {"main.py": _make_ast_summary()}
        precheck = _make_precheck_result()
        prompt = build_review_prompt(bp, summaries, precheck)

        assert "## Blueprint: test-service" in prompt

    def test_prompt_contains_intent(self):
        bp = _make_blueprint()
        summaries = {"main.py": _make_ast_summary()}
        precheck = _make_precheck_result()
        prompt = build_review_prompt(bp, summaries, precheck)

        assert "### Intent" in prompt
        assert "Test service for review prompt builder testing" in prompt
        assert "Cognitive role: Tester" in prompt

    def test_prompt_contains_interfaces(self):
        bp = _make_blueprint()
        summaries = {"main.py": _make_ast_summary()}
        precheck = _make_precheck_result()
        prompt = build_review_prompt(bp, summaries, precheck)

        assert "### Interfaces" in prompt
        assert "/health" in prompt
        assert "/process" in prompt

    def test_prompt_contains_dependencies(self):
        bp = _make_blueprint()
        summaries = {"main.py": _make_ast_summary()}
        precheck = _make_precheck_result()
        prompt = build_review_prompt(bp, summaries, precheck)

        assert "### Dependencies" in prompt
        assert "gaia-core" in prompt

    def test_prompt_contains_failure_modes(self):
        bp = _make_blueprint()
        summaries = {"main.py": _make_ast_summary()}
        precheck = _make_precheck_result()
        prompt = build_review_prompt(bp, summaries, precheck)

        assert "### Failure Modes" in prompt
        assert "core unavailable" in prompt

    def test_prompt_contains_open_questions(self):
        bp = _make_blueprint()
        summaries = {"main.py": _make_ast_summary()}
        precheck = _make_precheck_result()
        prompt = build_review_prompt(bp, summaries, precheck)

        assert "### Open Questions" in prompt
        assert "caching" in prompt

    def test_prompt_contains_precheck_results(self):
        bp = _make_blueprint()
        summaries = {"main.py": _make_ast_summary()}
        precheck = _make_precheck_result()
        prompt = build_review_prompt(bp, summaries, precheck)

        assert "## Mechanical Pre-Check Results" in prompt
        assert "[FOUND]" in prompt

    def test_prompt_contains_ast_summaries(self):
        bp = _make_blueprint()
        summaries = {"main.py": _make_ast_summary()}
        precheck = _make_precheck_result()
        prompt = build_review_prompt(bp, summaries, precheck)

        assert "## Source Files Under Review" in prompt
        assert "### File: main.py" in prompt

    def test_prompt_contains_review_task(self):
        bp = _make_blueprint()
        summaries = {"main.py": _make_ast_summary()}
        precheck = _make_precheck_result()
        prompt = build_review_prompt(bp, summaries, precheck)

        assert "## Review Task" in prompt
        assert "SEMANTIC fidelity" in prompt

    def test_prompt_contains_five_dimensions(self):
        bp = _make_blueprint()
        summaries = {"main.py": _make_ast_summary()}
        precheck = _make_precheck_result()
        prompt = build_review_prompt(bp, summaries, precheck)

        assert "CONTRACT FIDELITY" in prompt
        assert "DEPENDENCY CORRECTNESS" in prompt
        assert "FAILURE MODE COVERAGE" in prompt
        assert "INTENT COHERENCE" in prompt
        assert "OPEN QUESTIONS" in prompt

    def test_prompt_no_intent_section_if_missing(self):
        bp = _make_blueprint()
        bp = bp.model_copy(update={"intent": None})
        summaries = {"main.py": _make_ast_summary()}
        precheck = _make_precheck_result()
        prompt = build_review_prompt(bp, summaries, precheck)

        assert "### Intent" not in prompt


class TestReverseDirection:
    """Test reverse direction (code is truth, blueprint evaluated)."""

    def test_reverse_uses_different_system_prompt(self):
        bp = _make_blueprint()
        summaries = {"main.py": _make_ast_summary()}
        precheck = _make_precheck_result()
        forward = build_review_prompt(bp, summaries, precheck, review_direction="forward")
        reverse = build_review_prompt(bp, summaries, precheck, review_direction="reverse")

        # Forward should reference "faithfully implements"
        assert "faithfully implements" in forward
        assert "faithfully implements" not in reverse

        # Reverse should reference "DRAFT blueprint" and "code is the source of truth"
        assert "DRAFT blueprint" in reverse
        assert "code is the source of truth" in reverse

    def test_reverse_review_task_text(self):
        bp = _make_blueprint()
        summaries = {"main.py": _make_ast_summary()}
        precheck = _make_precheck_result()
        prompt = build_review_prompt(bp, summaries, precheck, review_direction="reverse")

        assert "BLUEPRINT ACCURACY" in prompt
        assert "over-claim" in prompt

    def test_reverse_still_contains_all_sections(self):
        bp = _make_blueprint()
        summaries = {"main.py": _make_ast_summary()}
        precheck = _make_precheck_result()
        prompt = build_review_prompt(bp, summaries, precheck, review_direction="reverse")

        assert "## Blueprint: test-service" in prompt
        assert "### Intent" in prompt
        assert "### Interfaces" in prompt
        assert "### Dependencies" in prompt
        assert "### Failure Modes" in prompt
        assert "## Mechanical Pre-Check Results" in prompt
        assert "## Source Files Under Review" in prompt

    def test_forward_is_default(self):
        bp = _make_blueprint()
        summaries = {"main.py": _make_ast_summary()}
        precheck = _make_precheck_result()
        default = build_review_prompt(bp, summaries, precheck)
        explicit = build_review_prompt(bp, summaries, precheck, review_direction="forward")

        assert default == explicit


class TestReviewResultSchema:
    """Test that ReviewResult parses correctly from structured data."""

    def test_valid_review_result(self):
        data = {
            "service_id": "test-service",
            "reviewer": "cc",
            "review_direction": "forward",
            "review_timestamp": "2026-02-19T12:00:00Z",
            "overall_fidelity_score": 0.85,
            "discrepancies": [
                {
                    "dimension": "contract",
                    "severity": "minor",
                    "blueprint_claim": "GET /health returns status",
                    "code_evidence": "Returns {'status': 'healthy'} — matches",
                    "recommendation": "None needed",
                    "affected_file": "main.py",
                }
            ],
            "open_question_updates": [
                {
                    "question": "Should we add caching?",
                    "status": "new",
                    "evidence": "No caching logic found in source",
                }
            ],
            "promotion_recommendation": "approve_with_notes",
            "summary_note": "Service largely conforms. Minor contract observation.",
        }
        result = ReviewResult.model_validate(data)
        assert result.service_id == "test-service"
        assert result.overall_fidelity_score == 0.85
        assert len(result.discrepancies) == 1
        assert result.promotion_recommendation == "approve_with_notes"

    def test_review_result_json_roundtrip(self):
        data = {
            "service_id": "test-service",
            "reviewer": "gaia-study",
            "review_direction": "reverse",
            "review_timestamp": "2026-02-19T12:00:00Z",
            "overall_fidelity_score": 0.92,
            "discrepancies": [],
            "open_question_updates": [],
            "promotion_recommendation": "approve",
            "summary_note": "Perfect fidelity.",
        }
        result = ReviewResult.model_validate(data)
        json_str = result.model_dump_json()
        reparsed = ReviewResult.model_validate_json(json_str)
        assert reparsed.overall_fidelity_score == 0.92

    def test_fidelity_score_bounds(self):
        with pytest.raises(Exception):
            ReviewResult.model_validate({
                "service_id": "x",
                "reviewer": "cc",
                "review_direction": "forward",
                "review_timestamp": "2026-02-19T12:00:00Z",
                "overall_fidelity_score": 1.5,  # Out of bounds
                "discrepancies": [],
                "open_question_updates": [],
                "promotion_recommendation": "approve",
                "summary_note": "x",
            })


class TestTokenBudgetGuard:
    """Test progressive truncation under token budget."""

    def test_no_truncation_when_under_budget(self):
        bp = _make_blueprint()
        summaries = {"main.py": _make_ast_summary()}
        precheck = _make_precheck_result()
        prompt = build_review_prompt(bp, summaries, precheck, max_prompt_tokens=100000)

        # Should contain all sections
        assert "### Open Questions" in prompt
        assert "Design decisions:" in prompt

    def test_truncation_drops_open_questions_first(self):
        bp = _make_blueprint()
        summaries = {"main.py": _make_ast_summary()}
        precheck = _make_precheck_result()
        # Very tight budget to trigger truncation
        prompt = build_review_prompt(bp, summaries, precheck, max_prompt_tokens=500)

        # Open questions should be dropped first
        assert "Should we add caching?" not in prompt

    def test_truncation_produces_valid_prompt(self):
        bp = _make_blueprint()
        summaries = {"main.py": _make_ast_summary()}
        precheck = _make_precheck_result()
        prompt = build_review_prompt(bp, summaries, precheck, max_prompt_tokens=500)

        # Core structure should survive
        assert "SYSTEM:" in prompt
        assert "USER:" in prompt
        assert "## Blueprint: test-service" in prompt

    def test_interfaces_never_truncated(self):
        bp = _make_blueprint()
        summaries = {"main.py": _make_ast_summary()}
        precheck = _make_precheck_result()
        # Extremely tight budget
        prompt = build_review_prompt(bp, summaries, precheck, max_prompt_tokens=200)

        # Interfaces and failure modes should survive
        assert "/health" in prompt
        assert "/process" in prompt

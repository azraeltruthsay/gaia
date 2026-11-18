"""
AgentReviewResult — Structured output schema for all specialist agents.

This is a TOOLCHAIN model, not a runtime model. It lives in .claude/agents/shared/
and is never imported by production services. Review agents and creative agents
both produce AgentReviewResult; for creative agents, findings capture template
deviations rather than code issues.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Finding(BaseModel):
    """A single issue, observation, or template deviation found by an agent."""

    category: str = Field(
        description="Domain tag: 'naming', 'security', 'drift', 'template-deviation', 'performance', etc."
    )
    severity: Literal["info", "warning", "error", "critical"]
    file: str = Field(description="File path relative to repo root")
    line: int | None = Field(default=None, description="Line number, if applicable")
    description: str = Field(description="What was found — factual, not prescriptive")
    suggestion: str | None = Field(
        default=None, description="Recommended fix or action"
    )


class AgentReviewResult(BaseModel):
    """Structured output from any specialist agent invocation."""

    agent_id: str = Field(
        description="Identifies the specialist: 'codemind', 'sentinel', 'ux-designer', etc."
    )
    schema_version: str = Field(
        default="1.0", description="For forward compatibility"
    )
    target: str = Field(
        description="File, service, or page under review or construction"
    )
    timestamp: datetime = Field(
        description="ISO 8601 — audit trail and corpus ordering"
    )
    verdict: Literal["approve", "approve_with_notes", "reject"] = Field(
        description="approve: no issues. approve_with_notes: non-blocking findings. reject: blocks promotion."
    )
    findings: list[Finding] = Field(
        default_factory=list,
        description="Typed findings — code issues for review agents, template deviations for creative agents",
    )
    metrics: dict = Field(
        default_factory=dict,
        description="Agent-specific: divergence_score, vuln_count, template_deviation_count, etc.",
    )
    summary: str = Field(
        description="Written last, derived from findings. Always consistent with structured data. Human-readable."
    )

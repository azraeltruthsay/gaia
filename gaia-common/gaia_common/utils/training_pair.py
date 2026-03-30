"""
gaia_common/utils/training_pair.py

Training pair schema for the code-architect QLoRA curriculum.

Each training pair captures a review cycle: blueprint (input context),
source code (desired output), and quality signal (from CC review).

Pairs come from four sources:
  - forward:     CC generates code from blueprint, then reviews it
  - retroactive: CC reviews existing live code against its blueprint
  - reverse:     CC reviews a draft blueprint against existing code
  - journal:     Extracted from dev journal entries
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ModificationRecord(BaseModel):
    """A single change made to code between review and promotion."""
    file: str
    change_type: Literal["added", "removed", "modified"]
    description: str


class TrainingPair(BaseModel):
    """A single training example for the code-architect QLoRA adapter."""

    pair_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pair_type: Literal["forward", "retroactive", "reverse", "journal"]
    granularity: Literal["service", "file"]
    service_id: str
    file_scope: Optional[str] = None  # non-null only for file-level pairs
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Input context
    blueprint_yaml: str
    blueprint_scoped: Optional[str] = None  # subset for file-level, full for service-level
    ast_summaries: Dict[str, dict]  # filename -> raw AST summary dict
    reference_services: List[str] = Field(default_factory=list)  # IDs of reference exemplars

    # Review quality signal
    cc_review: Optional[dict] = None  # raw ReviewResult dict (null if review not yet performed)
    promotion_outcome: Literal["passed", "failed", "modified", "pending"] = "pending"
    modifications_before_promotion: List[ModificationRecord] = Field(default_factory=list)

    # Computed scores
    divergence_score_final: Optional[float] = None
    ground_truth_fidelity: Optional[float] = None
    total_checkpoints: Optional[int] = None  # for future normalization

    @staticmethod
    def compute_fidelity(cc_review: dict) -> float:
        """
        Compute ground_truth_fidelity from a ReviewResult dict.

        Formula: 1.0 - (critical * 0.3 + major * 0.15 + minor * 0.05), clamped [0, 1].
        """
        discrepancies = cc_review.get("discrepancies", [])
        critical = sum(1 for d in discrepancies if d.get("severity") == "critical")
        major = sum(1 for d in discrepancies if d.get("severity") == "major")
        minor = sum(1 for d in discrepancies if d.get("severity") == "minor")

        score = 1.0 - (critical * 0.3 + major * 0.15 + minor * 0.05)
        return max(0.0, min(1.0, score))


class CorpusMetadata(BaseModel):
    """Metadata for the assembled training corpus."""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_pairs: int = 0
    train_count: int = 0
    validation_count: int = 0
    pair_type_counts: Dict[str, int] = Field(default_factory=dict)
    service_counts: Dict[str, int] = Field(default_factory=dict)
    mean_fidelity: Optional[float] = None
    min_corpus_size: int = 50
    corpus_ready: bool = False  # True if total_pairs >= min_corpus_size

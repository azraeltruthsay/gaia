"""
gaia_common/utils/promotion_request.py

Promotion request lifecycle management — the data layer for GAIA's
two-gate promotion approval flow.

Flow:
  1. GAIA creates a request (status=pending) after readiness assessment
  2. Human approves (status=approved)  — Gate 1
  3. Dry-run executes (status=dry_run_passed or dry_run_failed)
  4. Human confirms (status=confirmed) — Gate 2
  5. Live promotion executes (status=promoted or failed)

Storage: /knowledge/promotion_requests/{service_id}_{timestamp}.json
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional

logger = logging.getLogger("GAIA.PromotionRequest")

_REQUESTS_DIR = Path("/gaia/GAIA_Project/knowledge/promotion_requests")

# Valid status transitions
_VALID_TRANSITIONS: Dict[str, List[str]] = {
    "pending": ["approved", "rejected", "expired"],
    "approved": ["dry_run_passed", "dry_run_failed", "rejected"],
    "dry_run_passed": ["confirmed", "rejected"],
    "dry_run_failed": ["rejected", "pending"],  # Can retry
    "confirmed": ["promoted", "failed"],
    "promoted": [],  # Terminal
    "rejected": [],  # Terminal
    "expired": [],   # Terminal
    "failed": ["pending"],  # Can retry
}

RequestStatus = Literal[
    "pending", "approved", "dry_run_passed", "dry_run_failed",
    "confirmed", "promoted", "rejected", "expired", "failed",
]


@dataclass
class PromotionRequest:
    """A promotion request with two-gate approval lifecycle."""
    request_id: str
    service_id: str
    requested_at: str  # ISO format
    verdict: str       # From readiness report
    recommendation: str
    pipeline_cmd: str
    check_summary: str  # Human-readable summary of checks
    status: RequestStatus = "pending"
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    confirmed_at: Optional[str] = None
    promoted_at: Optional[str] = None
    dry_run_output: Optional[str] = None
    rejection_reason: Optional[str] = None
    history: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PromotionRequest:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def _record(self, action: str, detail: str = ""):
        self.history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "detail": detail,
        })


# ── Public API ───────────────────────────────────────────────────────────────


def create_promotion_request(
    service_id: str,
    verdict: str,
    recommendation: str,
    pipeline_cmd: str,
    check_summary: str,
) -> PromotionRequest:
    """Create a new promotion request from a readiness assessment."""
    now = datetime.now(timezone.utc)
    request_id = f"{service_id}_{now.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"

    req = PromotionRequest(
        request_id=request_id,
        service_id=service_id,
        requested_at=now.isoformat(),
        verdict=verdict,
        recommendation=recommendation,
        pipeline_cmd=pipeline_cmd,
        check_summary=check_summary,
    )
    req._record("created", f"Readiness verdict: {verdict}")

    _save_request(req)
    logger.info("Promotion request created: %s (verdict=%s)", request_id, verdict)
    return req


def load_request(request_id: str) -> Optional[PromotionRequest]:
    """Load a promotion request by ID."""
    path = _REQUESTS_DIR / f"{request_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return PromotionRequest.from_dict(data)
    except Exception as exc:
        logger.warning("Could not load request %s: %s", request_id, exc)
        return None


def load_pending_request(service_id: str) -> Optional[PromotionRequest]:
    """Load the most recent pending/approved request for a service."""
    if not _REQUESTS_DIR.exists():
        return None

    active_statuses = {"pending", "approved", "dry_run_passed", "confirmed"}
    best: Optional[PromotionRequest] = None

    for path in sorted(_REQUESTS_DIR.glob(f"{service_id}_*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            req = PromotionRequest.from_dict(data)
            if req.status in active_statuses:
                best = req
                break
        except Exception:
            continue

    return best


def list_requests(
    service_id: str | None = None,
    status_filter: str | None = None,
) -> List[PromotionRequest]:
    """List all promotion requests, optionally filtered."""
    if not _REQUESTS_DIR.exists():
        return []

    results = []
    pattern = f"{service_id}_*.json" if service_id else "*.json"
    for path in sorted(_REQUESTS_DIR.glob(pattern), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            req = PromotionRequest.from_dict(data)
            if status_filter and req.status != status_filter:
                continue
            results.append(req)
        except Exception:
            continue

    return results


def approve_request(request_id: str, approved_by: str = "human") -> PromotionRequest:
    """Gate 1: Approve a promotion request."""
    req = _load_and_validate(request_id, expected_status="pending")
    req.status = "approved"
    req.approved_by = approved_by
    req.approved_at = datetime.now(timezone.utc).isoformat()
    req._record("approved", f"Approved by {approved_by}")
    _save_request(req)
    logger.info("Request %s approved by %s", request_id, approved_by)
    return req


def record_dry_run(request_id: str, passed: bool, output: str = "") -> PromotionRequest:
    """Record the result of a dry-run execution."""
    req = _load_and_validate(request_id, expected_status="approved")
    req.status = "dry_run_passed" if passed else "dry_run_failed"
    req.dry_run_output = output[:5000]  # Cap stored output
    req._record("dry_run", f"{'passed' if passed else 'failed'}: {output[:200]}")
    _save_request(req)
    logger.info("Request %s dry-run %s", request_id, "passed" if passed else "failed")
    return req


def confirm_request(request_id: str) -> PromotionRequest:
    """Gate 2: Confirm promotion after successful dry-run."""
    req = _load_and_validate(request_id, expected_status="dry_run_passed")
    req.status = "confirmed"
    req.confirmed_at = datetime.now(timezone.utc).isoformat()
    req._record("confirmed", "Human confirmed after dry-run")
    _save_request(req)
    logger.info("Request %s confirmed for live promotion", request_id)
    return req


def record_promotion(request_id: str, success: bool, detail: str = "") -> PromotionRequest:
    """Record the result of the live promotion."""
    req = _load_and_validate(request_id, expected_status="confirmed")
    req.status = "promoted" if success else "failed"
    if success:
        req.promoted_at = datetime.now(timezone.utc).isoformat()
    req._record("promotion", f"{'success' if success else 'failed'}: {detail[:200]}")
    _save_request(req)
    logger.info("Request %s promotion %s", request_id, "succeeded" if success else "failed")
    return req


def reject_request(request_id: str, reason: str = "") -> PromotionRequest:
    """Reject a promotion request at any active stage."""
    req = load_request(request_id)
    if req is None:
        raise FileNotFoundError(f"Request not found: {request_id}")
    if req.status in ("promoted", "rejected", "expired"):
        raise ValueError(f"Cannot reject request in terminal state: {req.status}")
    req.status = "rejected"
    req.rejection_reason = reason
    req._record("rejected", reason)
    _save_request(req)
    logger.info("Request %s rejected: %s", request_id, reason)
    return req


# ── Helpers ──────────────────────────────────────────────────────────────────


def _load_and_validate(request_id: str, expected_status: str) -> PromotionRequest:
    """Load a request and verify it's in the expected state."""
    req = load_request(request_id)
    if req is None:
        raise FileNotFoundError(f"Request not found: {request_id}")
    if req.status != expected_status:
        valid = _VALID_TRANSITIONS.get(req.status, [])
        raise ValueError(
            f"Request {request_id} is in state '{req.status}', "
            f"expected '{expected_status}'. Valid transitions: {valid}"
        )
    return req


def _save_request(req: PromotionRequest) -> Path:
    """Persist a request to disk."""
    _REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _REQUESTS_DIR / f"{req.request_id}.json"
    path.write_text(json.dumps(req.to_dict(), indent=2), encoding="utf-8")
    return path

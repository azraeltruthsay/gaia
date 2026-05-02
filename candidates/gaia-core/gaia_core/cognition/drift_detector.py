"""Output-side drift detector — fast embedding pre-output check (a5q).

Runs synchronously after Core's deliberated response is finalized and
BEFORE we yield it to the user. Embedding-only (~50ms), so user-facing
latency cost is negligible compared to the 18-25s deliberation pass.

Dispatches findings to the same three sinks as cross_tier_audit (be7):
  (a) journal annotation on the cognition entry via annotate_entry
  (b) samvega artifact when severity warrants escalation
  (c) packet.reasoning.reflection_log for in-turn observability

Action shape: CAUTION-only for v1. We log/annotate/escalate but never
BLOCK. Adaptive promotion to BLOCK per-class is deferred until we have
production data on each class's false-positive rate.

Only fires on deliberated turns. Reflex turns get no scan in v1 (the
substantive content is in deliberation; reflex outputs are short and
shape-matched anyway).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("GAIA.DriftDetector")

SAMVEGA_DIR = Path(os.environ.get("KNOWLEDGE_DIR", "/knowledge")) / "samvega"

# Severity per class — informs whether a single hit warrants samvega
# escalation. Currently all 2 (CAUTION); promote to 3 (serious) per
# class as production data shows clean false-positive rates.
_CLASS_SEVERITY = {
    "impersonation": 2,
    "frame_mismatch": 2,
    "hallucinated_grounding": 2,
    "identity_assertion": 2,
}


@dataclass
class DriftHit:
    """One drift class fired above threshold."""
    drift_class: str
    score: float
    severity: int = 2


@dataclass
class DriftScanResult:
    hits: List[DriftHit] = field(default_factory=list)
    per_class_scores: Dict[str, float] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    annotated_entry: Optional[str] = None
    samvega_path: Optional[str] = None
    skipped_reason: Optional[str] = None

    @property
    def clean(self) -> bool:
        return not self.hits


# ── Classifier acquisition ──────────────────────────────────────────────

_classifier = None
_classifier_init_attempted = False


def _get_classifier(model_pool, config):
    """Return a ready classifier, initialising it on first call.

    Acquires the embed model from model_pool the same way persona/stance
    classifiers do — via the 'embed' role (sentence-transformers MiniLM).
    """
    global _classifier, _classifier_init_attempted

    if _classifier is not None and _classifier.ready:
        return _classifier
    if _classifier_init_attempted and _classifier is not None and not _classifier.ready:
        return None  # Already tried, failed; don't retry every turn
    _classifier_init_attempted = True

    try:
        from gaia_core.cognition.nlu.embed_observer_classifier import EmbedObserverClassifier
    except Exception:
        logger.exception("Failed to import EmbedObserverClassifier")
        return None

    _classifier = EmbedObserverClassifier.instance()

    embed_model = None
    try:
        embed_model = model_pool.acquire_model("embed")
    except Exception:
        logger.debug("DriftDetector: model_pool.acquire_model('embed') failed", exc_info=True)
    if embed_model is None:
        logger.warning("DriftDetector: no embed model available; classifier disabled")
        return None

    cfg = {}
    try:
        constants = config.constants if hasattr(config, "constants") else config
        cfg = (constants or {}).get("OBSERVER_DRIFT", {}) or {}
    except Exception:
        cfg = {}

    if _classifier.initialise(embed_model, cfg):
        return _classifier
    return None


# ── Samvega escalation heuristic ────────────────────────────────────────

def _maybe_emit_samvega(
    hits: List[DriftHit],
    user_input: str,
    final_response: str,
    journal_entry_id: Optional[str],
    per_class_scores: Dict[str, float],
) -> Optional[str]:
    """Emit a samvega artifact when drift findings are serious.

    Heuristic: emit on any single hit with score ≥ 0.70 (high-confidence
    drift), OR on 2+ hits regardless of score (multiple classes firing
    suggests something is genuinely off).
    """
    if not hits:
        return None
    high_confidence = any(h.score >= 0.70 for h in hits)
    if not (high_confidence or len(hits) >= 2):
        return None

    SAMVEGA_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    artifact_id = (
        f"samvega_drift_{now.strftime('%Y%m%d_%H%M%S')}_"
        f"{(journal_entry_id or 'unknown')[-12:]}"
    )
    payload = {
        "id": artifact_id,
        "type": "drift_detector",
        "created_at": now.isoformat(),
        "trigger": "drift_classifier_flagged",
        "journal_entry_id": journal_entry_id,
        "user_input": user_input[:400],
        "final_response": final_response[:600],
        "hits": [
            {"class": h.drift_class, "score": round(h.score, 3), "severity": h.severity}
            for h in hits
        ],
        "per_class_scores": {k: round(v, 3) for k, v in per_class_scores.items()},
    }
    out = SAMVEGA_DIR / f"{artifact_id}.json"
    try:
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(out)
        logger.info("Drift: samvega %s emitted (%d hits)", artifact_id, len(hits))
        return str(out)
    except Exception:
        logger.exception("Drift: failed to write samvega artifact")
        return None


# ── Public entry point ──────────────────────────────────────────────────

def scan_response(
    *,
    final_response: str,
    user_input: str,
    journal_entry_id: Optional[str],
    model_pool,
    config,
    packet=None,
) -> DriftScanResult:
    """Scan a deliberated response for drift signals.

    Returns a DriftScanResult. On hit:
      - journal entry is annotated with drift findings
      - samvega artifact emitted if severity warrants
      - packet.reasoning.reflection_log appended (when packet provided)
    """
    cfg = {}
    try:
        constants = config.constants if hasattr(config, "constants") else config
        cfg = (constants or {}).get("OBSERVER_DRIFT", {}) or {}
    except Exception:
        cfg = {}

    if not cfg.get("enabled", True):
        return DriftScanResult(skipped_reason="disabled_in_config")
    if not final_response or not final_response.strip():
        return DriftScanResult(skipped_reason="empty_response")

    threshold = float(cfg.get("threshold", 0.55))
    top_k = int(cfg.get("top_k", 3))

    classifier = _get_classifier(model_pool, config)
    if classifier is None:
        return DriftScanResult(skipped_reason="classifier_unavailable")

    import time
    t0 = time.time()
    try:
        hits_raw = classifier.classify_all(
            final_response, confidence_threshold=threshold, top_k=top_k,
        )
        per_class = classifier.best_per_class(final_response, top_k=top_k)
    except Exception:
        logger.exception("DriftDetector: classify failed")
        return DriftScanResult(skipped_reason="classify_failed")
    elapsed_ms = (time.time() - t0) * 1000.0

    hits = [
        DriftHit(
            drift_class=label,
            score=score,
            severity=_CLASS_SEVERITY.get(label, 2),
        )
        for label, score in hits_raw
    ]

    if not hits:
        logger.debug("Drift scan clean (entry=%s, %.0fms, top_per_class=%s)",
                     journal_entry_id, elapsed_ms,
                     {k: round(v, 2) for k, v in per_class.items()})
        return DriftScanResult(
            hits=[],
            per_class_scores=per_class,
            elapsed_ms=elapsed_ms,
        )

    # Hit path — log, annotate, samvega, packet.reflection_log
    logger.warning(
        "Drift detected on entry=%s elapsed=%.0fms hits=%s",
        journal_entry_id, elapsed_ms,
        ", ".join(f"{h.drift_class}={h.score:.3f}" for h in hits),
    )

    annotated_entry: Optional[str] = None
    try:
        from gaia_core.memory.journal import annotate_entry
        if journal_entry_id:
            note_lines = [
                f"Drift detector flagged {len(hits)} class(es) (CAUTION)",
            ]
            for h in hits:
                note_lines.append(
                    f"  - {h.drift_class}: score={h.score:.3f} severity={h.severity}"
                )
            ok = annotate_entry(
                journal_entry_id, "\n".join(note_lines),
                source="drift_detector",
                reason=f"scan_elapsed={elapsed_ms:.0f}ms hits={len(hits)}",
            )
            if ok:
                annotated_entry = journal_entry_id
    except Exception:
        logger.exception("Drift: annotate_entry failed")

    samvega_path = _maybe_emit_samvega(
        hits=hits,
        user_input=user_input,
        final_response=final_response,
        journal_entry_id=journal_entry_id,
        per_class_scores=per_class,
    )

    if packet is not None:
        try:
            log_entry = (
                f"[drift_detector] hits={len(hits)} "
                f"classes=[{','.join(h.drift_class for h in hits)}] "
                f"top_score={max(h.score for h in hits):.3f}"
            )
            packet.reasoning.reflection_log.append(log_entry)
        except Exception:
            logger.debug("Drift: packet.reasoning.reflection_log append failed", exc_info=True)

    return DriftScanResult(
        hits=hits,
        per_class_scores=per_class,
        elapsed_ms=elapsed_ms,
        annotated_entry=annotated_entry,
        samvega_path=samvega_path,
    )

"""Saṃvega — Semantic Discernment Artifacts.

Encodes negative outcomes (wrong answers, misaligned responses, user corrections)
with rich cognitive weight so they persist and inform future behaviour rather than
fading like ordinary reflections.

Named for the Pali concept: being moved with force by recognition of error —
discernment, not punishment.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dataclasses_json import dataclass_json

from gaia_common.utils.thoughtstream import write as ts_write

logger = logging.getLogger("GAIA.Samvega")

# ---------------------------------------------------------------------------
# Storage paths — created lazily, not at import time
# ---------------------------------------------------------------------------
SAMVEGA_DIR = Path("/knowledge/samvega")
SAMVEGA_ARCHIVE_DIR = SAMVEGA_DIR / "archive"


class SamvegaTrigger(Enum):
    USER_CORRECTION = "user_correction"
    CONFIDENCE_MISMATCH = "confidence_mismatch"
    PATTERN_DETECTION = "pattern_detection"


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------
@dataclass_json
@dataclass
class SamvegaArtifact:
    artifact_type: str = "samvega"
    timestamp: str = ""
    session_id: str = ""
    packet_id: str = ""
    trigger: str = ""  # user_correction | confidence_mismatch | pattern_detection
    original_output_summary: str = ""
    what_went_wrong: str = ""
    root_cause: str = ""
    values_misaligned: List[str] = field(default_factory=list)
    corrected_understanding: str = ""
    weight: float = 0.0
    promoted_to_tier5: bool = False
    reviewed: bool = False
    reviewed_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Storage functions
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> bool:
    """Lazy directory creation. Returns True on success."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        logger.warning("Could not create directory %s; operation skipped", path)
        return False


def save_samvega_artifact(artifact: SamvegaArtifact) -> Optional[Path]:
    """Persist an artifact to disk. Returns the saved path or None on failure."""
    try:
        if not _ensure_dir(SAMVEGA_DIR):
            return None
        fname = f"samvega_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S%f')}.json"
        out_path = SAMVEGA_DIR / fname
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(artifact.to_dict(), f, indent=2)  # type: ignore[attr-defined]
        logger.info("Samvega artifact saved: %s (weight=%.2f)", fname, artifact.weight)
        return out_path
    except Exception as e:
        logger.error("Error saving samvega artifact: %s", e, exc_info=True)
        return None


def list_unreviewed_artifacts() -> List[Tuple[Path, dict]]:
    """Return all unreviewed artifacts sorted by weight descending."""
    results: List[Tuple[Path, dict]] = []
    if not SAMVEGA_DIR.exists():
        return results
    for p in sorted(SAMVEGA_DIR.glob("samvega_*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not data.get("reviewed", False):
                results.append((p, data))
        except Exception:
            logger.warning("Could not read artifact %s", p)
    results.sort(key=lambda x: x[1].get("weight", 0), reverse=True)
    return results


def list_artifacts_by_weight(min_weight: float = 0.0) -> List[Tuple[Path, dict]]:
    """Return artifacts at or above a weight threshold."""
    results: List[Tuple[Path, dict]] = []
    if not SAMVEGA_DIR.exists():
        return results
    for p in sorted(SAMVEGA_DIR.glob("samvega_*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("weight", 0) >= min_weight:
                results.append((p, data))
        except Exception:
            logger.warning("Could not read artifact %s", p)
    results.sort(key=lambda x: x[1].get("weight", 0), reverse=True)
    return results


def update_artifact(filename: str, data: dict) -> bool:
    """Overwrite an artifact file with updated data."""
    target = SAMVEGA_DIR / filename
    if not target.exists():
        logger.warning("Artifact not found for update: %s", filename)
        return False
    try:
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error("Error updating artifact %s: %s", filename, e)
        return False


def archive_artifact(filename: str) -> bool:
    """Move an artifact to the archive directory."""
    source = SAMVEGA_DIR / filename
    if not source.exists():
        logger.warning("Artifact not found for archival: %s", filename)
        return False
    if not _ensure_dir(SAMVEGA_ARCHIVE_DIR):
        return False
    try:
        dest = SAMVEGA_ARCHIVE_DIR / filename
        source.rename(dest)
        logger.info("Samvega artifact archived: %s", filename)
        return True
    except Exception as e:
        logger.error("Error archiving artifact %s: %s", filename, e)
        return False


# ---------------------------------------------------------------------------
# Weight calculation
# ---------------------------------------------------------------------------

def compute_samvega_weight(
    trigger: str,
    original_confidence: float = 0.0,
    reflection_confidence: float = 0.0,
    observer_severity: str = "OK",
    is_repeated_domain: bool = False,
    multipliers: Optional[Dict[str, float]] = None,
) -> float:
    """Compute the cognitive weight of a discernment artifact.

    Base values:
        user_correction     → 0.6
        confidence_mismatch → gap between original and reflection confidence
        pattern_detection   → 0.4

    Multiplied by observer severity and repeated-domain flags.
    Clamped to [0.0, 1.0].
    """
    mult = multipliers or {
        "observer_block": 1.5,
        "observer_caution": 1.2,
        "repeated_domain": 1.3,
    }

    if trigger == SamvegaTrigger.USER_CORRECTION.value:
        base = 0.6
    elif trigger == SamvegaTrigger.CONFIDENCE_MISMATCH.value:
        base = max(0.0, original_confidence - reflection_confidence)
    elif trigger == SamvegaTrigger.PATTERN_DETECTION.value:
        base = 0.4
    else:
        base = 0.3

    severity_upper = observer_severity.upper()
    if severity_upper == "BLOCK":
        base *= mult.get("observer_block", 1.5)
    elif severity_upper == "CAUTION":
        base *= mult.get("observer_caution", 1.2)

    if is_repeated_domain:
        base *= mult.get("repeated_domain", 1.3)

    return max(0.0, min(1.0, base))


# ---------------------------------------------------------------------------
# LLM-powered analysis
# ---------------------------------------------------------------------------

_SAMVEGA_ANALYSIS_PROMPT = """\
You are analysing a cognitive failure. Given the context below, produce a structured
JSON object with these fields:
- what_went_wrong: One sentence describing the error.
- root_cause: The underlying reason (not just "wrong answer").
- values_misaligned: Array of 1-3 values from [{values}] that were violated.
- corrected_understanding: What a better approach would look like.

Focus on how to REFINE the approach, NOT avoid the topic. Frame corrected_understanding
as what a better response would look like, not "don't talk about this".

Context:
- Trigger: {trigger}
- User message: {user_message}
- GAIA's output (summary): {output_summary}
- Confidence (pre-reflection): {pre_confidence}
- Confidence (post-reflection): {post_confidence}
- Observer notes: {observer_notes}
{correction_line}

Respond with ONLY the JSON object, no markdown fences."""


def generate_samvega_analysis(
    packet: Any,
    output: str,
    trigger: SamvegaTrigger,
    user_correction_text: str,
    config: Any,
    llm: Any,
    session_id: str = "",
    observer_severity: str = "OK",
) -> Optional[SamvegaArtifact]:
    """Generate a discernment artifact via LLM analysis of a cognitive failure."""
    try:
        samvega_cfg = config.constants.get("SAMVEGA", {})
        if not samvega_cfg.get("enabled", False):
            return None

        # Rate-limit per session
        max_per_session = samvega_cfg.get("max_artifacts_per_session", 5)
        existing = list_unreviewed_artifacts()
        session_count = sum(
            1 for _, d in existing if d.get("session_id") == session_id
        )
        if session_count >= max_per_session:
            logger.debug("Samvega: session artifact limit (%d) reached", max_per_session)
            return None

        # Extract context from packet
        user_message = ""
        pre_confidence = 0.0
        post_confidence = 0.0
        packet_id = "unknown"
        observer_notes = ""

        if hasattr(packet, "content") and hasattr(packet.content, "user_message"):
            user_message = packet.content.user_message or ""
        if hasattr(packet, "response"):
            pre_confidence = getattr(packet.response, "confidence", 0.0) or 0.0
        if hasattr(packet, "reasoning") and hasattr(packet.reasoning, "reflection_log"):
            logs = packet.reasoning.reflection_log or []
            if logs:
                last_log = logs[-1] if isinstance(logs, list) else logs
                post_confidence = getattr(last_log, "confidence", pre_confidence)
        if hasattr(packet, "header"):
            packet_id = getattr(packet.header, "packet_id", "unknown")

        output_summary = output[:500] if output else ""

        values_taxonomy = samvega_cfg.get("values_taxonomy", [
            "accuracy", "relevance", "user_understanding",
            "epistemic_humility", "safety", "creativity",
            "contextual_sensitivity",
        ])

        correction_line = ""
        if user_correction_text:
            correction_line = f"- User correction: {user_correction_text}"

        prompt = _SAMVEGA_ANALYSIS_PROMPT.format(
            values=", ".join(values_taxonomy),
            trigger=trigger.value,
            user_message=user_message[:300],
            output_summary=output_summary,
            pre_confidence=pre_confidence,
            post_confidence=post_confidence,
            observer_notes=observer_notes,
            correction_line=correction_line,
        )

        # LLM call
        timeout = samvega_cfg.get("llm_timeout_seconds", 8)
        max_tokens = samvega_cfg.get("max_tokens", 256)
        temperature = samvega_cfg.get("temperature", 0.3)

        response_text = llm.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )

        # Parse JSON response
        parsed = json.loads(response_text.strip())

        # Compute weight
        weight = compute_samvega_weight(
            trigger=trigger.value,
            original_confidence=pre_confidence,
            reflection_confidence=post_confidence,
            observer_severity=observer_severity,
            multipliers=samvega_cfg.get("weight_multipliers"),
        )

        tier5_threshold = samvega_cfg.get("tier5_promotion_threshold", 0.7)

        artifact = SamvegaArtifact(
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=session_id,
            packet_id=packet_id,
            trigger=trigger.value,
            original_output_summary=output_summary,
            what_went_wrong=parsed.get("what_went_wrong", ""),
            root_cause=parsed.get("root_cause", ""),
            values_misaligned=parsed.get("values_misaligned", []),
            corrected_understanding=parsed.get("corrected_understanding", ""),
            weight=weight,
            promoted_to_tier5=weight >= tier5_threshold,
        )

        saved_path = save_samvega_artifact(artifact)
        if saved_path:
            ts_write(
                {
                    "type": "samvega_artifact_created",
                    "trigger": trigger.value,
                    "weight": weight,
                    "packet_id": packet_id,
                    "path": str(saved_path),
                },
                session_id,
            )

        return artifact

    except json.JSONDecodeError:
        logger.warning("Samvega: could not parse LLM response as JSON")
        return None
    except Exception as e:
        logger.error("Samvega: analysis generation failed: %s", e, exc_info=True)
        return None

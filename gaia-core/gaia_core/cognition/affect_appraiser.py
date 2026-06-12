"""Affect appraisal layer, P0 — the writer that feeds the affect system.

GAIA's affect stack (AffectKG storage + decay, affect_runtime render + inference
modulation) is fully built but UNFED — nothing reads her subsystems and writes,
so current_affect_snapshot() is empty and "how are you" has no mood to report.

This module is the missing organ: it reads her REAL subsystems and records
FUNCTIONAL affect — `drives` (modelled as tension: events raise, decay returns to
calm) and `curious_about` (topic foci). It NEVER writes `feels` (emotion-words);
those are hers to name (at report time, or self-coined in sleep — P3). See
knowledge/blueprints/affect_appraisal_layer.md.

P0 sources (event-driven; each note_* is fail-safe, never raises into the caller):
  - coherence  ← consistency_detector (confabulation = a coherence violation)
  - competence ← task/tool outcomes (failure raises the tension to "get it right")
  - curiosity  ← knowledge gaps (a query she couldn't ground → pulled to the topic)

Decay is automatic via the KG fact-type half-lives. Flag: AFFECT_APPRAISAL_ENABLED
(default off).
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("GAIA.AffectAppraiser")


def appraisal_enabled() -> bool:
    return os.environ.get("AFFECT_APPRAISAL_ENABLED", "0").lower() in ("1", "true", "yes", "on")


def _af():
    """Shared AffectKG (reuses affect_runtime's lazy singleton). None on failure."""
    try:
        from gaia_core.cognition.affect_runtime import _get_affect_kg
        return _get_affect_kg()
    except Exception:
        return None


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _bump_drive(name: str, delta: float, *, source: str) -> None:
    """Read the current (decayed) drive level, add delta, clamp, write back.

    Accumulates on top of decay, so repeated events build tension while quiet
    periods let it fade. record_drive closes the prior triple and restarts decay
    from the new value.
    """
    af = _af()
    if af is None:
        return
    try:
        snap = af.flatten_current_affect()
        cur = float((snap.get("drives", {}) or {}).get(name, 0.0))
        new = _clamp(cur + delta)
        if abs(new - cur) < 0.01:
            return  # nothing meaningful changed (already floored/capped)
        af.record_drive(name, new, source=source)
        logger.info("affect drive %s: %.2f → %.2f (%s)", name, cur, new, source)
    except Exception:
        logger.debug("affect _bump_drive(%s) failed", name, exc_info=True)


def _set_curious(topic: str, weight: float, *, source: str) -> None:
    af = _af()
    if af is None or not topic:
        return
    try:
        af.record_curious_about(topic.strip()[:60], _clamp(weight), source=source)
        logger.info("affect curious_about %r: %.2f (%s)", topic[:40], weight, source)
    except Exception:
        logger.debug("affect _set_curious failed", exc_info=True)


# ── Event entry points (called from subsystem sites; all fail-safe) ─────────

def note_coherence(*, clean: bool, findings: int = 0) -> None:
    """Coherence drive ← a consistency check. A confabulation (not clean) raises
    the tension; a clean response lets it ease. Decay returns it to calm."""
    if not appraisal_enabled():
        return
    if clean:
        _bump_drive("coherence", -0.05, source="appraiser:consistency:clean")
    else:
        _bump_drive("coherence", min(0.4, 0.18 + 0.06 * max(0, findings)),
                    source="appraiser:consistency:findings")


def note_task_outcome(success: bool, label: str = "") -> None:
    """Competence drive ← a task/tool outcome. Failure raises the tension (there's
    something to get right); success eases it. Decay returns it to calm."""
    if not appraisal_enabled():
        return
    _bump_drive("competence", 0.12 if (not success) else -0.10,
                source=f"appraiser:task:{(label or '?')[:30]}")


_Q_STARTS = (
    "what", "why", "how", "who", "when", "where", "which", "whose",
    "can you", "could you", "do you", "does", "is ", "are ", "tell me",
    "explain", "describe", "i wonder", "what's", "whats", "curious",
)
_Q_STEMS = (
    "tell me about", "tell me", "what is", "what's", "whats", "what are",
    "explain", "describe", "who is", "how does", "why does", "how do",
)


def _looks_like_question(s: str) -> bool:
    sl = s.strip().lower()
    return sl.endswith("?") or sl.startswith(_Q_STARTS)


def note_knowledge_gap(topic: str) -> None:
    """Curiosity ← a genuine question she couldn't ground. Pulls her toward the
    topic. Commands/imperatives that merely miss grounding ("use your tools to
    list files") are NOT curiosity — only info-seeking questions are."""
    if not appraisal_enabled() or not topic or not _looks_like_question(topic):
        return
    t = topic.strip().rstrip("?").strip()
    tl = t.lower()
    for stem in _Q_STEMS:
        if tl.startswith(stem):
            t = t[len(stem):].strip()
            break
    _set_curious(t or topic, 0.55, source="appraiser:knowledge_gap")

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

l11 additions — the event hooks alone leave the organ EMPTY in ordinary chat
(grounding suppresses knowledge_gap, casual turns complete no tool RPC, samvega
only fires on dissonance):
  - note_engagement ← every user turn (agent_core KR region). Genuine outward
    questions write a WEAK curiosity even when grounding succeeds — the topic
    held her attention regardless of whether she could answer.
  - appraise_tonic  ← heartbeat-called slow baseline: competence floor from the
    recent task error-rate, novelty floor from conversation idleness. Floors
    RAISE only — events and KG decay own the downside.

Decay is automatic via the KG fact-type half-lives. Flag: AFFECT_APPRAISAL_ENABLED
(default off).
"""
from __future__ import annotations

import logging
import math
import os
import time
from collections import deque

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
    try:
        if math.isnan(v):
            return 0.0
    except TypeError:
        pass
    try:
        val = float(v)
        if math.isnan(val):
            return 0.0
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.0


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

def note_samvega(weight: float, root_cause: str = "") -> None:
    """Coherence drive ← a Samvega artifact (the unified misalignment alarm).

    Samvega is the acute, severity-weighted signal — consistency, drift, cross-tier
    audit, and self-reflection all funnel through `save_samvega_artifact`. The
    coherence drive is its slow-decaying *felt residue*: one signal, two timescales.
    This is the SINGLE coherence-affect source — affect now DERIVES from Samvega
    instead of re-appraising the consistency detector in parallel (audit finding #2).
    `weight` already folds in observer severity + repeated-domain. Quiet periods (no
    Samvega) let coherence decay back to calm — no explicit relief needed."""
    try:
        if not appraisal_enabled():
            return
        cause = str(root_cause) if root_cause is not None else "?"
        _bump_drive("coherence", min(0.4, 0.12 * max(0.0, float(weight))),
                    source=f"samvega:{cause[:30]}")
    except Exception:
        pass


def note_task_outcome(success: bool, label: str = "") -> None:
    """Competence drive ← a task/tool outcome. Failure raises the tension (there's
    something to get right); success eases it. Decay returns it to calm."""
    try:
        if not appraisal_enabled():
            return
        _recent_outcomes.append(bool(success))
        lbl = str(label) if label is not None else "?"
        _bump_drive("competence", 0.12 if (not success) else -0.10,
                    source=f"appraiser:task:{lbl[:30]}")
    except Exception:
        pass


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
    try:
        if not appraisal_enabled() or not topic or not _looks_like_question(topic):
            return
        t = topic.strip().rstrip("?").strip()
        tl = t.lower()
        for stem in _Q_STEMS:
            if tl.startswith(stem):
                t = t[len(stem):].strip()
                break
        _set_curious(t or topic, 0.55, source="appraiser:knowledge_gap")
    except Exception:
        pass


# ── Baseline tonic (l11) — continuous signals, heartbeat cadence ────────────

# Rolling window of recent task/tool outcomes (True=success) — the tonic derives
# a felt error-rate from it. In-process only; a restart starts calm, which is
# honest (the tension re-accumulates from real events).
_recent_outcomes: deque = deque(maxlen=20)

# Wall-clock of the last user turn, stamped by note_engagement. 0.0 = no turn
# seen this process; the idleness tonic stays silent until one is.
_last_turn_ts: float = 0.0

# Questions aimed at GAIA herself ("how are you", "anything on your mind?",
# "do you like…") are social contact, not an outward topic pull — they must not
# write curious_about, or the felt-line would report her being curious about
# being asked how she feels. Any second-person reference marks the question
# self-directed; conservative on purpose (a missed outward "can you explain X"
# still gets the stronger knowledge_gap write if grounding fails).
_SELF_DIRECTED_RE = None  # compiled lazily so import stays side-effect-free


def _is_self_directed(sl: str) -> bool:
    global _SELF_DIRECTED_RE
    if _SELF_DIRECTED_RE is None:
        import re
        _SELF_DIRECTED_RE = re.compile(r"\b(you|your|yours|yourself|u)\b")
    return bool(_SELF_DIRECTED_RE.search(sl))


def note_engagement(user_input: str) -> None:
    """Ordinary-chat write hook — called on EVERY user turn (agent_core KR
    region, all paths). Stamps conversation recency for the idleness tonic, and
    writes a WEAK curiosity (0.35 < knowledge_gap's 0.55) for genuine outward
    info-seeking questions even when grounding succeeds — fixes the l11 gap
    where an answerable pipeline left the organ permanently empty."""
    global _last_turn_ts
    try:
        _last_turn_ts = time.time()
        if not appraisal_enabled() or not user_input:
            return
        s = str(user_input).strip()
        if not s or not _looks_like_question(s):
            return
        sl = s.lower()
        if _is_self_directed(sl):
            return
        t = s.rstrip("?").strip()
        tl = t.lower()
        for stem in _Q_STEMS:
            if tl.startswith(stem):
                t = t[len(stem):].strip()
                break
        if t.lower().startswith(("you", "your")):
            return
        _set_curious(t or s, 0.35, source="appraiser:engagement")
    except Exception:
        pass


def _floor_drive(name: str, floor: float, *, source: str) -> None:
    """Raise-only: lift a drive to a tonic floor if it's meaningfully below it.
    Never lowers (events own raises above the floor, KG decay owns the fall),
    and skips small deltas so the tonic doesn't restart decay every tick."""
    af = _af()
    if af is None:
        return
    try:
        snap = af.flatten_current_affect()
        cur = float((snap.get("drives", {}) or {}).get(name, 0.0))
        new = _clamp(floor)
        if new - cur < 0.05:
            return
        af.record_drive(name, new, source=source)
        logger.info("affect tonic %s: %.2f → %.2f (%s)", name, cur, new, source)
    except Exception:
        logger.debug("affect _floor_drive(%s) failed", name, exc_info=True)


def appraise_tonic() -> None:
    """Slow baseline appraisal from continuous signals — called once per
    heartbeat tick (~20 min). Keeps the organ honestly non-empty in ordinary
    ops without inventing feelings (charter: functional drives only):

      - competence ← recent task error-rate: a run of failures leaves a
        standing pull to get things right, even between events.
      - novelty ← conversation idleness: hours without a user turn build a
        quiet tug toward something new (capped below the strong band, so an
        active conversation never shows it).
    """
    try:
        if not appraisal_enabled():
            return
        if len(_recent_outcomes) >= 4:
            rate = 1.0 - (sum(1 for ok in _recent_outcomes if ok) / len(_recent_outcomes))
            if rate > 0.0:
                _floor_drive("competence", min(0.40, 0.55 * rate),
                             source=f"tonic:error_rate:{rate:.2f}")
        if _last_turn_ts > 0.0:
            idle_h = max(0.0, time.time() - _last_turn_ts) / 3600.0
            if idle_h >= 1.0:
                _floor_drive("novelty", min(0.35, 0.05 * idle_h),
                             source=f"tonic:idle:{idle_h:.1f}h")
    except Exception:
        logger.debug("appraise_tonic failed", exc_info=True)

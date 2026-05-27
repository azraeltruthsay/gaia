"""Stakes-clarification decision engine (GAIA_Project-pbb).

Phase 3 of 6ho. When the stakes_classifier returns
``requires_clarification=True``, this module decides whether to actually
ASK and what to say — handling debouncing (don't ask twice in a row),
confidence-thresholding (only ask when we're really unsure), and the
disambiguating-reply parser (when the user comes back with "real, sorry"
or "in-character lol", interpret + clear the pending clarification).

Architecture:

  ┌─ Turn N ─────────────────────────────────────────────────────────┐
  │  classify_stakes("I broke my leg", role_play_active=True)       │
  │      → AMBIGUOUS, requires_clarification=True, conf=0.7         │
  │  decide_clarification(...) → ask=True, question="real or ..."   │
  │  main.py emits the question, stashes pending state, returns.    │
  └─────────────────────────────────────────────────────────────────┘
  ┌─ Turn N+1 ───────────────────────────────────────────────────────┐
  │  pending_clarification(session_id) → {trigger: AMBIGUOUS, ...}  │
  │  resolve_clarification_reply("ooc, my real leg")                │
  │      → resolution="real_world", clears pending state            │
  │  main.py treats the original utterance as real_world from now.  │
  └─────────────────────────────────────────────────────────────────┘

The module is **stateful** — it keeps an in-memory dict of pending
clarifications keyed by session_id, with a TTL so abandoned sessions
don't accumulate. State is intentionally process-local (no /shared
write) — clarifications are inherently turn-local; a process restart
between turns just means we ask again on the next ambiguous case.

Thread-safe via a module-level Lock.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger("GAIA.StakesClarification")


# ── Constants ───────────────────────────────────────────────────────

# Only ask for clarification when classifier confidence is below this.
# Above this threshold, even if requires_clarification=True, we trust
# the classifier — the explicit-marker override paths already pin
# confidence high.
DEFAULT_CONFIDENCE_THRESHOLD = 0.85

# Don't re-ask within this window after a previous ask. The user gets
# one clarification per ambiguous instance — if they don't reply, we
# fall back to ambient context rather than badger.
DEBOUNCE_SECONDS = 60.0

# Pending clarifications expire after this — covers the "user walked
# away after we asked" case. Beyond this, we give up on resolution.
PENDING_TTL_SECONDS = 600.0  # 10 minutes


# ── Question templates ──────────────────────────────────────────────

# Templates are PURE strings — no LLM call. The classifier already
# captured the trigger; we just turn that into a one-liner.

_TEMPLATE_SAFETY_AND_GAME = (
    "Quick check before I respond — real-world or game-state? "
    "(For example: you actually feel sick vs. your character has a poisoned condition.)"
)

_TEMPLATE_SAFETY_AND_RP = (
    "Real quick — is that real-world or in-character? "
    "(For example: your real leg vs. your character's leg.)"
)

_TEMPLATE_FALLBACK = (
    "Want to make sure I respond right — is this in-character or real-life?"
)


# ── State tracking ──────────────────────────────────────────────────


@dataclass
class _PendingState:
    """One row in the per-session pending-clarification dict."""
    asked_at: datetime
    trigger_classification: dict
    question_asked: str
    # When the user replies and we resolve, the original utterance
    # gets re-classified with role_play_active forced according to
    # the resolution. We preserve it so main.py can resume with the
    # right framing.
    original_user_input: str


_pending: dict[str, _PendingState] = {}
_lock = threading.Lock()


def reset_for_tests() -> None:
    """Clear all pending state. Tests only."""
    with _lock:
        _pending.clear()


def _expire_stale(now: datetime) -> None:
    """Drop entries older than PENDING_TTL_SECONDS. Caller holds the lock."""
    stale = [
        sid for sid, p in _pending.items()
        if (now - p.asked_at).total_seconds() > PENDING_TTL_SECONDS
    ]
    for sid in stale:
        del _pending[sid]


# ── Decision API ────────────────────────────────────────────────────


@dataclass
class ClarificationDecision:
    """What to do about a possibly-ambiguous utterance.

    `ask=False` means proceed with normal generation. `ask=True` means
    short-circuit and emit `question` as GAIA's response.
    """
    ask: bool
    question: Optional[str] = None
    reason: str = ""
    suppressed_by: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ask": self.ask,
            "question": self.question,
            "reason": self.reason,
            "suppressed_by": self.suppressed_by,
            "notes": self.notes,
        }


def _pick_question(stakes_result_dict: dict) -> str:
    """Choose the question template based on what the classifier matched."""
    safety = stakes_result_dict.get("matched_safety") or []
    game = stakes_result_dict.get("matched_game") or []
    if safety and game:
        return _TEMPLATE_SAFETY_AND_GAME
    if safety:
        return _TEMPLATE_SAFETY_AND_RP
    return _TEMPLATE_FALLBACK


def decide_clarification(
    stakes_result,
    session_id: str,
    *,
    user_input: str = "",
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    now: Optional[datetime] = None,
) -> ClarificationDecision:
    """Decide whether and how to ask the user for clarification.

    Args:
        stakes_result: StakesResult from classify_stakes (or compatible
            object exposing requires_clarification, confidence, to_dict).
        session_id: per-conversation key for debounce.
        user_input: the original utterance — stashed for resumption.
        confidence_threshold: skip ask above this (trust the classifier).
        now: testability hook.

    Returns:
        ClarificationDecision. If ask=True, the caller should emit
        ``question`` as GAIA's response and short-circuit normal
        generation. Pending state is stashed automatically.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    if not getattr(stakes_result, "requires_clarification", False):
        return ClarificationDecision(
            ask=False, reason="classifier did not flag ambiguity",
        )

    confidence = getattr(stakes_result, "confidence", 1.0)
    if confidence >= confidence_threshold:
        return ClarificationDecision(
            ask=False,
            reason=f"confidence {confidence:.2f} ≥ {confidence_threshold} — trust classifier",
            suppressed_by="confidence_threshold",
        )

    with _lock:
        _expire_stale(now)
        prior = _pending.get(session_id)
        if prior is not None:
            # Debounce: if we already asked recently, don't ask again.
            elapsed = (now - prior.asked_at).total_seconds()
            if elapsed < DEBOUNCE_SECONDS:
                return ClarificationDecision(
                    ask=False,
                    reason=f"already asked {elapsed:.0f}s ago in this session",
                    suppressed_by="debounce",
                )

        # Cleared all suppression paths — ask.
        try:
            stakes_dict = stakes_result.to_dict()
        except Exception:
            stakes_dict = {}
        question = _pick_question(stakes_dict)
        _pending[session_id] = _PendingState(
            asked_at=now,
            trigger_classification=stakes_dict,
            question_asked=question,
            original_user_input=user_input,
        )

    logger.info(
        "Clarification asked (session=%s, stakes=%s, conf=%.2f): %s",
        session_id, getattr(stakes_result, "stakes", "?"),
        confidence, question[:80],
    )
    return ClarificationDecision(
        ask=True,
        question=question,
        reason="ambiguous + low-confidence + not in debounce window",
        notes=[f"confidence={confidence:.2f}"],
    )


# ── Reply parsing ───────────────────────────────────────────────────


@dataclass
class ClarificationReply:
    """Result of parsing the user's response to a clarification question.

    `resolution` is one of: "real_world", "in_game", or "unresolved"
    (couldn't tell — caller should clear pending state and proceed
    with normal classification on the new utterance).
    """
    resolution: str  # "real_world" | "in_game" | "unresolved"
    original_user_input: str = ""
    trigger_classification: dict = field(default_factory=dict)
    pending_cleared: bool = False


# Reply phrases (multi-word, matched as substrings) that signal
# real-world resolution.
_REAL_PHRASES = (
    "real world", "real-world", "real life", "real-life",
    "out of character", "out-of-character", "[ooc]",
    "for real", "no joke", "my real",
    "i'm serious", "im serious", "this is real",
    "real one", "real leg", "real arm", "real hand",
    "not a game", "not in-game", "not in game",
)

# Reply phrases that signal in-game resolution.
_GAME_PHRASES = (
    "in character", "in-character", "[ic]",
    "my character", "the character", "as my character",
    "in game", "in-game", "in the game", "in the story",
    "character's", "character does", "character is",
    "rupert's", "the pc's", "my pc's",
    "narrative", "in the campaign", "in the fiction",
    "rolled", "i rolled",
)

# Single-word tokens. Matched against the reply's whitespace+punctuation
# tokenized set so "real" matches the bare reply "real" but not "really"
# or "realtor". Keep tight — bare-word matches are higher risk.
_REAL_TOKENS = frozenset({
    "real", "irl", "ooc", "actual", "actually", "serious",
})
# "character" is intentionally EXCLUDED — it appears in OOC markers
# ("out of character") and would create false game-signal overlap. Use
# "my character" / "in-character" via the phrase list instead.
_GAME_TOKENS = frozenset({
    "ic", "game", "campaign", "story", "fiction",
    "narrative",
})


def _tokenize(text: str) -> set[str]:
    """Lowercase + strip non-word chars → set of word tokens."""
    return set(re.findall(r"[a-z]+", text.lower()))


def _classify_reply(reply: str) -> str:
    """Return 'real_world' | 'in_game' | 'unresolved' from reply text."""
    if not reply:
        return "unresolved"
    rl = reply.lower()
    tokens = _tokenize(rl)
    has_real = (
        any(p in rl for p in _REAL_PHRASES)
        or bool(tokens & _REAL_TOKENS)
    )
    has_game = (
        any(p in rl for p in _GAME_PHRASES)
        or bool(tokens & _GAME_TOKENS)
    )
    if has_real and not has_game:
        return "real_world"
    if has_game and not has_real:
        return "in_game"
    if has_real and has_game:
        # Both signals — user gave a complicated answer; better to
        # treat as unresolved than guess wrong on the safety side.
        return "unresolved"
    return "unresolved"


def pending_clarification(session_id: str) -> Optional[dict]:
    """Return the pending clarification entry for a session, or None.

    Does NOT clear the entry — call ``resolve_clarification_reply`` to
    interpret a user reply and clear. Returns a snapshot dict with
    keys: asked_at, trigger_classification, question_asked,
    original_user_input.
    """
    with _lock:
        p = _pending.get(session_id)
        if p is None:
            return None
        # Expire on read too — stale entries shouldn't surface.
        now = datetime.now(timezone.utc)
        if (now - p.asked_at).total_seconds() > PENDING_TTL_SECONDS:
            del _pending[session_id]
            return None
        return {
            "asked_at": p.asked_at.isoformat(),
            "trigger_classification": dict(p.trigger_classification),
            "question_asked": p.question_asked,
            "original_user_input": p.original_user_input,
        }


def resolve_clarification_reply(
    session_id: str, reply_text: str,
) -> Optional[ClarificationReply]:
    """Interpret a user reply to a pending clarification.

    If there's no pending clarification for this session, returns None.
    Otherwise returns a ClarificationReply with the resolution and clears
    the pending state (regardless of whether the reply was parseable —
    we only ask once per ambiguous instance).
    """
    with _lock:
        p = _pending.pop(session_id, None)
    if p is None:
        return None
    resolution = _classify_reply(reply_text)
    logger.info(
        "Clarification reply (session=%s): %r → %s",
        session_id, (reply_text or "")[:80], resolution,
    )
    return ClarificationReply(
        resolution=resolution,
        original_user_input=p.original_user_input,
        trigger_classification=dict(p.trigger_classification),
        pending_cleared=True,
    )


def clear_pending(session_id: str) -> bool:
    """Manually clear a session's pending clarification (e.g. user
    started a new topic). Returns True if anything was cleared."""
    with _lock:
        return _pending.pop(session_id, None) is not None

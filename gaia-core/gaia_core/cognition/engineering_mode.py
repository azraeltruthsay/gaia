"""Engineering mode: her engineering turns, as Andy's intern (oeql, 2026-11-19).

A turn is an engineering turn when her study notes are near the question
(knowledge_router's study step). On such a turn, if an engineering adapter is
switched on (ENGINEERING_ADAPTER, empty = off until Azrael approves one), Prime
answers with it and with the intern role card; every other turn runs without
it, so her ordinary answers don't drift (v3/v4 moved them 32-41%).

What the answer stands on decides what she says about it (Azrael: general
knowledge must be validated against something):
  notes   her study notes match strongly: she answers from them, citing the page
  web     notes don't cover it, the web found something: she cites the page
  memory  neither: her answer is from general knowledge. Her other mind answers
          the same question on its own; code compares the two. Agreeing, the
          answer is marked "From general knowledge (worth checking against a
          source)"; disagreeing, she says she isn't sure and would look it up.
          Without another mind awake, it is marked and not checked.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

logger = logging.getLogger("GAIA.EngineeringMode")

ROLE_CARD_PATH = Path(os.environ.get(
    "ENGINEERING_ROLE_CARD",
    "/gaia/GAIA_Project/knowledge/curricula/professional_engineering/role_card.md"))
NEAR = 0.45      # knowledge_router.STUDY_MIN_SCORE: her notes are near the question
STRONG = 0.5     # knowledge_router.STUDY_STRONG: her notes answer it
GENERAL = "From general knowledge (worth checking against a source): "
UNSURE = ("I'm not sure enough to answer that from memory, and my notes don't cover it. "
          "I'd want to look it up, or ask Andy.")
_WORD = re.compile(r"[a-z][a-z'-]{3,}")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def adapter() -> str:
    """The engineering adapter for Prime, or '' (off)."""
    return os.environ.get("ENGINEERING_ADAPTER", "").strip()


def role_card() -> str:
    try:
        text = ROLE_CARD_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""
    return " ".join(ln[1:].strip() for ln in text.splitlines() if ln.startswith(">"))


def turn_kind(results: Iterable) -> Optional[str]:
    """None (not an engineering turn), 'notes', 'web' or 'memory'."""
    results = list(results or [])
    study = [r for r in results if getattr(r, "source", "") == "study"]
    if not study or max(getattr(r, "score", 0.0) for r in study) < NEAR:
        return None
    if max(r.score for r in study) >= STRONG:
        return "notes"
    if any(getattr(r, "source", "") == "web" and len(getattr(r, "content", "") or "") > 50 for r in results):
        return "web"
    return "memory"


def _stems(t: str) -> set:
    return {w[:5] for w in _WORD.findall((t or "").lower())}


def agree(a: str, b: str) -> bool:
    """Two answers given independently agree: most of the first's content words
    are in the second, and every number in the first is."""
    sa = _stems(a)
    if not sa:
        return False
    return len(sa & _stems(b)) / len(sa) >= 0.45 and set(_NUM.findall(a)) <= set(_NUM.findall(b))


def check_general(question: str, answer: str, other: Optional[Callable[[List[dict]], str]]) -> Tuple[str, str]:
    """(final text, verdict) for an answer from general knowledge. verdict:
    'agreed', 'disagreed' or 'unchecked'."""
    body = answer.strip()
    if body.startswith(GENERAL.strip()[:20]):
        body = body[len(GENERAL):].strip() if body.startswith(GENERAL) else body
    if other is None:
        return GENERAL + body, "unchecked"
    try:
        second = other([{"role": "system", "content": "Answer briefly and factually."},
                        {"role": "user", "content": question}]) or ""
    except Exception:
        logger.debug("second mind unavailable", exc_info=True)
        return GENERAL + body, "unchecked"
    if not second.strip():
        return GENERAL + body, "unchecked"
    if agree(body, second):
        return GENERAL + body, "agreed"
    logger.info("engineering mode: her two minds disagree on %r; not answering from memory", question[:60])
    return UNSURE, "disagreed"


def other_mind(model_pool, model_role: str) -> Optional[Callable[[List[dict]], str]]:
    """The mind that didn't answer, to answer the same question on its own."""
    from gaia_core.cognition.self_review import chat_generator
    other = "core" if model_role in ("prime", "cpu_prime") else "prime"
    return chat_generator(model_pool, roles=(other,), max_tokens=260, temperature=0.2)

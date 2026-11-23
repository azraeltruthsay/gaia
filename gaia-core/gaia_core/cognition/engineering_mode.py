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
UNSURE_PLAIN = ("I'm not sure enough to answer that from memory, and I couldn't confirm it "
                "against a source. I'd want to look into it further.")


def check_hedged(text: str) -> bool:
    """The reply is the 'not sure' an answer check put in place of her answer
    (her two minds disagreed), not her own reading of a source. It says
    nothing about the pages she looked up (wtfr: a good SBCC page was marked
    as failing the question after one)."""
    t = re.sub(r"^\s*\[[^\]\n]{1,40}\]\s*", "", text or "").strip()
    return t.startswith((UNSURE[:60], UNSURE_PLAIN[:60]))


_WORD = re.compile(r"[a-z][a-z'-]{3,}")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def adapter() -> str:
    """The engineering adapter for Prime, or '' (off): the registry's active
    one (switched there by an approved activation, live on the next turn),
    else ENGINEERING_ADAPTER."""
    env = os.environ.get("ENGINEERING_ADAPTER", "").strip()
    try:
        from gaia_common.utils.adapter_registry import active
        return active("engineering", env)
    except Exception:
        return env


def role_card() -> str:
    try:
        text = ROLE_CARD_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""
    return " ".join(ln[1:].strip() for ln in text.splitlines() if ln.startswith(">"))


def turn_kind(results: Iterable) -> Optional[str]:
    """None (not an engineering turn), 'notes', 'web' or 'memory'."""
    results = list(results or [])
    # notes from pages she studied; her notes on herself (CFR) aren't engineering
    study = [r for r in results if getattr(r, "source", "") == "study"
             and str(getattr(r, "url", "") or "").startswith("http")]
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


def check_general(question: str, answer: str, other: Optional[Callable[[List[dict]], str]],
                  engineering: bool = True) -> Tuple[str, str]:
    """(final text, verdict) for an answer from general knowledge. verdict:
    'agreed', 'disagreed' or 'unchecked'. Off an engineering turn, her not
    being sure doesn't mention her notes or Andy."""
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
    return (UNSURE if engineering else UNSURE_PLAIN), "disagreed"


def other_mind(model_pool, model_role: str) -> Optional[Callable[[List[dict]], str]]:
    """The mind that didn't answer, to answer the same question on its own."""
    from gaia_core.cognition.self_review import chat_generator
    other = "core" if model_role in ("prime", "cpu_prime") else "prime"
    return chat_generator(model_pool, roles=(other,), max_tokens=260, temperature=0.2)


_CLAIM_RE = re.compile(r"(?:,?\s*but)?\s*I looked it up:?\s*|\s*\((?:Source|That's from)[^)]*\)", re.I)


def _backed(line: str, source: str) -> bool:
    if any(n not in source for n in _NUM.findall(line)):
        return False
    w = _stems(line)
    return not w or len(w & _stems(source)) / len(w) >= 0.6


def check_web(question: str, answer: str, sources: str,
              other: Optional[Callable[[List[dict]], str]]) -> Tuple[str, str]:
    """An answer on a turn with a page in front of her: kept if the page (or her
    notes) backs most of it; otherwise it is general knowledge wearing a
    citation (the v4 eval: 'I looked it up' over the finite element method,
    which the page didn't mention), so the claim comes off and it is checked
    as general knowledge."""
    sents = [x for x in re.split(r"(?<=[.!?])\s+", answer.strip()) if len(x.split()) > 3]
    if not sents:
        return answer, "kept"
    ok = sum(1 for x in sents if _backed(_CLAIM_RE.sub(" ", x), sources))
    if ok / len(sents) >= 0.5:
        return answer, "kept"
    logger.info("engineering mode: %d of %d sentences backed by the page; checking as general knowledge",
                ok, len(sents))
    plain = re.sub(r"^My notes on .*? don't cover that\.?\s*", "", _CLAIM_RE.sub(" ", answer)).strip()
    return check_general(question, plain, other)


_ASKS_RE = re.compile(r"\b(?:gear\s*2|second gear|engineering\s+(?:adapter|mode)|shift\s+(?:up|to\s+(?:gear\s*2|prime)))\b",
                      re.IGNORECASE)


def asked_for(text: str) -> bool:
    """The person asked for engineering mode ("shift to gear 2 and load your
    engineering adapter"): she tried a tool and got a 404 (2026-11-20)."""
    return bool(_ASKS_RE.search(text or ""))

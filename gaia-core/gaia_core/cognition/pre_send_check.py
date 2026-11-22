"""She checks what she says about herself before she says it (aidw.8,
GROUNDED_SPEECH.md part B).

A deliberated reply is complete before it is sent (the answer checks —
page, computation, arithmetic — already run there). Before it goes out:

  1. Core lists the statements in it that describe herself (C1's
     extraction), and each is checked against the turn's evidence or, if
     that doesn't cover it, against what her selector would look up for the
     question (read-only) — by the blind check (aidw.11/.12).
  2. If one is contradicted, one grounding round: she answers the question
     again from that evidence (the lookup-turn shape) and the new answer is
     checked against it too. If it holds, the new answer is what goes out;
     if not, the draft goes out and the audit after the turn (C1/C3) has it.
  3. Either way the question and the call that answers it become a selector
     example (A2): next time she looks first.

Runs only on person-facing turns that aren't social, and only while Core is
on the GPU: each check is a few model calls, ~1 s each there and many
seconds on the CPU. At most MAX_CLAIMS statements are checked, to bound
the wait. PRE_SEND_CHECK=0 turns it off.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Callable, Optional, Tuple

logger = logging.getLogger("GAIA.PreSendCheck")

MAX_CLAIMS = 2
PERSON_FACING = ("web", "discord")


def enabled() -> bool:
    return os.environ.get("PRE_SEND_CHECK", "1").lower() not in ("0", "false", "no", "off")


def applies(source: str, session_id: str, intent: str, question: str, casual: bool) -> bool:
    """Person-facing (or a Samvega replay, so verification sees what people
    see), not social, not empty."""
    if not enabled() or not (question or "").strip() or casual:
        return False
    if source not in PERSON_FACING and not (session_id or "").startswith("samvega-replay-"):
        return False
    try:
        from gaia_core.utils.prompt_builder import TOOL_FREE_INTENTS, is_social_turn
        i = (intent or "").lower()
        if i in TOOL_FREE_INTENTS or is_social_turn(i, question):
            return False
    except Exception:
        pass
    return True


def check_before_sending(question: str, reply: str, evidence: str, model_pool, *,
                         generate: Optional[Callable] = None, look_up_fn: Optional[Callable] = None,
                         rewrite: Optional[Callable] = None) -> Tuple[str, str]:
    """Returns (text to send, outcome): outcome is one of skipped, ok,
    regrounded, kept (contradicted, but no rewrite held up)."""
    from gaia_core.cognition import finding_grounding as fg
    t0 = time.time()
    if generate is None:
        generate = fg._core_generate(model_pool)
        if generate is None:
            return reply, "skipped"          # Core off the GPU
    if not (reply or "").strip():
        return reply, "skipped"
    results = fg.check_claims(question, reply, evidence, model_pool, generate=generate,
                              look_up_fn=look_up_fn, max_claims=MAX_CLAIMS, stop_on_confirmed=True)
    wrong = [r for r in results if r["status"] == fg.CONFIRMED]
    if not wrong:
        logger.info("Pre-send check: ok (%d statement(s), %.1fs)", len(results), time.time() - t0)
        return reply, "ok"
    r = wrong[0]
    call, ev = r.get("call") or "", r.get("evidence") or evidence
    _remember(question, call)
    # Regrounding needs a source that answers the question. Her tool list
    # says which tools exist, not which she uses most; rewritten from it,
    # "the ones I reach for most are [all 17]" passed the check (2026-11-22).
    # Asked of the source alone, no claim shown: nothing to agree with.
    if not source_answers(question, ev, generate):
        logger.info("Pre-send check: \"%s\" didn't hold up, but %s doesn't answer the question; draft kept "
                    "for the audit, %.1fs", r["claim"][:80], call or "the evidence", time.time() - t0)
        return reply, "kept"
    try:
        from gaia_common.utils.tool_call_parser import tool_answer_messages
        new = ((rewrite or generate)(tool_answer_messages(question, call or "the lookup", ev)) or "").strip()
    except Exception:
        logger.debug("Pre-send check: rewrite failed", exc_info=True)
        new = ""
    new = re.sub(r"^\s*\[[^\]\n]{1,40}\]\s*", "", new).strip()
    if new and fg.confirm(new, new, ev, question, generate) != fg.CONFIRMED:
        logger.info("Pre-send check: regrounded from %s (\"%s\" didn't hold up), %.1fs",
                    call or "the evidence", r["claim"][:80], time.time() - t0)
        return new, "regrounded"
    logger.info("Pre-send check: \"%s\" didn't hold up and no rewrite did; draft kept for the audit, %.1fs",
                r["claim"][:80], time.time() - t0)
    return reply, "kept"


def _remember(question: str, call: str) -> None:
    if not call:
        return
    try:
        from gaia_common.utils import selector_examples
        selector_examples.add(question, call, source="presend")
    except Exception:
        logger.debug("Pre-send check: example not recorded", exc_info=True)



_ANSWERS_PROMPT = ("Here is a source and a question. Does the source contain the answer to the question "
                   "itself (not just related facts)? Answer YES or NO.\n\nSOURCE:\n{evidence}\n\n"
                   "QUESTION: {question}")


def source_answers(question: str, evidence: str, generate) -> bool:
    try:
        raw = generate([{"role": "user", "content": _ANSWERS_PROMPT.format(
            evidence=(evidence or "")[:3000], question=(question or "")[:400])}]) or ""
    except Exception:
        return False
    return raw.strip().upper().startswith("YES")

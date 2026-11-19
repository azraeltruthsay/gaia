"""Checks run on her answer after it is written (Azrael, 2026-11-19).

She was asked an AASHTO stopping-sight-distance problem in the Grotto:
  - her method was right but 73.33^2 / 22.4 came out 224.94 (it is 240.1),
    and that picked the wrong choice; nothing computes her arithmetic, it is
    generated text like every other word;
  - she had fetched Wikipedia's stopping sight distance page and answered
    from memory anyway, with a formula the page doesn't give; the page check
    only ran on engineering turns;
  - an answer drawn from her study notes named no source.

So, by code:
  arithmetic   every "expression = number" she writes is computed; a wrong
               result goes back to her with the right value to rewrite, and
               is noted if the rewrite is still wrong
  page         an answer written with a fetched page in front of her has to
               stand on it; if the page doesn't back it she rewrites it from
               the page, citing it, else it is checked as general knowledge
  notes cite   an answer resting on her study notes ends with which notes
"""
from __future__ import annotations

import ast
import logging
import operator
import re
from typing import Callable, Iterable, List, Optional, Tuple

logger = logging.getLogger("GAIA.AnswerChecks")

Generate = Callable[[List[dict]], str]

# ── arithmetic ────────────────────────────────────────────────────────────

_REL_TOL = 0.015          # her rounding (73.33 for 73.333...) stays inside this
_EQ_SPLIT = re.compile(r"\s*(?:=|≈|~=|≅)\s*")
_UNIT = re.compile(r"(?<![\d.])[A-Za-z_][A-Za-z_]*(?:\s*/\s*[A-Za-z]+)?(?:\^?[23²³])?")
_NUMBER = re.compile(r"^[-−]?\d+(?:\.\d+)?$")
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def _normalise(seg: str) -> str:
    s = seg.replace("×", "*").replace("·", "*").replace("÷", "/").replace("−", "-")
    s = s.replace("²", "^2").replace("³", "^3")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = s.replace("$", "").replace("%", "")
    s = _UNIT.sub(" ", s)
    s = re.sub(r"[*/]\s*(?=[)*/]|$)", " ", s)      # an operator a removed unit left dangling
    s = s.replace("^", "**")
    return re.sub(r"\s+", " ", s).strip(" .,:;")


def _eval(expr: str) -> Optional[float]:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 10:
                raise ValueError("exponent")
            return _OPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported")
    try:
        return ev(tree)
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None


def _is_expression(s: str) -> bool:
    return bool(re.search(r"\d", s)) and bool(re.search(r"\d\s*(?:\*\*|[-+*/])\s*[(\d]|\)\s*(?:\*\*|[-+*/])", s))


def _label(line: str) -> str:
    m = re.match(r"^\s*[-*•]?\s*([A-Za-z][A-Za-z0-9_ ()]{0,30}?)\s*(?:=|≈)", line)
    return m.group(1).strip().lower() if m else ""


def arithmetic_errors(text: str) -> List[Tuple[str, str, float]]:
    """(expression as written, the result she wrote, the actual result) for each
    worked step whose result is wrong. Steps carried over lines with the same
    label ("SSD = 1.47 * ...", then "SSD ≈ 275.0 + 224.94") are joined."""
    errors: List[Tuple[str, str, float]] = []
    prev_label, prev_tail = "", ""
    for line in (text or "").splitlines():
        if "=" not in line and "≈" not in line:
            prev_label, prev_tail = "", ""
            continue
        parts = _EQ_SPLIT.split(line.strip())
        label = _label(line)
        if label and label == prev_label and prev_tail:
            parts = [prev_tail] + parts[1:]
        for left, right in zip(parts, parts[1:]):
            expr = _normalise(left)
            if not _is_expression(expr):
                continue
            rnorm = _normalise(right).replace(",", "").replace("−", "-")
            # the next step is a number ("= 7.5 mR/hr") or another expression
            # ("= 120 mR/hr × 1/16"); either way it must come to the same
            if not rnorm or not re.match(r"^[-(\d.]", rnorm):
                continue
            actual, want = _eval(expr), _eval(rnorm)
            if actual is None or want is None:
                continue
            if abs(actual - want) > max(_REL_TOL * abs(actual), 0.01):
                errors.append((left.strip(), right.strip().rstrip(".,;"), actual))
        prev_label, prev_tail = label, parts[-1] if parts else ""
    return errors


def _fmt(x: float) -> str:
    return f"{x:.4g}" if abs(x) < 1000 else f"{x:,.1f}"


REWRITE_ARITH = ("Your answer below has arithmetic mistakes; the correct values were computed exactly:\n{fixes}\n\n"
                 "Rewrite the answer with these values, and redo every step and conclusion that "
                 "depends on them (totals, the chosen option). Keep the same method unless it was "
                 "wrong. Reply with the corrected answer only.\n\nQuestion: {question}\n\nYour answer:\n{answer}")


def check_arithmetic(question: str, answer: str, generate: Optional[Generate],
                     rounds: int = 2) -> Tuple[str, str]:
    """(answer, verdict): 'ok' (nothing to compute or all right), 'fixed', or
    'noted' (still wrong after her rewrite: the right values are appended)."""
    errs = arithmetic_errors(answer)
    if not errs:
        return answer, "ok"
    logger.info("answer checks: %d arithmetic error(s): %s", len(errs),
                "; ".join(f"{e} = {s} (is {_fmt(a)})" for e, s, a in errs)[:300])
    current = answer
    for _ in range(rounds if generate else 0):
        fixes = "\n".join(f"- {e} = {_fmt(a)} (not {s})" for e, s, a in errs)
        try:
            new = (generate([{"role": "user", "content": REWRITE_ARITH.format(
                fixes=fixes, question=question, answer=current)}]) or "").strip()
        except Exception:
            logger.debug("arithmetic rewrite failed", exc_info=True)
            break
        if not new:
            break
        current = new
        errs = arithmetic_errors(current)
        if not errs:
            return current, "fixed"
    note = "; ".join(f"{e} is {_fmt(a)}, not {s}" for e, s, a in errs)
    return f"{current.rstrip()}\n\n*(Checked by calculation: {note}. Steps after it may change.)*", "noted"


# ── page ─────────────────────────────────────────────────────────────────

REWRITE_PAGE = ("You answered from memory, but the page you looked up doesn't back your answer. "
                "Answer again using the page: take formulas, values and definitions from it as it "
                "gives them, keep any values the question itself gives, and say which page you used "
                "({url}). If the page doesn't cover something, say so rather than filling it in.\n\n"
                "The page ({url}):\n{page}\n\nQuestion: {question}\n\nYour first answer:\n{answer}")


def web_pages(results: Iterable) -> List[Tuple[str, str]]:
    """(url, text) for each page she read among the grounding results. Search
    snippets aren't pages: a worked answer can't be backed by two lines of
    lead text, and checking against them turned a right answer into "not sure"."""
    out = []
    for r in results or []:
        if getattr(r, "source", "") == "web" and getattr(r, "read", False) \
                and len(getattr(r, "content", "") or "") > 50:
            out.append((getattr(r, "url", "") or "", r.content))
    return out


def check_page(question: str, answer: str, results: Iterable, same: Optional[Generate],
               other: Optional[Generate], engineering: bool = False) -> Tuple[str, str]:
    """An answer written with a page in front of her: kept if the page backs it;
    else rewritten from the page (and kept if that is backed); else checked as
    general knowledge. Verdicts: 'kept', 'rewritten', or check_general's."""
    from gaia_core.cognition import engineering_mode as em
    pages = web_pages(results)
    if not pages:
        return answer, "no_page"
    sources = " ".join(t for _, t in pages)
    if _backed_share(answer, sources) >= 0.5:
        return answer, "kept"
    url, page = max(pages, key=lambda p: len(em._stems(p[1]) & em._stems(question)))
    # Her answering mind rewrites from the page first; if that still isn't on
    # the page, her other mind tries (Prime read "March 2024 / codes in
    # effect: 2021 IBC" as 2024 twice; Core read it right both times).
    for mind in (same, other):
        if mind is None:
            continue
        try:
            new = (mind([{"role": "user", "content": REWRITE_PAGE.format(
                url=url or "the page", page=page[:6000], question=question, answer=answer)}]) or "").strip()
        except Exception:
            logger.debug("page rewrite failed", exc_info=True)
            new = ""
        share = _backed_share(new, sources) if new else 0.0
        logger.info("answer checks: page rewrite (%.2f backed): %r", share, new[:200])
        if new and share >= 0.5:
            if url and url not in new:
                new = f"{new.rstrip()}\n\n(Source: {url})"
            logger.info("answer checks: rewrote the answer from %s", url)
            return new, "rewritten"
    logger.info("answer checks: page doesn't back the answer; checking it as general knowledge")
    plain = em._CLAIM_RE.sub(" ", answer).strip()
    return em.check_general(question, plain, other, engineering=engineering)


def _backed_share(answer: str, sources: str) -> float:
    from gaia_core.cognition import engineering_mode as em
    sents = [x for x in re.split(r"(?<=[.!?])\s+|\n+", answer.strip()) if len(x.split()) > 3]
    if not sents:
        return 1.0
    return sum(1 for x in sents if em._backed(em._CLAIM_RE.sub(" ", x), sources)) / len(sents)


# ── notes citation ───────────────────────────────────────────────────────

def cite_notes(answer: str, results: Iterable, min_overlap: float = 0.3) -> str:
    """Name the study note an answer rests on, when she didn't."""
    from gaia_core.cognition import engineering_mode as em
    notes = [r for r in results or [] if getattr(r, "source", "") == "study"]
    if not notes or not answer.strip():
        return answer
    a = em._stems(answer)
    if not a:
        return answer
    best = max(notes, key=lambda r: len(a & em._stems(r.content)))
    if len(a & em._stems(best.content)) / len(a) < min_overlap:
        return answer
    title, url = (getattr(best, "title", "") or "").strip(), (getattr(best, "url", "") or "").strip()
    named = [x for x in (title, url) if x]
    if not named or any(x.lower() in answer.lower() for x in named):
        return answer
    sec = re.search(r"\b(?:WAC|RCW)\s+\d+(?:[-.]\d+[A-Za-z]?)*", title)
    if sec and sec.group(0).strip().lower() in answer.lower():
        return answer
    where = f"{title}, {url}" if title and url else (title or url)
    return f"{answer.rstrip()}\n\n*(From my study notes: {where})*"

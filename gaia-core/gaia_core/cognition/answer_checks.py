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
    s = re.sub(r"π|\bpi\b", " 3.141592653589793 ", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d)\s+(?=3\.141592653589793)", " * ", s)
    s = s.replace("²", "^2").replace("³", "^3")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = s.replace("$", "").replace("%", "")
    s = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", s)   # "half-value" is words, not a minus
    s = _UNIT.sub(" ", s)
    s = re.sub(r"^\s*[-+*/]\s*(?=[\d(])", "", s)       # an operator left at the front by removed words
    s = re.sub(r"[*/]\s*(?=[)*/]|$)", " ", s)      # an operator a removed unit left dangling
    # what a removed compound unit leaves: "gal/(ft³/s)" -> "/( )"
    for _ in range(3):
        s = re.sub(r"\(\s*[*/]?\s*\)", " ", s)
        s = re.sub(r"[*/]\s*(?=[)*/]|$)", " ", s)
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


# ── an answer that only points at her notes ─────────────────────────────

_POINTER_RE = re.compile(r"^\s*(?:\[\w+\]\s*)?My notes on\b.{0,200}?(?<!n't\s)(?<!not\s)\b(?:cover|covers|address|discuss)\s+(?:that|this|it)\b",
                         re.IGNORECASE | re.DOTALL)


def points_without_answering(answer: str) -> bool:
    """"My notes on WAC 196-23: Stamping and seals, from <url> cover that -- it
    outlines the rules ..." and nothing else: the v5 adapter's "My notes on X"
    opening (learned from its "don't cover that" answers) firing on a question
    the notes do answer (2026-11-19, Azrael's quiz)."""
    m = _POINTER_RE.match(answer or "")
    if not m or re.search(r"\b(?:don'?t|doesn'?t|do not|does not)\s+cover", answer[:m.end()], re.IGNORECASE):
        return False
    rest = re.sub(r"https?://\S+", "", answer[m.end():])
    return len(rest.split()) < 40


# ── worked problems: the numbers computed in code ───────────────────────

import math as _math

_FUNCS = {"sqrt": _math.sqrt, "log": _math.log, "log10": _math.log10, "ln": _math.log, "exp": _math.exp,
          "sin": _math.sin, "cos": _math.cos, "tan": _math.tan, "abs": abs, "radians": _math.radians}
_CONSTS = {"pi": _math.pi, "e": _math.e}
_STEP_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^#\n]+?)\s*(?:#\s*(.*))?$")

STEPS = ("Work this problem as steps a calculator will run, one per line, written exactly as\n"
         "name = expression  # what it is, with units\n"
         "First write one line for each given value in consistent units, with any conversion shown "
         "(a diameter of 6 inches is D = 6 / 12  # ft). Then the formula steps. Use only numbers, names "
         "from earlier lines, + - * / ** ( ), pi, sqrt, log, exp; units go in the # comment. Convert "
         "results with explicit factors (448.831 gpm per cfs, 7.48052 gal per ft3). One line per "
         "quantity asked for. No other text.\n\nProblem: {question}")
REWRITE_CALC = ("Your answer's numbers don't match the calculation. These steps were computed exactly:\n"
                "{steps}\n\nAnswer the problem again with these values: show the steps briefly and give "
                "the result with units. Reply with the answer only.\n\nProblem: {question}\n\n"
                "Your first answer:\n{answer}")


def _eval_named(expr: str, env: dict) -> Optional[float]:
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-")
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expr)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name):
            if n.id in env:
                return env[n.id]
            if n.id in _CONSTS:
                return _CONSTS[n.id]
            raise ValueError(n.id)
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 10:
                raise ValueError("exponent")
            return _OPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return float(_FUNCS[n.func.id](*[ev(a) for a in n.args]))
        raise ValueError("unsupported")
    try:
        return ev(tree)
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None


def run_steps(text: str) -> List[Tuple[str, float, str]]:
    """(name, value, comment) for each step that computes; stops at one that doesn't."""
    env, out = {}, []
    for line in (text or "").splitlines():
        line = line.strip().strip("`").strip()
        m = _STEP_RE.match(line)
        if not m:
            continue
        expr = m.group(2)
        val = _eval_named(expr, env)
        if val is None:
            # a given value with its unit written in: "I0 = 400 mR/hr"
            lit = re.match(r"^\s*(-?\d[\d,]*(?:\.\d+)?)\s*[A-Za-z°%µ][\w/°%²³^.\- ]*$", expr)
            val = float(lit.group(1).replace(",", "")) if lit else None
        if val is None:
            break
        env[m.group(1)] = val
        out.append((m.group(1), val, (m.group(3) or "").strip()))
    return out


_ANS_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> List[float]:
    return [float(x.replace(",", "")) for x in _ANS_NUM.findall(text or "")]


def check_computation(question: str, answer: str, generate: Optional[Generate],
                      second: Optional[Generate] = None) -> Tuple[str, str]:
    """A worked problem answered without working (or with it): she writes the
    calculation as steps, code runs them, and an answer whose results aren't
    among the computed values is rewritten from them. Verdicts: 'skipped',
    'ok', 'rewritten', 'noted'. (2026-11-19: "5.76 cfs" for a 6-inch pipe at
    3 ft/s, which is 0.589.)"""
    from gaia_core.cognition.nlu.intent_detection import is_worked_problem
    if generate is None or not is_worked_problem(question):
        return answer, "skipped"
    # Two tries at the calculation; it counts only when they agree (one try
    # set a 6-inch pipe up as 3 ft and "corrected" her to 21 cfs).
    runs = []
    for gen in (generate, second or generate):
        steps_run = []
        for g in (gen, generate):      # her other mind may be unloaded
            try:
                steps_run = run_steps(g([{"role": "user", "content": STEPS.format(question=question)}]) or "")
            except Exception:
                logger.debug("computation steps failed", exc_info=True)
            if steps_run:
                break
        runs.append(steps_run)
    if not all(runs):
        return answer, "skipped"
    def same(x, y):
        return abs(x - y) <= max(_REL_TOL * max(abs(x), abs(y)), 0.01)
    a, b = runs[0][-1][1], runs[1][-1][1]
    steps = runs[0]
    if not same(a, b):
        # a third try breaks the tie (Prime 12.5, Core 23.67 on a shielding problem)
        try:
            third = run_steps(generate([{"role": "user", "content": STEPS.format(question=question)}]) or "")
        except Exception:
            third = []
        c = third[-1][1] if third else None
        if c is not None and same(c, a):
            steps = runs[0]
        elif c is not None and same(c, b):
            steps = runs[1]
        else:
            logger.info("answer checks: calculations disagree (%s, %s, %s); leaving the answer",
                        _fmt(a), _fmt(b), _fmt(c) if c is not None else "-")
            return answer, "unsure"
    given = set(_numbers(question))
    finals = [v for _, v, _ in steps[-3:]]
    # everything she stated counts, a number the question also gave included
    # ("(PD/2t = 7,200 psi)" in the question hid her 7,200 and she was
    # "corrected" to 7,200, 2026-11-20)
    stated = _numbers(answer)

    def near(x, ys):
        return any(abs(x - y) <= max(_REL_TOL * abs(y), 0.01) for y in ys)
    computed = [v for _, v, _ in steps] + list(given)
    # right if what it states is computed (a partial answer isn't a wrong one)
    if stated and all(near(x, computed) for x in stated[-2:]) and (
            near(finals[-1], stated) or any(near(x, finals) for x in stated)):
        return answer, "ok"
    shown = "\n".join(f"{n} = {_fmt(v)}" + (f"  ({c})" if c else "") for n, v, c in steps)
    logger.info("answer checks: computed %s; her answer said %s", shown.replace("\n", "; ")[:300],
                stated[-3:])
    try:
        new = (generate([{"role": "user", "content": REWRITE_CALC.format(
            steps=shown, question=question, answer=answer)}]) or "").strip()
    except Exception:
        new = ""
    if new and near(finals[-1], _numbers(new)):
        if len(new) < 80:
            new = f"{new.rstrip('.')}. Worked out: {plain_steps(steps, question, given)}."
        return new, "rewritten"
    # Her rewrite missed the computed value, but the calculation stands (two
    # tries agreed, code ran it): said in plain words from the steps (her
    # rewrite of the 20 mR/hr answer never came to 12.5, 2026-11-20).
    return phrase_result(question, steps, given, generate), "rewritten"


# ── the computed steps in plain words ──────────────────────────────────

_ABBREV = {"hvl": "half-value layers", "hvls": "half-value layers", "q": "flow", "cfs": "cfs", "gpm": "gpm",
           "psi": "psi", "ft": "ft", "ft2": "ft²", "ft3": "ft³", "d": "diameter", "a": "area", "v": "velocity",
           "p": "pressure", "t": "thickness", "r": "radius", "i": "intensity", "num": "number", "sigma": "stress"}
_UNIT_WORDS = {"per": "/", "hr": "hr", "sec": "s", "seconds": "s", "meters": "m", "meter": "m", "feet": "ft"}


def _words(name: str) -> str:
    """exposure_rate_at_4m_with_hvl -> exposure rate at 4 m with half-value layers"""
    parts = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", name.replace("_per_", " per "))
    out = []
    for p in parts:
        low = p.lower()
        out.append(_ABBREV.get(low, p if (p.isupper() and len(p) > 1) or p[0].isdigit() else low))
    text = " ".join(out)
    text = re.sub(r"(\d) (m|ft|in|s)\b", r"\1 \2", text)
    return re.sub(r"\bmR per hr\b", "mR/hr", text)


def _unit(comment: str) -> str:
    """The unit in a step's comment, if it is short enough to be one."""
    u = (comment or "").split("(")[0].strip()
    return u if u and len(u) <= 12 and len(u.split()) <= 2 else ""


def _question_unit(question: str, value: float, steps) -> str:
    """No unit on the result: the unit of the given quantity it derives from
    (the first given whose unit names a rate, else the first given)."""
    units = re.findall(r"\d+(?:\.\d+)?\s*([A-Za-zµ°][\w/°²³]*(?:/[A-Za-z]+)?)", question or "")
    rates = [u for u in units if "/" in u]
    return (rates or units or [""])[0]


def plain_steps(steps, question: str = "", given=()) -> str:
    out = []
    for n, v, c in steps:
        if v in given:
            continue
        u = _unit(c)
        out.append(f"{_words(n)} = {_fmt(v)}{(' ' + u) if u else ''}")
    return "; ".join(out)


PHRASE = ("Say this worked result in one or two plain sentences, as a correction of your earlier answer: "
          "what the right result is and the one or two steps that get there, with units. Use the numbers "
          "exactly as given. No variable names, no formatting.\n\nProblem: {question}\n\nComputed steps:\n{steps}")


def phrase_result(question: str, steps, given=(), generate: Optional[Generate] = None) -> str:
    """The computed result in plain words: she phrases it, and her sentence
    is kept only if it carries the computed value and none of its numbers
    contradict the steps; otherwise code writes it from the step names."""
    name, val, comment = steps[-1]
    unit = _unit(comment) or _question_unit(question, val, steps)
    if generate is not None:
        try:
            said = (generate([{"role": "user", "content": PHRASE.format(
                question=question, steps=plain_steps(steps, question, given))}]) or "").strip()
        except Exception:
            said = ""
        nums = _numbers(said)
        known = [v for _, v, _ in steps] + list(given) + [float(k) for k in range(1, 13)]
        if said and any(abs(x - val) <= max(_REL_TOL * abs(val), 0.01) for x in nums) and \
                all(any(abs(x - k) <= max(_REL_TOL * abs(k), 0.01) for k in known) for x in nums) and \
                not re.search(r"\b[a-z]+_[a-z_]+\b", said):
            return said
    final_words = set(_words(name).split()) - {"at", "with", "without", "of", "the", "per", "m", "ft"}

    def unit_of(n, c):
        # an intermediate measuring the same thing as the result shares its unit
        return _unit(c) or (unit if final_words & set(_words(n).split()) else "")
    middle = [f"{_words(n)} is {_fmt(v)}{(' ' + unit_of(n, c)) if unit_of(n, c) else ''}" for n, v, c in steps[:-1]
              if v not in given][-2:]
    lead = ("Working it through: " + ", and ".join(middle) + ". ") if middle else ""
    return f"{lead}So the {_words(name)} is {_fmt(val)}{(' ' + unit) if unit else ''}."

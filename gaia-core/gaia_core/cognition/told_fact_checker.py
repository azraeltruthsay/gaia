"""Check told facts against sources she didn't hear them from (aym9).

She can't be asked whether a page supports a claim: shown the claim, Core said
"supports" for nearly everything, "The Eiffel Tower is in Berlin" included
(known-answer test, 2026-11-03). So she never sees it. One name or number is
blanked out, the search looks for the rest, and she fills the blank from each
page. Her answer counts only if its words are on the page, so it can't come
from her weights, and the comparison with the claim is made here, not by her.

Strict on purpose: most true facts stay unchecked (search snippets are short).
Not yet checked is an honest state; falsely validated would poison the ladder.
"""
import logging
import re
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("GAIA.ToldFactChecker")

_SPAN_RE = re.compile(r"(?:[A-Z][\w'’-]*|\d[\d,]*)(?:(?:\s+(?:of\s+)?|-)(?:[A-Z][\w'’-]*|\d[\d,]*))*")
_NOT_STATED_RE = re.compile(r"not\s+stated", re.IGNORECASE)
_MAX_BLANKS = 3
_MIN_SITES = 2
_MAX_CONTRARY_WORDS = 5
_PASSAGE_CHARS = 800
_FILLER = {"the", "a", "an", "of", "in"}


def _words(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def blanks(fact: str) -> List[str]:
    """The names and numbers in a fact, each of which can be hidden in turn."""
    out = []
    for m in _SPAN_RE.finditer(fact or ""):
        term = m.group(0)
        if m.start() == 0 and term.split()[0] in ("The", "A", "An", "It", "This", "That"):
            term = " ".join(term.split()[1:])
        if term:
            out.append(term)
    return out[-_MAX_BLANKS:]


def _read_blank(blank: str, passage: str, generate: Callable[[list], str]) -> str:
    prompt = (f"Passage:\n{passage}\n\nUsing only the passage, what goes in the blank? "
              f"Answer with just those words. If the passage doesn't say, answer NOT STATED.\n\n{blank}")
    return (generate([{"role": "user", "content": prompt}]) or "").strip().split("\n")[0]


def _site(url: str) -> str:
    host = re.sub(r"^https?://", "", url or "").split("/")[0].lower()
    return ".".join(host.split(".")[-2:])


def judge_blank(term: str, readings: List[dict], claim: str = "") -> dict:
    """What the pages said for one blank, each site counted once. `readings`
    are {url, passage, answer}. Answers not on their own page are dropped, and
    an answer only repeating the claim's other words says nothing against it."""
    supports, others = {}, {}
    for r in readings:
        answer, site = (r.get("answer") or "").strip(), _site(r.get("url", ""))
        said = _words(answer)
        if not site or not said or _NOT_STATED_RE.search(answer) or not said - _FILLER <= _words(r.get("passage", "")):
            continue
        if _words(term) <= said or said - _FILLER <= _words(term):
            supports.setdefault(site, r["url"])
        elif len(answer.split()) <= _MAX_CONTRARY_WORDS and not said - _FILLER <= _words(claim):
            others.setdefault(" ".join(sorted(said - _FILLER)), {})[site] = {"url": r["url"], "answer": answer}
    contrary = max(others.values(), key=len, default={})
    return {
        "term": term,
        "supports": list(supports.values()),
        "contrary": [v for v in contrary.values()] if len(contrary) >= _MIN_SITES else [],
    }


def decide(results: List[dict]) -> str:
    """supported: enough sites filled each blank back in (one blank may miss
    if there are three). contradicted: sites agree on something else, and none
    backs the claim. Otherwise nothing either way."""
    if not results:
        return "uncheckable"
    if any(r["contrary"] and len(r["supports"]) < _MIN_SITES for r in results):
        return "contradicted"
    missing = sum(len({*r["supports"]}) < _MIN_SITES for r in results)
    return "supported" if missing == 0 or (len(results) >= 3 and missing == 1) else "no_evidence"


def check_fact(fact: str, search: Callable[[str], List[dict]], generate: Callable[[list], str]) -> dict:
    results = []
    for term in blanks(fact):
        blank = fact.replace(term, "____", 1)
        readings = []
        for hit in search(blank.replace("____", " ").replace(".", " ")) or []:
            passage = (hit.get("snippet") or "")[:_PASSAGE_CHARS]
            if passage:
                readings.append({"url": hit.get("url", ""), "passage": passage,
                                 "answer": _read_blank(blank, passage, generate)})
        results.append(judge_blank(term, readings, fact))
    return {"outcome": decide(results), "blanks": results}


def check_told_facts(call: Callable[[str, dict], dict], generate: Callable[[list], str], limit: int = 3) -> Dict[str, int]:
    """One sleep pass: the facts due a check, each result written to its record."""
    def search(query: str) -> List[dict]:
        found = call("web_search", {"query": query, "max_results": 5, "remember": False})
        if found.get("ok") is False:
            raise BudgetSpent(found.get("error", "search failed"))
        return found.get("results") or []

    tally: Dict[str, int] = {}
    for rec in call("palace_told_to_check", {"limit": limit}).get("results") or []:
        try:
            result = check_fact(rec["snippet"], search, generate)
        except BudgetSpent as e:
            logger.info("Told-fact check stopped: %s", e)
            break
        call("palace_record_check", {"id": rec["id"], **result})
        tally[result["outcome"]] = tally.get(result["outcome"], 0) + 1
    if tally:
        logger.info("Told-fact check: %s", tally)
    return tally


class BudgetSpent(Exception):
    """A search was refused, so the rest wait for the next pass."""

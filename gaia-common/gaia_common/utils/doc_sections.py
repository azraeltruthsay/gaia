"""Read the part of a document that was asked for (g47g).

Tool results were cut at 4,000 characters. Article VII of her constitution starts
at 4,474, so both minds read the first half and invented it, confidently; given
the full text, both got it right. A long document now comes back as the sections
the question points at, and always says when it is only a part.
"""
import re
from typing import List, Tuple

_HEADING_RE = re.compile(r"^(?:#{1,6}\s+.+|\*\*[^*\n]{3,120}\*\*\s*)$", re.MULTILINE)
_ARTICLE_RE = re.compile(r"\barticle\s+([IVXLC]+|[A-Z])\b", re.IGNORECASE)
_STOP = frozenset("what which does this that your about have with from there their tell says said read file document "
                  "constitution declaration article artisanal intelligence gaia please show give think".split())


def sections(text: str) -> List[Tuple[str, str]]:
    """(heading, body) for each section. Text before the first heading has ""."""
    marks = list(_HEADING_RE.finditer(text or ""))
    if not marks:
        return [("", text or "")]
    out = [("", text[:marks[0].start()])] if text[:marks[0].start()].strip() else []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((m.group(0).strip("#* \n"), text[m.end():end].strip()))
    return out


def asked_articles(question: str) -> List[str]:
    return [a.upper() for a in _ARTICLE_RE.findall(question or "")]


def _terms(text: str) -> set:
    """Word stems, enough that "retraining" finds "retrain"."""
    return {w[:6] for w in re.findall(r"[a-z]{4,}", (text or "").lower()) if w not in _STOP}


def _matching(parts: List[Tuple[str, str]], question: str) -> List[Tuple[str, str]]:
    articles = asked_articles(question)
    if articles:
        wanted = [p for p in parts if any(re.search(rf"\barticle\s+{a}\b", p[0], re.IGNORECASE) for a in articles)]
        if wanted:
            return wanted
    terms = _terms(question)
    if not terms:
        return []
    scored = [(2 * len(terms & _terms(h)) + len(terms & _terms(b)), i) for i, (h, b) in enumerate(parts)]
    return [parts[i] for score, i in sorted(scored, key=lambda s: (-s[0], s[1])) if score > 0][:6]


def excerpt_for(text: str, question: str = "", limit: int = 4000) -> str:
    """`text` if it fits; else the sections `question` points at, or the opening,
    with a note saying how much this is."""
    if len(text or "") <= limit:
        return text or ""
    chosen = _matching(sections(text), question)
    body = "\n\n".join(f"{h}\n{b}".strip() for h, b in chosen)[:limit] if chosen else text[:limit]
    what = "the sections that match the question" if chosen else "the opening"
    return f"{body}\n\n(This is only part of a longer document: {what}, {len(body)} of {len(text)} characters.)"

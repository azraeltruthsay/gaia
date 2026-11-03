"""Work memory — texts GAIA knows by heart, word for word, with their sources.

The weights are how she thinks, not what she has read: asked for a poem from
the weights alone, Core wrote 2-9% of The Raven's real words and presented the
rest as the poem (yx93). A work she has really read belongs here instead.

A text is learned only when two independent sites agree on its words, and only
the part they agree on is kept. That is also what strips a page down to the
work: menus, summaries and catalog entries are not shared between sites
(gutenberg.org/ebooks/17192 is about The Raven, is 2.6k of text, and contains
none of it). Unlike lookups (first 1500 chars, swept when idle), these are
whole and kept for good.

Layout: one JSON record per work under <palace root>/knowledge/works/<id>.json.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger("GAIA.WorkMemory")

_SHINGLE_WORDS = 5
# Editions differ ("visitor"/"visiter"): one changed word leaves a gap of two.
# Much wider and a shared site menu joins on to the front of the poem.
_MAX_GAP_WORDS = 8
# Enough for the shortest poems; shared page furniture rarely runs this long.
MIN_AGREED_WORDS = 12
_CAPTION_CHARS = 500

_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_ORDINALS = {
    "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}
_UNIT = r"(?P<unit>stanza|verse|line|paragraph)s?"
_COUNT = r"(?P<count>\d+|" + "|".join(k for k in _NUMBER_WORDS if len(k) > 2) + r")"
_FROM_END = r"(?P<end>first|opening|last|closing|final)"
_PART_RE = re.compile(
    r"\b(?:" + _FROM_END + r"\s+(?:" + _COUNT + r"\s+)?" + _UNIT
    + r"|(?P<ordinal>" + "|".join(_ORDINALS) + r")\s+(?P<ounit>stanza|verse|paragraph|line))\b",
    re.IGNORECASE,
)
_PART_OF_RE = re.compile(
    r"(?i)\b(?:the\s+)?(?:(?:first|opening|last|closing|final|"
    + "|".join(_ORDINALS) + r")\s+)?(?:" + _COUNT.replace("?P<count>", "?:")
    + r"\s+)?(?:stanza|verse|line|paragraph)s?\s+(?:of|from|in)\s+"
)
_PREAMBLE_RE = re.compile(
    r"(?i)^(?:(?:hey|hi|gaia|ok|okay)[,!\s]+)*(?:(?:please|can you|could you|would you|will you)\s+)*"
    r"(?:recite|read|share|give me|show me|write out|type out|present|say)\s+(?:me\s+)?"
)
_QUESTION_RE = re.compile(r"(?i)^(?:what(?:'s|’s| is| are| was)|tell me|how does|how do)\s+")
_TRAILER_RE = re.compile(r"(?i)[\s,]+(?:for me|please|if you (?:can|could|would)|from memory)\s*$")
_FORM_RE = re.compile(r"(?i)^(?:the\s+)?(?:full\s+|whole\s+|entire\s+|complete\s+)?(?:text|poem|speech)\s+(?:of\s+)?")
_TERM_NOISE = frozenset("""
a an the of and in on to from for by poem poems speech text full whole entire complete
""".split())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tokens(text: str) -> List[Tuple[str, int, int]]:
    """(normalised word, start, end): punctuation and case are edition noise."""
    return [(m.group().lower().replace("’", "").replace("'", ""), m.start(), m.end())
            for m in _WORD_RE.finditer(text or "")]


def agreed_text(a: str, b: str, min_words: int = MIN_AGREED_WORDS) -> Optional[str]:
    """The longest stretch of `a` that `b` also contains, whole lines, as `a`
    has it. None when the two don't share enough to call it the same text."""
    ta, tb = _tokens(a), _tokens(b)
    n = _SHINGLE_WORDS
    if len(ta) < n or len(tb) < n:
        return None
    wb = [t[0] for t in tb]
    shared = {tuple(wb[i:i + n]) for i in range(len(wb) - n + 1)}
    wa = [t[0] for t in ta]

    best: Optional[Tuple[int, int, int]] = None  # (matched words, first, last)
    run = None
    for i in range(len(wa) - n + 1):
        if tuple(wa[i:i + n]) not in shared:
            continue
        if run and i - run[2] <= _MAX_GAP_WORDS:
            run = (run[0] + min(i + n - 1 - run[2], n), run[1], i + n - 1)
        else:
            run = (n, i, i + n - 1)
        if best is None or run[0] > best[0]:
            best = run
    if best is None or best[0] < min_words:
        return None

    # Whole lines, except an end line matched only in part: "forgotten lore /
    # The end of" can be shared while the next words are each site's own.
    line_of, line, pos = [], 0, 0
    for _, begin, _ in ta:
        line += a.count("\n", pos, begin)
        pos = begin
        line_of.append(line)
    first, last = best[1], best[2]
    if first - bisect_left(line_of, line_of[first]) > 1:
        first = bisect_right(line_of, line_of[first])
    if bisect_right(line_of, line_of[last]) - 1 - last > 1:
        last = bisect_left(line_of, line_of[last]) - 1
    if first > last:
        return None

    start = a.rfind("\n", 0, ta[first][1]) + 1
    end = a.find("\n", ta[last][2])
    return a[start:end if end != -1 else len(a)].strip()


def _site(page: dict) -> str:
    """The site behind a host: en. and simple.wikipedia.org are one source."""
    host = (page.get("domain") or urlparse(page.get("url", "")).netloc or "").lower()
    return ".".join(host.split(".")[-2:])


def _reads_as_about(text: str, name: str) -> bool:
    """True if agreed words describe the work instead of being it: two sites
    can share a catalog blurb ('"The Raven" by Edgar Allan Poe is a narrative
    poem...'), and a work seldom names itself in quotes or its own author."""
    head = re.split(r"(?i)\s+by\s+", name or "", maxsplit=1)
    title = head[0].strip().strip("\"“”")
    if title and re.search(r"[\"“']\s*" + re.escape(title) + r"\s*[,.]?[\"”']", text, re.IGNORECASE):
        return True
    # A short text naming itself mid-sentence is a caption: "United States
    # Declaration of Independence, broadside printed by John Dunlap (1776)."
    # Long ones can: "In Xanadu did Kubla Khan".
    bare = re.sub(r"(?i)^the\s+", "", title)
    if len(text) < _CAPTION_CHARS and len(title_terms(bare)) > 1:
        found = re.search(re.escape(bare), text, re.IGNORECASE)
        if found and found.start() > 0:
            return True
    if len(head) > 1 and head[1].strip() and head[1].strip().lower() in text.lower():
        return True
    return bool(re.search(r"(?i)\b(generated summary|narrative poem|download for free|is a (poem|speech) (by|written))\b", text))


def best_agreement(pages: List[dict], name: str) -> Optional[Dict[str, object]]:
    """The fullest text two different sites agree on, as {text, sources}.

    Two pages of one site share their furniture, so sites must differ. Each
    page must mention the title and together they must name everything asked
    for, so another poem of the same length can't pass. Of the two copies, take
    the one with stanza breaks and fewest lines: editions rewrap the same words.
    """
    wanted, title = name_terms(name), title_terms(name)
    usable = []
    for page in pages:
        words = {t[0] for t in _tokens(f"{page.get('title', '')} {page.get('content', '')}")}
        if page.get("content") and title <= words:
            usable.append((page, words))

    best: Optional[Dict[str, object]] = None
    best_key: Tuple[int, int, int] = (0, 0, 0)
    for i, (pa, wa) in enumerate(usable):
        for pb, wb in usable[i + 1:]:
            if _site(pa) == _site(pb) or not wanted <= (wa | wb):
                continue
            for x, y in ((pa, pb), (pb, pa)):
                agreed = agreed_text(x["content"], y["content"])
                if not agreed:
                    continue
                agreed = strip_trailer(strip_heading(agreed, name), name)
                if len(_tokens(agreed)) < MIN_AGREED_WORDS or _reads_as_about(agreed, name):
                    continue
                # Nearly equal lengths are one text in two editions, so length
                # only counts in coarse steps; then layout decides.
                key = (round(len(agreed) / 300) or 1, int("\n\n" in agreed), -agreed.count("\n"))
                if best is None or key > best_key:
                    best_key = key
                    best = {"text": agreed, "sources": [
                        {k: pg.get(k, "") for k in ("url", "title", "domain")} for pg in (x, y)]}
    return best


def work_name(request: str) -> str:
    """The work a request names: 'Please recite the first three stanzas of
    The Raven by Edgar Allan Poe.' -> 'The Raven by Edgar Allan Poe'."""
    name = (request or "").strip().rstrip(".!?")
    name = _PREAMBLE_RE.sub("", name)
    name = _QUESTION_RE.sub("", name)
    name = _TRAILER_RE.sub("", name)
    name = _PART_OF_RE.sub("", name, count=1)
    name = _FORM_RE.sub("", name)
    return name.strip().strip("\"“”‘’").strip()


def name_terms(name: str) -> set:
    return {t[0] for t in _tokens(name)} - _TERM_NOISE


def title_terms(name: str) -> set:
    """Terms of the title alone, before any 'by <author>'."""
    return name_terms(re.split(r"(?i)\s+by\s+", name or "", maxsplit=1)[0])


_WANTS_WORDS_RE = re.compile(
    r"(?i)\b(recite|recitation|read me|read out|read aloud|word for word|verbatim|full text|"
    r"whole text|quote|type out|write out)\b"
)
_ABOUT_RE = re.compile(
    r"(?i)\b(mean|means|meaning|about|explain|why|analy[sz]e|interpret|summar\w+|theme|who|"
    r"compare|recall|remember)\b"
)


def asks_for_the_words(request: str) -> bool:
    """True if the request wants the text itself ('recite X', 'the first line
    of X'), not an answer about it ('what does the bird say in X?'). The intent
    classifier files both under recitation."""
    if _WANTS_WORDS_RE.search(request or ""):
        return True
    return bool(requested_part(request)) and not _ABOUT_RE.search(request or "")


def requested_part(request: str) -> Optional[Dict[str, object]]:
    """Which part a request asks for, or None for the whole work."""
    m = _PART_RE.search(request or "")
    if not m:
        return None
    if m.group("ordinal"):
        return {"from": "nth", "count": _ORDINALS[m.group("ordinal").lower()],
                "unit": m.group("ounit").lower()}
    raw = (m.group("count") or "one").lower()
    count = int(raw) if raw.isdigit() else _NUMBER_WORDS[raw]
    return {"from": "end" if m.group("end").lower() in ("last", "closing", "final") else "start",
            "count": count, "unit": m.group("unit").lower()}


def excerpt(text: str, part: Optional[Dict[str, object]]) -> Optional[str]:
    """The asked-for part of `text`. None when it can't be cut honestly: the
    text has too few stanzas, or was stored with no stanza breaks at all."""
    if not part:
        return text
    count = int(part["count"])
    if part["unit"] == "line":
        pieces = [ln for ln in text.splitlines() if ln.strip()]
        joiner = "\n"
    else:
        pieces = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        joiner = "\n\n"
    if count < 1 or count > len(pieces) or (len(pieces) == 1 and count == 1 and part["unit"] != "line"):
        return None
    if part["from"] == "nth":
        return pieces[count - 1]
    chosen = pieces[:count] if part["from"] == "start" else pieces[-count:]
    return joiner.join(chosen)


def strip_heading(text: str, name: str) -> str:
    """Drop opening lines that only repeat the title and author; they would
    otherwise be counted as the first stanza. A line ending in verse punctuation
    is the work itself: 'Do not go gentle into that good night,' opens its poem."""
    terms = name_terms(name) | {"by"}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        words = {t[0] for t in _tokens(lines[i])} - _TERM_NOISE
        if (words and not words <= terms) or lines[i].rstrip().endswith((",", ";", ":", "—", ".")):
            break
        i += 1
    return "\n".join(lines[i:]).strip()


def strip_trailer(text: str, name: str) -> str:
    """Drop closing credit lines the two sites share ('Dylan Thomas, "Do not go
    gentle..." from The Poems of Dylan Thomas. Copyright ...')."""
    head = re.split(r"(?i)\s+by\s+", name or "", maxsplit=1)
    author = head[1].strip().lower() if len(head) > 1 else ""
    lines = text.rstrip().splitlines()
    while lines and (not lines[-1].strip()
                     or (author and author in lines[-1].lower())
                     or re.search(r"(?i)copyright|©|all rights reserved|public domain", lines[-1])):
        lines.pop()
    return "\n".join(lines).strip()


class WorkMemory:
    """File-backed store of whole texts she has verified and kept."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, rid: str) -> Path:
        return self.root / f"{rid}.json"

    def _records(self):
        if not self.root.is_dir():
            return
        for path in sorted(self.root.glob("*.json")):
            try:
                yield json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                logger.warning("WorkMemory: unreadable record %s: %s", path.name, e)

    def _save(self, rec: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self._path(rec["id"])
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(target)

    def remember(self, name: str, text: str, sources: List[dict]) -> Optional[str]:
        """Keep a verified text. Learning it again adds its sources and names;
        the words already kept stay unless the new copy holds more."""
        text = (text or "").strip()
        if not name_terms(name) or not text:
            return None
        rid = hashlib.sha256(" ".join(sorted(title_terms(name) or name_terms(name))).encode()).hexdigest()[:16]
        existing: Dict[str, object] = {}
        try:
            existing = json.loads(self._path(rid).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        names = list(existing.get("names", []))
        if name not in names:
            names.append(name)
        known = {s.get("url") for s in existing.get("sources", [])}
        merged = list(existing.get("sources", [])) + [s for s in sources if s.get("url") not in known]
        old_text = str(existing.get("text", ""))
        self._save({
            "id": rid,
            "name": str(existing.get("name") or name),
            "names": names,
            "text": text if len(text) > len(old_text) else old_text,
            "sources": merged,
            "learned_at": existing.get("learned_at") or _now(),
            "times_recited": int(existing.get("times_recited", 0)),
        })
        return rid

    def recall(self, query: str) -> Optional[dict]:
        """The work a name or request refers to. Every term asked for must be
        known, and one must be in the title: 'a poem by Poe' is not The Raven,
        and neither is 'The Raven by Ted Hughes'."""
        wanted = name_terms(work_name(query))
        if not wanted:
            return None
        best = None
        for rec in self._records():
            names = rec.get("names") or [rec.get("name", "")]
            known = set().union(*(name_terms(n) for n in names))
            titles = set().union(*(title_terms(n) for n in names))
            if wanted <= known and wanted & titles:
                if best is None or len(known) < len(best[0]):
                    best = (known, rec)
        return best[1] if best else None

    def mentioned(self, text: str) -> Optional[dict]:
        """A work a question names in passing ('the first line of The Raven').
        Common words make titles, so a one-word title must keep its capital."""
        lowered = (text or "").lower()
        best = None
        for rec in self._records():
            for name in rec.get("names") or [rec.get("name", "")]:
                title = re.sub(r"(?i)^the\s+", "", re.split(r"(?i)\s+by\s+", name, maxsplit=1)[0]).strip()
                terms = title_terms(title)
                if not terms:
                    continue
                pattern = r"\b" + re.escape(title) + r"\b"
                found = re.search(pattern, text or "") if len(terms) == 1 else re.search(pattern, lowered, re.IGNORECASE)
                if found and (best is None or len(title) > best[0]):
                    best = (len(title), rec)
        return best[1] if best else None

    def note_recited(self, rid: str) -> None:
        try:
            rec = json.loads(self._path(rid).read_text(encoding="utf-8"))
            rec["times_recited"] = int(rec.get("times_recited", 0)) + 1
            self._save(rec)
        except (OSError, ValueError) as e:
            logger.warning("WorkMemory: could not count recitation of %s: %s", rid, e)

    def stats(self) -> Dict[str, int]:
        return {"works": sum(1 for _ in self._records())}


def compact_work(rec: dict, max_chars: int = 1500) -> dict:
    """A work as she reads it when she rummages: its words and where they came
    from. Long works are cut for context; palace_recall_work gives the rest."""
    text = rec.get("text", "")
    out = {
        "id": rec.get("id"),
        "name": rec.get("name", ""),
        "text": text[:max_chars],
        "sources": [s.get("url", "") for s in rec.get("sources", [])],
        "learned": (rec.get("learned_at") or "")[:10],
    }
    if len(text) > max_chars:
        out["truncated_from_chars"] = len(text)
    return out

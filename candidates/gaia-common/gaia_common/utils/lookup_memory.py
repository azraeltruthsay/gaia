"""Lookup memory — what GAIA has looked up before, kept with where it came from.

GAIA_Project-0vuz. Every successful web search/fetch is remembered verbatim
(not AAAK-compressed: the compression kept only codes like
'ACT+KIN+ART|actually_king_arthur', so the facts themselves were lost), with
its provenance — query, title, url, domain, trust tier, when it was
retrieved — so she can answer from memory, cite it, and know when to look
again instead of searching every time.

Sibling of the skill fabric: capture happens automatically at the tool layer
(gaia-mcp web_tools), recall happens at grounding time (gaia-core
knowledge_router), and freshness/dispute marks drive revalidation.

Layout: one JSON record per (url, snippet) under
<palace root>/knowledge/lookups/<id>.json. Recall is a keyword-overlap scan
over the records — fine for thousands of records; an index or embeddings
can replace it later without changing the record format.

Earning trust (6afh): a record retrieved for a question is not yet proof it
ANSWERS that question — the 0vuz live test stored three Excalibur pages that
never say where Arthur got the sword, matched them strongly next session, and
skipped the web while Core invented. So after each grounded answer, feedback()
checks it: an answer that hedges, or names things none of its sources say, adds
the question to the records' `unhelpful_queries`; a supported answer adds it to
`confirmed_queries`. Only a record confirmed for (nearly) the same question may
answer from memory alone; anything else rides along with a fresh search.

Lifecycle (6afh): `pinned` records are kept forever (pinned by hand, or
automatically once confirmed PIN_AFTER_CONFIRMATIONS times). sweep() moves
idle, never-used or long-stale records into <root>/archive/, which recall
skips unless include_archive=True — searchable, but not in her way. A record
the web returns again, or one she pins, comes back out of the archive.

Consolidation (prip), run during sleep: consolidate() merges records that
say the same thing (one page's lead seen by many searches, a mirror copying
Wikipedia verbatim) into one, keeping every source url; revalidation_candidates()
picks the stale or disputed records she actually uses for a fresh look-up;
bake_candidates() lists facts proven often enough to be curriculum material.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger("GAIA.LookupMemory")

# Snippets shorter than this carry no answer worth remembering.
MIN_SNIPPET_CHARS = 40
# Page fetches are long; keep the opening, which is what grounding shows.
MAX_SNIPPET_CHARS = 1500
MAX_QUERIES_PER_RECORD = 10

# How long a lookup stays trusted before it should be checked again. Stable
# facts (legends, history, definitions) barely change; anything about "now"
# goes stale within days.
STABLE_DAYS = 365
VOLATILE_DAYS = 2
_VOLATILE_RE = re.compile(
    r"\b(today|tonight|yesterday|tomorrow|this (week|month|year)|latest|current(ly)?|"
    r"right now|recent(ly)?|news|headlines?|price|prices|cost|stock|shares|weather|"
    r"forecast|score|scores|standings|election|polls?|release date|newest|version|"
    r"who is the (ceo|president|prime minister|head|leader))\b",
    re.IGNORECASE,
)

# Lifecycle (days of inactivity — since retrieved, recalled or confirmed —
# before a record moves to the archive). Pinned records never move.
PIN_AFTER_CONFIRMATIONS = 3     # answered well this often: keep forever
ARCHIVE_UNUSED_DAYS = 30        # never recalled, never confirmed
ARCHIVE_IDLE_DAYS = 120         # recalled before, but not lately
ARCHIVE_CONFIRMED_DAYS = 365    # proved useful, then fell out of use
ARCHIVE_STALE_GRACE_DAYS = 30   # stale/disputed this long with no refresh
MAX_ACTIVE_RECORDS = 5000       # beyond this, the least-recently-active go
ARCHIVE_DIRNAME = "archive"

# Ranking bonus for records that answered well before (confirmed or pinned):
# rummaging "Excalibur" should surface the card that says where Arthur got
# it before an article about divers looking for it (prip live). Small
# enough not to outrank a clearly more relevant record.
PROVEN_BONUS = 0.25

# Consolidation (prip).
MERGE_SIMILARITY = 0.85         # same-url snippets this alike are one record
RECHECK_COOLDOWN_DAYS = 7       # don't re-search a record more often than this
BAKE_MIN_CONFIRMATIONS = 2      # confirmed this often → bake/curriculum candidate
# Beside the lookups dir, not in it: everything in it is read as a record.
BAKE_CANDIDATES_FILE = "lookup_bake_candidates.json"

# An answer that says, in effect, "my sources don't say".
_HEDGE_RE = re.compile(
    r"\b(i'?m not (sure|certain)|not (entirely )?(sure|certain|clear)|uncertain|unclear|"
    r"it depends on (which|the) (version|source|telling|account|story)|"
    r"(depending|depends) on (which|the) (version|source|telling|account)|"
    r"(the|my|these) (sources?|search results?|results?) (don'?t|do not|doesn'?t|does not)|"
    r"(couldn'?t|could not|can'?t|cannot|wasn'?t able to) (find|confirm|verify|determine)|"
    r"(don'?t|do not) have (enough|any|specific|reliable) (information|details|data)|"
    r"no (specific |clear |reliable )?(information|details|mention)|"
    r"(isn'?t|is not|aren'?t|are not|wasn'?t|was not) (specified|mentioned|stated|clear)|"
    r"if you (can )?(provide|share|tell me)|i (don'?t|do not) know)\b",
    re.IGNORECASE,
)
# A sourced answer may add this share of named things the sources don't say
# (her own framing: "Arthurian", "Britain") before it counts as unsupported.
MAX_UNSUPPORTED_NAME_SHARE = 0.5

_STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does doing for from get got
had has have he her hers him his how i if in into is it its just me my of on
or our she so than that the their them then there these they this those to
too up us was we were what when where which who whom why will with would you
your yours about tell know please name named called
""".split())

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'\-]*")

# Only function words. Question words and verbs stay: "where … get" and
# "what … name" are different questions about the same sword.
_FUNCTION_WORDS = frozenset("""
a an the and or but of to in on at by for with as is are was were be been
did do does has have had he she it they them his her its their him you your
i me my we our us this that these those can could would should will please
tell know
""".split())


def _words(text: str) -> List[str]:
    # "Arthur's" and "Arthur" are the same name for matching.
    return [re.sub(r"'s$", "", w).strip("'-") for w in _WORD_RE.findall((text or "").lower())]


def content_terms(text: str) -> set:
    """Lowercased content words (len > 2, not stopwords) for matching."""
    return {w for w in _words(text) if len(w) > 2 and w not in _STOPWORDS}


def question_terms(text: str) -> set:
    """Words that identify a question, including what it asks ("where",
    "get"), for matching a new question against ones asked before."""
    return {w for w in _words(text) if len(w) > 1 and w not in _FUNCTION_WORDS}


def question_match(query: str, past_queries: Iterable[str]) -> float:
    """Best Jaccard overlap between `query` and any past query."""
    q = question_terms(query)
    best = 0.0
    for past in past_queries:
        p = question_terms(past)
        if q and p:
            best = max(best, len(q & p) / len(q | p))
    return best


_NAME_RE = re.compile(r"\b[A-Z][a-zA-Z'\-]{2,}")
# Capitalised only because they open a sentence; not names.
_OPENERS = frozenset("""
according well sure okay however also then yes great good hmm after before
once since while although though there here this that these those some many
most legend legends story stories tradition traditionally later eventually
finally first one another other both each yet still instead
""".split())


def answer_names(text: str) -> set:
    """Capitalised words — the names an answer commits to ("Merlin",
    "Lancelot", "Avalon") — minus stopwords and sentence openers, lowercased.
    Names that open a sentence still count ("Merlin gave it to him.")."""
    out = set()
    for w in _NAME_RE.findall(text or ""):
        w = re.sub(r"'s$", "", w.lower()).strip("'-")
        if w not in _STOPWORDS and w not in _FUNCTION_WORDS and w not in _OPENERS:
            out.add(w)
    return out


def assess_answer(question: str, answer: str, source_texts: Iterable[str]) -> Dict[str, object]:
    """Did `answer` come from its sources? Lexical and deliberately simple.

    verdict:
      hedged      — the answer says its sources don't say (or asks for one).
      unsupported — most names the answer adds (beyond the question's) appear
                    in none of the sources: Core filled the gap itself.
      supported   — otherwise.
    `unsupported_names` lists the names no source mentions.
    """
    answer = answer or ""
    if not answer.strip():
        return {"verdict": "hedged", "unsupported_names": [], "reason": "empty answer"}
    if _HEDGE_RE.search(answer):
        return {"verdict": "hedged", "unsupported_names": [],
                "reason": _HEDGE_RE.search(answer).group(0)}
    source_words = set()
    for t in source_texts:
        source_words |= set(_words(t))
    added = answer_names(answer) - set(_words(question))
    missing = sorted(n for n in added if n not in source_words)
    if added and len(missing) / len(added) > MAX_UNSUPPORTED_NAME_SHARE:
        return {"verdict": "unsupported", "unsupported_names": missing,
                "reason": f"{len(missing)}/{len(added)} names not in sources"}
    return {"verdict": "supported", "unsupported_names": missing, "reason": ""}


def is_volatile(query: str) -> bool:
    return bool(_VOLATILE_RE.search(query or ""))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def _parse_ts(ts: Optional[str]) -> float:
    dt = _parse(ts) if ts else None
    return dt.timestamp() if dt else 0.0


def _days_since(ts: Optional[str], now: datetime) -> Optional[float]:
    dt = _parse(ts) if ts else None
    return (now - dt).total_seconds() / 86400 if dt else None


def compact_lookup(rec: dict) -> dict:
    """A record as GAIA reads it when she rummages: the fact, where it came
    from and how much to trust it — not the bookkeeping."""
    out = {
        "id": rec.get("id"),
        "fact": rec.get("snippet", ""),
        "source": rec.get("title") or rec.get("domain", ""),
        "url": rec.get("url", ""),
        "looked_up": (rec.get("retrieved_at") or "")[:10],
        "asked_for": (rec.get("queries") or [""])[-1],
    }
    for flag in ("stale", "archived", "pinned"):
        if rec.get(flag):
            out[flag] = True
    if rec.get("disputed_at"):
        out["disputed"] = True
    if rec.get("times_confirmed"):
        out["answered_well"] = rec["times_confirmed"]
    if rec.get("also_at"):
        out["also_at"] = rec["also_at"]
    return out


class LookupMemory:
    """File-backed store of remembered lookups, with a searchable archive."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.archive_root = self.root / ARCHIVE_DIRNAME

    # ── write ─────────────────────────────────────────────────────────

    def _record_id(self, url: str, snippet: str) -> str:
        norm = " ".join(snippet.split())[:300].lower()
        return hashlib.sha1(f"{url}|{norm}".encode("utf-8")).hexdigest()[:16]

    def _path(self, rid: str) -> Path:
        return self.root / f"{rid}.json"

    def _archived_path(self, rid: str) -> Path:
        return self.archive_root / f"{rid}.json"

    def _find(self, rid: str) -> Optional[Path]:
        """Where record `rid` lives now — active first, then the archive."""
        for path in (self._path(rid), self._archived_path(rid)):
            if path.exists():
                return path
        return None

    def _load(self, path: Path) -> Optional[dict]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("LookupMemory: unreadable record %s: %s", path.name, e)
            return None

    def _save(self, rec: dict, archived: bool = False) -> None:
        """Write `rec` active (default) or into the archive, removing the
        copy on the other side so a record lives in exactly one place."""
        target = self._archived_path(rec["id"]) if archived else self._path(rec["id"])
        other = self._path(rec["id"]) if archived else self._archived_path(rec["id"])
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(target)  # atomic: a concurrent recall never reads half a record
        if other.exists():
            try:
                other.unlink()
            except OSError as e:
                logger.warning("LookupMemory: %s left in both active and archive: %s", rec["id"], e)

    def _records(self, include_archive: bool = False):
        """(path, record, archived) for every readable record."""
        dirs = [(self.root, False)] + ([(self.archive_root, True)] if include_archive else [])
        for d, archived in dirs:
            if not d.exists():
                continue
            for path in d.glob("*.json"):
                rec = self._load(path)
                if isinstance(rec, dict) and rec.get("id"):
                    yield path, rec, archived

    def remember(self, query: str, results: Iterable[dict], kind: str = "search",
                 now: Optional[datetime] = None) -> List[str]:
        """Remember each usable result of a lookup. Returns the record ids,
        and tags each result dict with its `memory_id` so the caller (and
        whoever it hands the results to) can give feedback on it later.

        A result seen again (same url + snippet) is refreshed — new
        retrieved_at, query added, any dispute cleared, since the web just
        corroborated it — rather than duplicated. An archived one comes back.
        """
        now = now or _now()
        days = VOLATILE_DAYS if is_volatile(query) else STABLE_DAYS
        ids: List[str] = []
        for r in results or []:
            if not isinstance(r, dict):
                continue
            snippet = (r.get("snippet") or r.get("content") or "").strip()[:MAX_SNIPPET_CHARS]
            if len(snippet) < MIN_SNIPPET_CHARS:
                continue
            url = (r.get("url") or "").strip()
            rid = self._record_id(url, snippet)
            path = self._find(rid)
            rec = self._load(path) if path else None
            if rec is None:
                rec = {
                    "id": rid,
                    "kind": kind,
                    "queries": [],
                    "title": (r.get("title") or "").strip(),
                    "url": url,
                    "domain": (r.get("domain") or "").strip(),
                    "trust_tier": r.get("trust_tier") or "unverified",
                    "snippet": snippet,
                    "first_retrieved_at": _iso(now),
                    "times_recalled": 0,
                    "last_recalled_at": None,
                }
            if query and query not in rec["queries"]:
                rec["queries"] = (rec["queries"] + [query])[-MAX_QUERIES_PER_RECORD:]
            rec["retrieved_at"] = _iso(now)
            rec["revalidate_after"] = _iso(now + timedelta(days=days))
            rec.pop("disputed_at", None)
            rec.pop("archived_at", None)       # back in everyday recall
            rec.pop("archived_reason", None)
            try:
                self._save(rec)
                ids.append(rid)
                r["memory_id"] = rid
            except OSError as e:
                logger.warning("LookupMemory: could not save %s: %s", rid, e)
        if ids:
            logger.info("LookupMemory: remembered %d result(s) for %r (%s, fresh for %dd)",
                        len(ids), (query or "")[:80], kind, days)
        return ids

    def _update(self, ids: Iterable[str], change, what: str) -> int:
        """Apply `change(rec)` to each record (active or archived; it stays
        where it is). Returns how many were changed."""
        n = 0
        for rid in ids:
            path = self._find(rid)
            rec = self._load(path) if path else None
            if rec is None:
                continue
            change(rec)
            try:
                self._save(rec, archived=path.parent == self.archive_root)
                n += 1
            except OSError as e:
                logger.warning("LookupMemory: could not %s %s: %s", what, rid, e)
        return n

    def mark_disputed(self, ids: Iterable[str], now: Optional[datetime] = None) -> int:
        """Flag records as needing revalidation (e.g. the user pushed back on
        an answer they supported). A disputed record is still recalled, but
        marked stale so it never stands in for a fresh lookup; the next
        lookup that returns it again clears the flag. Its confirmations go
        too: an answer the user disputes didn't prove the record useful."""
        stamp = _iso(now or _now())

        def change(rec):
            rec["disputed_at"] = stamp
            rec["confirmed_queries"] = []

        n = self._update(ids, change, "mark disputed")
        if n:
            logger.info("LookupMemory: %d record(s) marked for revalidation", n)
        return n

    def pin(self, ids: Iterable[str], pinned: bool = True) -> int:
        """Keep records forever (never archived) — or release them. Pinning
        an archived record brings it back."""
        ids = list(ids)

        def change(rec):
            rec["pinned"] = bool(pinned)

        n = self._update(ids, change, "pin")
        if pinned:
            for rid in ids:
                self.restore([rid])
        if n:
            logger.info("LookupMemory: %d record(s) %s", n, "pinned" if pinned else "unpinned")
        return n

    def restore(self, ids: Iterable[str]) -> int:
        """Bring archived records back into everyday recall."""
        n = 0
        for rid in ids:
            path = self._archived_path(rid)
            rec = self._load(path) if path.exists() else None
            if rec is None:
                continue
            rec.pop("archived_at", None)
            rec.pop("archived_reason", None)
            try:
                self._save(rec)
                n += 1
            except OSError as e:
                logger.warning("LookupMemory: could not restore %s: %s", rid, e)
        if n:
            logger.info("LookupMemory: %d record(s) restored from the archive", n)
        return n

    # ── feedback: did the memory actually answer? ────────────────────

    def _resolve_sources(self, sources: Iterable[dict]) -> Dict[str, dict]:
        """Map each source the answer was grounded on to its record: by
        `memory_id` when the caller kept it, else by url (+ snippet opening,
        when a url has several records)."""
        by_url: Dict[str, List[dict]] = {}
        wanted: Dict[str, dict] = {}
        need_url = []
        for src in sources or []:
            if not isinstance(src, dict):
                continue
            rid = src.get("memory_id")
            if rid and self._find(rid):
                wanted[rid] = src
            elif src.get("url"):
                need_url.append(src)
        if need_url:
            for _path, rec, _a in self._records(include_archive=True):
                by_url.setdefault(rec.get("url", ""), []).append(rec)
            for src in need_url:
                recs = by_url.get(src["url"], [])
                text = " ".join(str(src.get("snippet") or "").split()).lower()
                exact = [r for r in recs if " ".join(r.get("snippet", "").split())[:60].lower() in text]
                for r in exact or recs:
                    wanted[r["id"]] = src
        return wanted

    def feedback(self, question: str, answer: str, sources: Iterable[dict],
                 now: Optional[datetime] = None) -> Dict[str, object]:
        """Record whether the lookups behind an answer actually answered
        `question` (see assess_answer). Supported: the question joins each
        record's `confirmed_queries` (and leaves `unhelpful_queries`); a
        record confirmed PIN_AFTER_CONFIRMATIONS times is pinned. Hedged or
        unsupported: the question joins `unhelpful_queries`, so the next time
        it's asked memory rides along with a fresh search instead of
        standing in for one."""
        now = now or _now()
        sources = [s for s in (sources or []) if isinstance(s, dict)]
        resolved = self._resolve_sources(sources)
        # Judge against what the caller showed the model and the records'
        # own text (a caller may pass only ids, or a truncated snippet).
        texts = [str(s.get("snippet") or "") for s in sources]
        rec_words: Dict[str, set] = {}
        for rid, src in resolved.items():
            path = self._find(rid)
            rec = self._load(path) if path else None
            if rec:
                text = f"{rec.get('title', '')} {rec.get('snippet', '')}"
                texts.append(text)
                rec_words[rid] = set(_words(f"{text} {src.get('snippet') or ''}"))
        verdict = assess_answer(question, answer, texts)
        if not resolved or not (question or "").strip():
            return {**verdict, "records": 0}
        good = verdict["verdict"] == "supported"
        stamp = _iso(now)
        targets = list(resolved)
        if good:
            # Confirm only the records that say what the answer says (the
            # Quizlet card with "the Lake", not the Excalibur page lead shown
            # beside it); the rest were neither help nor harm. An answer that
            # names nothing new can't tell them apart — all are confirmed.
            names = answer_names(answer) - set(_words(question))
            contributed = [rid for rid in targets if names & rec_words.get(rid, set())]
            if names and contributed:
                targets = contributed

        def change(rec):
            confirmed = [q for q in rec.get("confirmed_queries", []) if q != question]
            unhelpful = [q for q in rec.get("unhelpful_queries", []) if q != question]
            if good:
                rec["confirmed_queries"] = (confirmed + [question])[-MAX_QUERIES_PER_RECORD:]
                rec["unhelpful_queries"] = unhelpful
                rec["times_confirmed"] = rec.get("times_confirmed", 0) + 1
                rec["last_confirmed_at"] = stamp
                if rec["times_confirmed"] >= PIN_AFTER_CONFIRMATIONS and not rec.get("pinned"):
                    rec["pinned"] = True
                    rec["pinned_reason"] = f"confirmed {rec['times_confirmed']} times"
                    logger.info("LookupMemory: %s pinned (%s)", rec["id"], rec["pinned_reason"])
            else:
                rec["confirmed_queries"] = confirmed
                rec["unhelpful_queries"] = (unhelpful + [question])[-MAX_QUERIES_PER_RECORD:]

        n = self._update(targets, change, "record feedback for")
        if good:
            self.restore(targets)  # it proved useful; back into everyday recall
        logger.info("LookupMemory: answer to %r was %s (%s); %d record(s) %s",
                    question[:80], verdict["verdict"], verdict["reason"] or "names found in sources",
                    n, "confirmed" if good else "marked unhelpful for it")
        return {**verdict, "records": n}

    # ── read ──────────────────────────────────────────────────────────

    def recall(self, query: str, top_k: int = 3, min_score: float = 0.34,
               now: Optional[datetime] = None, touch: bool = True,
               include_archive: bool = False) -> List[dict]:
        """Remembered lookups relevant to `query`, best first.

        The record's own title + snippet must cover at least `min_score` of
        the query's content words. Its past queries only add a half-weight
        bonus for words the text lacks: every result of a search inherits
        that search's query, so counting queries fully let an off-topic
        result ("Onimusha: Way of the Sword" for "King Arthur sword") look
        relevant. Each hit carries `score`, `question_match` (how close
        `query` is to a question this record was retrieved for),
        `confirmed_match` / `unhelpful_match` (… to one it answered well /
        failed to answer — 6afh), `stale` (past revalidate_after, or
        disputed) and `archived`. The archive is only searched on request.
        """
        now = now or _now()
        terms = content_terms(query)
        if not terms:
            return []
        scored = []
        for path, rec, archived in self._records(include_archive):
            text_terms = content_terms(f"{rec.get('title', '')} {rec.get('snippet', '')}")
            text_hit = terms & text_terms
            text_score = len(text_hit) / len(terms)
            if text_score < min_score:
                continue
            query_only = (terms & content_terms(" ".join(rec.get("queries", [])))) - text_hit
            score = min(1.0, text_score + 0.5 * len(query_only) / len(terms))
            qmatch = question_match(query, rec.get("queries", []))
            scored.append((score, qmatch, rec, archived))
        # Rank by relevance + question match: many records mention the
        # subject equally ("Excalibur is the sword of King Arthur" scores 1.0
        # for any Excalibur question); the one looked up for THIS question
        # (the Lady of the Lake, for "where did he get it") should lead, and
        # one that answered well before gets a small edge (PROVEN_BONUS).
        scored.sort(key=lambda t: (
            t[0] + t[1] + (PROVEN_BONUS if t[2].get("times_confirmed") or t[2].get("pinned") else 0.0),
            t[2].get("retrieved_at", "")), reverse=True)
        hits = []
        for score, qmatch, rec, archived in scored[:top_k]:
            revalidate = _parse(rec.get("revalidate_after", ""))
            stale = bool(rec.get("disputed_at")) or (revalidate is not None and now > revalidate)
            hits.append({**rec, "score": round(score, 3), "stale": stale,
                         "question_match": round(qmatch, 3),
                         "confirmed_match": round(question_match(query, rec.get("confirmed_queries", [])), 3),
                         "unhelpful_match": round(question_match(query, rec.get("unhelpful_queries", [])), 3),
                         "archived": archived})
            if touch:
                rec["times_recalled"] = rec.get("times_recalled", 0) + 1
                rec["last_recalled_at"] = _iso(now)
                try:
                    # Stats only; an archive hit stays archived until it
                    # proves useful (feedback) or the web returns it.
                    self._save(rec, archived=archived)
                except OSError:
                    # Usage stats are best-effort; recall still succeeds.
                    logger.debug("LookupMemory: could not update recall stats for %s", rec.get("id"))
        if hits:
            logger.info("LookupMemory: recalled %d for %r (best %.2f%s%s)", len(hits), query[:80],
                        hits[0]["score"], ", stale" if hits[0]["stale"] else "",
                        ", from archive" if hits[0]["archived"] else "")
        return hits

    def recent(self, limit: int = 10, include_archive: bool = False) -> List[dict]:
        """Browse: the most recently used lookups (retrieved, recalled or
        confirmed), for rummaging without a particular query in mind."""
        recs = [{**rec, "archived": archived}
                for _p, rec, archived in self._records(include_archive)]
        recs.sort(key=lambda r: max(r.get(k) or "" for k in (
            "retrieved_at", "last_recalled_at", "last_confirmed_at")), reverse=True)
        return recs[:limit]

    # ── lifecycle ─────────────────────────────────────────────────────

    def _archive_reason(self, rec: dict, now: datetime) -> Optional[str]:
        """Why `rec` should leave everyday recall, or None to keep it."""
        if rec.get("pinned"):
            return None
        activity = [d for d in (
            _days_since(rec.get("retrieved_at"), now),
            _days_since(rec.get("last_recalled_at"), now),
            _days_since(rec.get("last_confirmed_at"), now),
        ) if d is not None]
        idle = min(activity) if activity else None
        if idle is None:
            return None  # no dates to judge by; leave it alone
        stale_for = max(filter(lambda d: d is not None, (
            _days_since(rec.get("disputed_at"), now),
            _days_since(rec.get("revalidate_after"), now),
        )), default=None)
        if stale_for is not None and stale_for > ARCHIVE_STALE_GRACE_DAYS:
            return f"stale for {stale_for:.0f}d"
        if rec.get("times_confirmed"):
            limit = ARCHIVE_CONFIRMED_DAYS
        elif rec.get("times_recalled"):
            limit = ARCHIVE_IDLE_DAYS
        else:
            limit = ARCHIVE_UNUSED_DAYS
        if idle > limit:
            return f"idle {idle:.0f}d (limit {limit}d)"
        return None

    def sweep(self, now: Optional[datetime] = None, dry_run: bool = False,
              max_active: int = MAX_ACTIVE_RECORDS) -> Dict[str, object]:
        """Move records that are idle, never used or long stale into the
        archive; then, if more than `max_active` remain, the least-recently
        active unpinned ones too. Pinned records are never touched. Returns
        what moved (or would, with dry_run)."""
        now = now or _now()
        moved: List[Dict[str, str]] = []
        kept = []
        pinned = 0
        for _path, rec, _a in self._records():
            pinned += bool(rec.get("pinned"))
            reason = self._archive_reason(rec, now)
            if reason:
                moved.append({"id": rec["id"], "title": rec.get("title", ""), "reason": reason})
                if not dry_run:
                    self._archive(rec, reason, now)
            else:
                kept.append(rec)
        overflow = [r for r in kept if not r.get("pinned")]
        excess = len(kept) - max_active
        if excess > 0:
            overflow.sort(key=lambda r: max(r.get(k) or "" for k in (
                "retrieved_at", "last_recalled_at", "last_confirmed_at")))
            for rec in overflow[:excess]:
                moved.append({"id": rec["id"], "title": rec.get("title", ""), "reason": "over capacity"})
                if not dry_run:
                    self._archive(rec, "over capacity", now)
        active = len(kept) - max(0, min(excess, len(overflow)))
        logger.info("LookupMemory: sweep %s %d record(s) to the archive; %d active (%d pinned)",
                    "would move" if dry_run else "moved", len(moved), active, pinned)
        return {"archived": moved, "active": active, "pinned": pinned, "dry_run": dry_run}

    def _archive(self, rec: dict, reason: str, now: datetime) -> None:
        rec["archived_at"] = _iso(now)
        rec["archived_reason"] = reason
        try:
            self._save(rec, archived=True)
        except OSError as e:
            logger.warning("LookupMemory: could not archive %s: %s", rec.get("id"), e)

    # ── consolidation (prip) ──────────────────────────────────────────

    @staticmethod
    def _norm_text(rec: dict) -> str:
        return " ".join((rec.get("snippet") or "").split()).lower()

    @staticmethod
    def _rank(rec: dict) -> tuple:
        """Which of two duplicates survives: pinned, then proven, then used,
        then the older (it has the longer history)."""
        return (bool(rec.get("pinned")), rec.get("times_confirmed", 0),
                rec.get("times_recalled", 0), -_parse_ts(rec.get("first_retrieved_at")))

    def _merge_into(self, keep: dict, dup: dict) -> None:
        """Fold `dup`'s history into `keep` (queries, verdicts, usage, sources)."""
        for key in ("queries", "confirmed_queries", "unhelpful_queries"):
            merged = list(keep.get(key, []))
            merged += [q for q in dup.get(key, []) if q not in merged]
            if merged:
                keep[key] = merged[-MAX_QUERIES_PER_RECORD:]
        for key in ("times_recalled", "times_confirmed"):
            keep[key] = keep.get(key, 0) + dup.get(key, 0)
        for key in ("retrieved_at", "last_recalled_at", "last_confirmed_at", "revalidate_after"):
            keep[key] = max(filter(None, (keep.get(key), dup.get(key))), default=keep.get(key))
        keep["first_retrieved_at"] = min(filter(None, (keep.get("first_retrieved_at"),
                                                       dup.get("first_retrieved_at"))), default=None)
        if dup.get("pinned"):
            keep["pinned"] = True
        # A newer refresh of either clears an older dispute (as remember() does).
        if keep.get("disputed_at") and keep["disputed_at"] < keep.get("retrieved_at", ""):
            keep.pop("disputed_at")
        if dup.get("url") and dup["url"] != keep.get("url"):
            also = keep.setdefault("also_at", [])
            for u in [dup["url"]] + dup.get("also_at", []):
                if u not in also and u != keep.get("url"):
                    also.append(u)
        keep.setdefault("merged_ids", []).append(dup["id"])

    def consolidate(self, now: Optional[datetime] = None, dry_run: bool = False) -> Dict[str, object]:
        """Merge active records that say the same thing:
          - the same text at different urls (mirrors, syndication), and
          - near-identical text at the same url (a page's lead as returned
            by different searches; a search snippet vs the fetched opening).
        The survivor keeps every url (`also_at`) and the combined history;
        each duplicate is archived with reason "merged into <id>", not
        deleted, so the trail stays inspectable."""
        now = now or _now()
        recs = [rec for _p, rec, _a in self._records()]
        # Union-find, so duplication is transitive: a mirror of A and a
        # same-page variant of A all end up in A's group.
        parent = {r["id"]: r["id"] for r in recs}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        by_text: Dict[str, str] = {}
        by_url: Dict[str, List[dict]] = {}
        for rec in recs:
            text = self._norm_text(rec)
            if text in by_text:
                parent[find(rec["id"])] = find(by_text[text])
            else:
                by_text[text] = rec["id"]
            if rec.get("url"):
                by_url.setdefault(rec["url"], []).append(rec)
        for same_url in by_url.values():
            # Pages yield a handful of records each; pairwise is cheap here.
            terms = [(r["id"], content_terms(self._norm_text(r))) for r in same_url]
            for i, (rid_a, ta) in enumerate(terms):
                for rid_b, tb in terms[i + 1:]:
                    union = ta | tb
                    if union and len(ta & tb) / len(union) >= MERGE_SIMILARITY:
                        parent[find(rid_b)] = find(rid_a)
        by_root: Dict[str, List[dict]] = {}
        for rec in recs:
            by_root.setdefault(find(rec["id"]), []).append(rec)
        groups = [g for g in by_root.values() if len(g) > 1]
        merged = []
        for group in groups:
            group.sort(key=self._rank, reverse=True)
            keep, dups = group[0], group[1:]
            for dup in dups:
                merged.append({"id": dup["id"], "into": keep["id"], "title": dup.get("title", "")})
                if dry_run:
                    continue
                self._merge_into(keep, dup)
                self._archive(dup, f"merged into {keep['id']}", now)
            if not dry_run:
                try:
                    self._save(keep)
                except OSError as e:
                    logger.warning("LookupMemory: could not save merged %s: %s", keep["id"], e)
        if merged:
            logger.info("LookupMemory: consolidation %s %d duplicate record(s) into %d",
                        "would merge" if dry_run else "merged", len(merged), len(groups))
        return {"merged": merged, "dry_run": dry_run}

    def revalidation_candidates(self, limit: int = 5, now: Optional[datetime] = None) -> List[dict]:
        """Stale or disputed records worth a fresh look-up: ones she uses
        (pinned, confirmed or recalled), not re-checked in the last
        RECHECK_COOLDOWN_DAYS, most used first. Each carries the `query` to
        search — the latest question it was retrieved or confirmed for."""
        now = now or _now()
        out = []
        for _p, rec, _a in self._records():
            revalidate = _parse(rec.get("revalidate_after", ""))
            stale = bool(rec.get("disputed_at")) or (revalidate is not None and now > revalidate)
            used = rec.get("pinned") or rec.get("times_confirmed") or rec.get("times_recalled")
            since = _days_since(rec.get("recheck_attempted_at"), now)
            if not stale or not used or (since is not None and since < RECHECK_COOLDOWN_DAYS):
                continue
            query = ((rec.get("confirmed_queries") or [None])[-1]
                     or (rec.get("queries") or [None])[-1] or rec.get("title"))
            if query:
                out.append({"id": rec["id"], "query": query, "title": rec.get("title", ""),
                            "_rank": self._rank(rec)})
        out.sort(key=lambda c: c.pop("_rank"), reverse=True)
        return out[:limit]

    def note_recheck(self, ids: Iterable[str], now: Optional[datetime] = None) -> int:
        """Record a re-check attempt (whatever came of it) so the cooldown
        holds; a record the search returned again was refreshed by remember()."""
        stamp = _iso(now or _now())

        def change(rec):
            rec["recheck_attempted_at"] = stamp

        return self._update(ids, change, "note recheck for")

    def bake_candidates(self, write: bool = True) -> List[dict]:
        """Facts proven often enough to be worth learning outright: pinned or
        confirmed BAKE_MIN_CONFIRMATIONS+ times, not disputed. Each lists the
        questions it answered, the text and its sources — material for a
        curriculum, which stays a human/training decision (nothing trains
        from this automatically). Written beside the lookups directory
        (<palace>/knowledge/lookup_bake_candidates.json)."""
        out = []
        for _p, rec, _a in self._records():
            if rec.get("disputed_at"):
                continue
            if not (rec.get("pinned") or rec.get("times_confirmed", 0) >= BAKE_MIN_CONFIRMATIONS):
                continue
            out.append({"id": rec["id"], "questions": rec.get("confirmed_queries", []),
                        "fact": rec.get("snippet", ""), "title": rec.get("title", ""),
                        "sources": [rec.get("url", "")] + rec.get("also_at", []),
                        "times_confirmed": rec.get("times_confirmed", 0),
                        "times_recalled": rec.get("times_recalled", 0),
                        "retrieved_at": rec.get("retrieved_at")})
        out.sort(key=lambda c: (c["times_confirmed"], c["times_recalled"]), reverse=True)
        if write:
            try:
                path = self.root.parent / BAKE_CANDIDATES_FILE
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
                tmp.replace(path)
            except OSError as e:
                logger.warning("LookupMemory: could not write bake candidates: %s", e)
        if out:
            logger.info("LookupMemory: %d bake candidate(s)", len(out))
        return out

    def stats(self) -> Dict[str, int]:
        n = disputed = pinned = confirmed = archived = 0
        for _path, rec, is_archived in self._records(include_archive=True):
            if is_archived:
                archived += 1
                continue
            n += 1
            disputed += bool(rec.get("disputed_at"))
            pinned += bool(rec.get("pinned"))
            confirmed += bool(rec.get("times_confirmed"))
        return {"lookups": n, "disputed": disputed, "pinned": pinned,
                "confirmed": confirmed, "archived": archived}

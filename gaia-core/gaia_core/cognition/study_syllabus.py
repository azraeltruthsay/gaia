"""Studying a subject: she reads a syllabus of pages, writes notes in her own
words, and turns her notes into questions she can train on (2026-11-19:
Professional Engineering, so she can talk shop with Andy).

Her model proposes, code checks (as in told_fact_checker):
  - A note line is kept only if every number in it is on the page and most of
    its words are (a sentence the page doesn't back is dropped, not trusted).
  - A question's answer is kept only if her checked notes back it the same way.

Layout under /knowledge/studies/<slug>/:
  sources/<topic>.txt   the page as fetched (with its url)
  notes/<topic>.md      her checked notes
  pairs.jsonl           question/answer pairs from the notes
  report.json           per topic: lines kept/dropped, pairs kept/dropped

Run inside gaia-core:
  python -m gaia_core.cognition.study_syllabus /knowledge/curricula/.../syllabus.yaml
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import requests

MCP = os.environ.get("MCP_ENDPOINT", "http://gaia-mcp:8765/jsonrpc")
CORE = os.environ.get("STUDY_MODEL_URL", "http://localhost:8092/v1/chat/completions")
PAGE_CHARS = 9000
MIN_BACKED = 0.6

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
_WORD_RE = re.compile(r"[a-z][a-z'-]{4,}")
_STOP = set("""about above after again against because before being below between
could doing during further having other their there these those through under
until where which while would should every always never often usually which
whose within without among along""".split())

NOTES_SYSTEM = (
    "You are GAIA, studying {subject} so you can talk with engineers about their "
    "work. Read the page and write study notes in your own words: what {topic} "
    "is, the key terms with their meanings, any formulas or numbers that matter, "
    "and why it matters to a practicing engineer. Use only what the page says. "
    "Plain bullet points, at most 15.")

PAIRS_SYSTEM = (
    "You are GAIA. From your study notes on {topic}, write {n} questions an "
    "engineer might ask you, each with a short answer in your own voice, using "
    "only what the notes say. Format each as:\nQ: ...\nA: ...")


def fetch(url: str) -> Tuple[str, str]:
    """(title, text) of a page, through her own web_fetch tool."""
    r = requests.post(MCP, json={"jsonrpc": "2.0", "method": "web_fetch",
                                 "params": {"url": url}, "id": 1}, timeout=90).json()
    res = r.get("result") or {}
    if not res.get("ok"):
        raise RuntimeError(str(res.get("error") or r.get("error"))[:200])
    return res.get("title", ""), res.get("content", "")


def search_url(query: str) -> Optional[str]:
    r = requests.post(MCP, json={"jsonrpc": "2.0", "method": "web_search",
                                 "params": {"query": query}, "id": 1}, timeout=60).json()
    for hit in (r.get("result") or {}).get("results", []):
        if hit.get("trust_tier") == "trusted":
            return hit.get("url")
    return None


def http_generator(url: str = CORE, max_tokens: int = 700,
                   temperature: float = 0.3) -> Callable[[List[dict]], str]:
    def _gen(messages: List[dict]) -> str:
        r = requests.post(url, json={"messages": messages, "max_tokens": max_tokens,
                                     "temperature": temperature}, timeout=900).json()
        text = r["choices"][0]["message"]["content"] or ""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return _gen


def _words(text: str) -> set:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOP}


def backed(line: str, source: str, source_words: Optional[set] = None) -> bool:
    """Whether the source backs a line: its numbers all appear there, and most
    of its words do (stem-ish: a word counts if the source has its first 5 letters)."""
    src = source.lower()
    for n in _NUM_RE.findall(line):
        if n not in source:
            return False
    words = _words(line)
    if not words:
        return True
    stems = source_words if source_words is not None else {w[:5] for w in _words(src)}
    hit = sum(1 for w in words if w[:5] in stems)
    return hit / len(words) >= MIN_BACKED


def check_notes(notes: str, page: str) -> Tuple[List[str], List[str]]:
    stems = {w[:5] for w in _words(page)}
    kept, dropped = [], []
    for line in notes.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#") or (len(_words(s)) < 2 and not _NUM_RE.search(s)):
            kept.append(s)
        elif backed(s, page, stems):
            kept.append(s)
        else:
            dropped.append(s)
    return kept, dropped


_QA_RE = re.compile(r"Q:\s*(.+?)\s*\n\s*A:\s*(.+?)(?=\n\s*Q:|\Z)", re.S)


def make_pairs(topic: str, notes: str, generate: Callable[[List[dict]], str],
               n: int = 6) -> Tuple[List[Dict], List[Dict]]:
    out = generate([{"role": "system", "content": PAIRS_SYSTEM.format(topic=topic, n=n)},
                    {"role": "user", "content": notes}])
    stems = {w[:5] for w in _words(notes)}
    kept, dropped = [], []
    for q, a in _QA_RE.findall(out or ""):
        q, a = q.strip(), " ".join(a.split())
        if not q.endswith("?") or len(a) < 20:
            continue
        pair = {"instruction": q, "output": a}
        (kept if backed(a, notes, stems) else dropped).append(pair)
    return kept, dropped


def section(page: str, url: str) -> str:
    """The page from the heading its #anchor names (a long page's later section
    would otherwise fall past PAGE_CHARS), or the whole page."""
    if "#" not in url:
        return page
    from urllib.parse import unquote
    heading = unquote(url.split("#", 1)[1]).replace("_", " ").strip()
    m = re.search(rf"^\s*{re.escape(heading)}\s*$", page, re.M | re.I)
    return page[m.start():].lstrip() if m else page


def _cfr():
    """Her CFR store, with her CPU mind writing the summaries (its configured
    endpoint is Prime, which is unloaded while she rests)."""
    from gaia_common.utils.cfr_manager import CFRManager
    return CFRManager(vllm_endpoint=os.environ.get("STUDY_CFR_URL", "http://localhost:8092"))


def read_parts(body: str, doc_id: str, path: Path, most: int, cfr=None) -> List[Tuple[str, str]]:
    """(part, what came before) pairs to read. A page that fits is one part. A
    long one that the syllabus wants read whole (`parts`) goes through CFR: her
    store splits and summarizes it, and each section is read with a rolling
    summary of the sections before it, so a rule is read knowing the
    definitions above it. The document stays in her CFR store afterwards, for
    cfr_focus/expand when someone asks her about it."""
    if most <= 1 or len(body) <= PAGE_CHARS:
        return [(body[:PAGE_CHARS], "")]
    cfr = cfr or _cfr()
    path.write_text(body, encoding="utf-8")
    res = cfr.ingest(str(path), doc_id=doc_id)
    if not res.get("ok"):
        raise RuntimeError(f"CFR ingest: {res.get('error')}")
    out = []
    for i in range(min(res["section_count"], most)):
        text = cfr.focus(doc_id, i)["focused_section"]["full_text"]
        before = cfr.rolling_context(doc_id, i).get("rolling_summary", "") if i else ""
        out.append((text, before))
    return out


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60]


def study(syllabus: Dict, out_dir: Path, generate: Callable[[List[dict]], str],
          redo: bool = False, log=print, max_pages: int = 0, cfr=None) -> Dict:
    """max_pages: pages to read this run (0 = all). Her web_fetch allows 200 an
    hour, shared with her live lookups, so a run leaves her room (2026-11-19:
    one run used it all)."""
    subject = syllabus["subject"]
    for sub in ("sources", "notes"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() and not redo else {}
    pairs_path = out_dir / "pairs.jsonl"
    read = 0
    for unit, topics in syllabus["units"].items():
        for item in topics:
            topic = item["topic"]
            key = _slug(topic)
            if key in report and report[key].get("pairs_kept") is not None:
                continue
            if max_pages and read >= max_pages:
                log(f"read {read} pages this run; the rest wait for the next")
                return report
            read += 1
            t0 = time.time()
            try:
                url = item.get("url") or search_url(topic)
                if not url:
                    raise RuntimeError("no trusted page found")
                title, page = fetch(url)
                if len(page) < 500:
                    raise RuntimeError(f"page too short ({len(page)} chars)")
                if item.get("drop"):
                    page = re.sub(item["drop"], "", page)
                parts = read_parts(section(page, url), f"study_{key}",
                                   out_dir / "sources" / f"{key}_full.txt",
                                   item.get("parts", 1), cfr)
                totals = {"notes_kept": 0, "notes_dropped": [], "pairs_kept": 0, "pairs_dropped": []}
                for n, (part, before) in enumerate(parts, 1):
                    name = topic if len(parts) == 1 else f"{topic} (part {n} of {len(parts)})"
                    pk = key if len(parts) == 1 else f"{key}_p{n}"
                    (out_dir / "sources" / f"{pk}.txt").write_text(f"{url}\n{title}\n\n{part}")
                    notes = generate([
                        {"role": "system", "content": NOTES_SYSTEM.format(subject=subject, topic=name)},
                        {"role": "user", "content": f"Page: {title}\n\n" + (
                            f"What came before in this document (your summary):\n{before}\n\n"
                            f"This part:\n" if before else "") + part}])
                    kept, dropped = check_notes(notes, part)
                    text = "\n".join(kept)
                    (out_dir / "notes" / f"{pk}.md").write_text(
                        f"# {name}\n\nSource: {url}\nUnit: {unit}\n\n{text}\n")
                    pairs, bad = make_pairs(topic, text, generate) if len(kept) >= 3 else ([], [])
                    with pairs_path.open("a", encoding="utf-8") as f:
                        for p in pairs:
                            row = {**p, "category": syllabus.get("category", syllabus["slug"]),
                                   "group": unit, "topic": topic, "source": url}
                            if item.get("suggested_by"):
                                row["suggested_by"] = item["suggested_by"]
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    totals["notes_kept"] += len(kept)
                    totals["notes_dropped"] += dropped
                    totals["pairs_kept"] += len(pairs)
                    totals["pairs_dropped"] += bad
                kept, dropped, pairs, bad = (totals["notes_kept"], totals["notes_dropped"],
                                             totals["pairs_kept"], totals["pairs_dropped"])
                report[key] = {"topic": topic, "url": url, "parts": len(parts), **totals,
                               "seconds": round(time.time() - t0)}
                log(f"{topic}: notes {kept} kept / {len(dropped)} dropped, "
                    f"pairs {pairs} kept / {len(bad)} dropped ({time.time() - t0:.0f}s)")
            except Exception as e:  # one bad page doesn't end the study
                if "Rate limit" in str(e):
                    log(f"{topic}: her web reading limit is reached; stopping")
                    return report
                report[key] = {"topic": topic, "error": str(e)[:200]}
                log(f"{topic}: FAILED {e}")
            report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return report


def main(argv: List[str]) -> None:
    import yaml
    syllabus = yaml.safe_load(Path(argv[1]).read_text())
    out = Path(os.environ.get("STUDY_DIR", "/knowledge/studies")) / syllabus["slug"]
    study(syllabus, out, http_generator(), redo="--redo" in argv,
          log=lambda m: print(m, flush=True),
          max_pages=int(os.environ.get("STUDY_MAX_PAGES", "50")))


if __name__ == "__main__":
    main(sys.argv)

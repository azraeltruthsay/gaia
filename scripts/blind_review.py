#!/usr/bin/env python3
"""Review her new study pairs without a person (dcrs/p3ta, 2026-11-20).

Round 3's 375 pairs were reviewed by hand (Claude): 49 dropped, 18 edited.
For the loop to run without anyone, each new pair is checked blind against
the page she studied it from (the blind-fact-checking pattern: never ask a
model whether a claim is right, it says yes):

  1. the answer is hidden; a model answers the question from the page's
     relevant section alone, or says NOT IN TEXT;
  2. code compares: the pair's numbers must be on the page, and the page's
     answer must share most of the pair's key words;
  3. reference-list and history trivia ("who wrote...", "(1997)") is dropped.

Kept pairs go in review.json's keep list, drops with the reason. What this
can't catch: a page that is itself wrong, or a fact right on the page and
wrong in practice (round 3's AASHTO 1.5 + 1.0 s split). Those surface later as
her wrong answers, which the look back corrects and files (p3ta).

Usage (gaia-study, GAIA in meditation):
  python scripts/blind_review.py knowledge/curricula/<subject> /knowledge/studies/<slug> --prime-gpu
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from augment_curriculum import _slug, stems  # noqa: E402

_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_TRIVIA = re.compile(r"\b(?:who (?:wrote|developed|derived|proposed)|references?\b|key references|"
                     r"paper|bibliograph|\((?:19|20)\d\d\))", re.IGNORECASE)
ASK = ("Answer the question using only the text below, in one or two sentences. "
       "If the text does not contain the answer, reply exactly: NOT IN TEXT")


def _excerpt(text: str, question: str, chars: int = 5000) -> str:
    try:
        from gaia_common.utils.doc_sections import excerpt_for
        return excerpt_for(text, question, chars)
    except Exception:
        return text[:chars]


def check(pair: dict, page: str, gen) -> tuple:
    """(keep, reason)."""
    q, a = pair["instruction"], pair["output"]
    if _TRIVIA.search(q):
        return False, "blind check: reference/history trivia"
    if not page:
        return False, "blind check: no source page saved"
    missing = [n for n in _NUM.findall(a) if n not in page]
    if missing:
        return False, f"blind check: numbers not on the page ({', '.join(missing[:3])})"
    said = (gen([{"role": "system", "content": ASK},
                 {"role": "user", "content": f"Text:\n{_excerpt(page, q)}\n\nQuestion: {q}"}]) or "").strip()
    if not said or "NOT IN TEXT" in said.upper():
        return False, "blind check: the page doesn't answer it"
    key = stems(a)
    if key and len(key & stems(said)) / len(key) < 0.35:
        return False, f"blind check: the page answers differently ({said[:120]})"
    return True, ""


def main(cur: str, study: str, gen=None) -> dict:
    cur, study = Path(cur), Path(study)
    review_path = cur / "review.json"
    review = json.loads(review_path.read_text())
    seen = set(review["drop"]) | set(review["edit"]) | set(review.get("keep", []))
    pairs = [json.loads(line) for line in (study / "pairs.jsonl").read_text().splitlines()]
    new = [p for p in pairs if p["instruction"] not in seen]
    if not new:
        print("nothing new to review")
        return {"kept": 0, "dropped": 0}
    if gen is None:
        from augment_curriculum import prime_gpu_generator, core_generator
        gen = prime_gpu_generator() if "--prime-gpu" in sys.argv else core_generator()
    pages = {}
    kept = dropped = 0
    for p in new:
        topic = p.get("topic", "")
        if topic not in pages:
            f = study / "sources" / f"{_slug(topic)}.txt"
            pages[topic] = f.read_text(encoding="utf-8", errors="replace") if f.exists() else ""
        ok, why = check(p, pages[topic], gen)
        if ok:
            review.setdefault("keep", []).append(p["instruction"])
            kept += 1
        else:
            review["drop"][p["instruction"]] = why
            dropped += 1
        print(("keep " if ok else "drop ") + p["instruction"][:80] + (f"  [{why}]" if why else ""), flush=True)
    note = "blind check (scripts/blind_review.py)"
    if note not in review.get("reviewed_by", ""):
        review["reviewed_by"] = (review.get("reviewed_by", "") + "; later pairs: " + note).strip("; ")
    review_path.write_text(json.dumps(review, indent=1, ensure_ascii=False))
    print(f"blind review: kept {kept}, dropped {dropped}")
    return {"kept": kept, "dropped": dropped}


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

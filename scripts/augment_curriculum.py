#!/usr/bin/env python3
"""Augment a reviewed study curriculum for a knowledge adapter (oeql v2).

v1 learned the confident short style, not the facts, and stopped saying when
it hadn't read something. This adds, from her own mind (Core) with code checks:

  rewordings  4 other ways to ask each trained question. Kept only if a
              rewording ends in '?', isn't the original, and keeps at least
              half of the question's content words (still asks the same thing).
  unstudied   per topic, questions an engineer might ask that her notes do NOT
              answer. Kept only if most of the question's own content words
              she can't answer it from her notes and the page she read (asked
              to answer from that text alone, she says NOT IN TEXT), so 'I
              haven't read that yet' is the true answer.

Output: <study dir>/augment.jsonl, one record per line:
  {"kind": "reword", "instruction": original, "rewordings": [...]}
  {"kind": "unstudied", "topic": ..., "instruction": question}

Run in gaia-core (her Core is at localhost:8092):
  python /gaia/GAIA_Project/scripts/augment_curriculum.py \
      /gaia/GAIA_Project/knowledge/curricula/professional_engineering \
      /knowledge/studies/professional_engineering
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/app")
from gaia_core.cognition.study_syllabus import _slug, http_generator  # noqa: E402

_WORD = re.compile(r"[a-z][a-z'-]{3,}")
_STOP = set("that this with have from they there their about would could should which where when "
            "will your here into been were what also only must need does more many much some "
            "there other these those then than them make made like just very well even".split())


def stems(text):
    return {w[:5] for w in _WORD.findall((text or "").lower()) if w not in _STOP}


def rewordings(q, gen):
    out = gen([{"role": "system", "content": "Rewrite the question 4 different ways, as different "
                "people might ask it. Same meaning, same specifics. One per line, no numbering."},
               {"role": "user", "content": q}]) or ""
    base, keep = stems(q), []
    for line in out.splitlines():
        r = re.sub(r"^\s*(?:[-*\d.)]+\s*)", "", line).strip().strip('"')
        if not r.endswith("?") or r.lower() == q.lower() or r in keep:
            continue
        if base and len(base & stems(r)) / len(base) < 0.5:
            continue
        keep.append(r)
    return keep[:4]


def answerable(q, text, gen):
    """She tries to answer from the text alone; NOT IN TEXT means she can't.
    A question on a concept she read but can't answer from it is truly unstudied."""
    out = gen([{"role": "system", "content": "Answer the question using only the text below. If the text "
                "does not contain the answer, reply exactly: NOT IN TEXT"},
               {"role": "user", "content": f"Text:\n{text}\n\nQuestion: {q}"}]) or ""
    return "NOT IN TEXT" not in out.upper()


def unstudied(topic, notes, source, gen, n=2):
    out = gen([{"role": "system", "content": f"Here are study notes on {topic}. Write {n + 2} questions "
                "an engineer might ask about this subject that these notes do NOT answer. "
                "One per line, no numbering."},
               {"role": "user", "content": notes}]) or ""
    known = stems(notes) | stems(source)
    named = stems(topic)
    keep = []
    for line in out.splitlines():
        q = re.sub(r"^\s*(?:[-*\d.)]+\s*)", "", line).strip().strip('"')
        own = stems(q) - named
        if not q.endswith("?") or len(own) < 2:
            continue
        if len(own & known) / len(own) > 0.8:        # plainly about what she read
            continue
        if not answerable(q, notes + "\n\n" + source[:6000], gen):
            keep.append(q)
    return keep[:n]


def main(cur, study):
    cur, study = Path(cur), Path(study)
    out_path = study / "augment.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            if r["kind"] == "reword":
                done.add(("reword", r["instruction"]))
            elif r["kind"] == "unstudied_done":
                done.add(("unstudied", r["topic"]))
    gen = http_generator(max_tokens=300, temperature=0.7)
    train = [json.loads(l) for l in (cur / "train.jsonl").read_text().splitlines() if l.strip()]
    with out_path.open("a", encoding="utf-8") as f:
        for i, p in enumerate(train, 1):
            if ("reword", p["instruction"]) in done:
                continue
            f.write(json.dumps({"kind": "reword", "instruction": p["instruction"],
                                "rewordings": rewordings(p["instruction"], gen)}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"reworded {i}/{len(train)}", flush=True)
        for topic in dict.fromkeys(p["topic"] for p in train):
            if ("unstudied", topic) in done:
                continue
            key = _slug(topic)
            notes = "\n".join(x.read_text() for x in sorted((study / "notes").glob(f"{key}*.md")))
            source = "\n".join(x.read_text() for x in sorted((study / "sources").glob(f"{key}*.txt")))
            if not notes:
                continue
            for q in unstudied(topic, notes, source, gen):
                f.write(json.dumps({"kind": "unstudied", "topic": topic, "instruction": q},
                                   ensure_ascii=False) + "\n")
            f.write(json.dumps({"kind": "unstudied_done", "topic": topic}) + "\n")
            f.flush()
            print(f"unstudied: {topic}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])

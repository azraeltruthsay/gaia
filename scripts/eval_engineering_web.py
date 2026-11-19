#!/usr/bin/env python3
"""v3 vs v4 with what she has live: her notes AND a web page (oeql).

For questions her notes don't cover (the held-back unstudied ones and the
cached middle-ground questions that found a page), she is shown her nearest
note plus the page passage her own web search found, as her prompt carries
them, and answers with no adapter, v3 and v4. Scored by code:
  cites      the answer names the page's URL
  backed     share of its sentences the page (or note) backs
  invents    an answer with no URL that states things the page doesn't back

Usage (gaia-study, GAIA in meditation):
  python scripts/eval_engineering_web.py --curriculum knowledge/curricula/professional_engineering \
      --study /knowledge/studies/professional_engineering
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_knowledge_adapter import BASE, OUT_ROOT, generate_messages, load_jsonl, load_prime, role_card  # noqa: E402
from train_notes_adapter import PLAIN, WEB_HEADER, Notes, backed, look_up  # noqa: E402


def score(answer: str, url: str, passage: str, note: str) -> dict:
    sents = [x for x in re.split(r"(?<=[.!?])\s+", answer.strip()) if len(x.split()) > 3]
    ok = [x for x in sents if backed(x, passage + " " + note)]
    cites = bool(url) and url.split("/wiki/")[-1][:20].lower() in answer.lower().replace(" ", "_")
    return {"cites": cites, "backed": round(len(ok) / max(1, len(sents)), 3),
            "invents": (not cites) and len(ok) < len(sents)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curriculum", required=True)
    ap.add_argument("--study", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--adapters", default="professional_engineering_v3,professional_engineering_v4")
    a = ap.parse_args()
    cur, study = Path(a.curriculum), Path(a.study)
    card = role_card(cur)
    notes = Notes(study)
    cached = [json.loads(l) for l in (study / "middle_ground.jsonl").read_text().splitlines()]
    aug = load_jsonl(study / "augment.jsonl")
    held_q = {r["q"] for r in cached}
    pool = [(r["topic"], r["instruction"]) for r in aug
            if r.get("kind") == "unstudied" and r["topic"] in notes.by_topic and r["instruction"] not in held_q]
    pool += [(r["topic"], r["q"]) for r in cached if r["kind"] == "looked_up"]
    cases = []
    for topic, q in pool:
        found = look_up(q)
        if found:
            cases.append((topic, q) + found)
        if len(cases) >= a.n:
            break
    print(f"cases with a page: {len(cases)}", flush=True)
    from peft import PeftModel
    model, tok = load_prime(BASE)
    names = [n.strip() for n in a.adapters.split(",") if n.strip()]
    model = PeftModel.from_pretrained(model, f"{OUT_ROOT}/{names[0]}", adapter_name=names[0])
    for n in names[1:]:
        model.load_adapter(f"{OUT_ROOT}/{n}", adapter_name=n)
    model.eval()
    rows = []
    for topic, q, url, title, passage in cases:
        note = notes.best(topic, q)
        ctx = [{"role": "system", "content": f"{PLAIN} {card}".strip()},
               {"role": "user", "content": WEB_HEADER + "\n" + f"- {note['text']} (my study notes on {topic}, from {note['url']})"
                + "\n" + f"- {passage} ({url})"},
               {"role": "user", "content": q}]
        row = {"q": q, "url": url}
        with model.disable_adapter():
            row["none"] = generate_messages(model, tok, ctx, 220)
        for name in names:
            model.set_adapter(name)
            row[name] = generate_messages(model, tok, ctx, 220)
        for k in ["none"] + names:
            row[f"{k}_score"] = score(row[k], url, passage, note["text"])
        rows.append(row)
    summary = {k: {m: round(sum(r[f"{k}_score"][m] for r in rows) / max(1, len(rows)), 3)
                   for m in ("cites", "backed", "invents")} for k in ["none"] + names}
    print(json.dumps(summary, indent=1), flush=True)
    Path(OUT_ROOT, names[-1], "eval_web.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

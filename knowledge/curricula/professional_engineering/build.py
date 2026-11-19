"""Build the Professional Engineering curriculum from GAIA's study pairs.

  pairs.jsonl (her checked pairs, /knowledge/studies/professional_engineering)
  + review.json (drops and edits, keyed by question)
  -> train.jsonl and held_out.jsonl (every 10th kept pair, for a recall test)

Pairs the review hasn't seen are listed so they get reviewed before training.
"""
import json
import sys
from pathlib import Path

here = Path(__file__).parent
src = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/gaia/gaia-instance/knowledge/studies/professional_engineering/pairs.jsonl")
review = json.loads((here / "review.json").read_text())
seen = set(review["drop"]) | set(review["edit"]) | set(review.get("keep", []))
kept, unreviewed = [], []
for line in src.read_text().splitlines():
    p = json.loads(line)
    q = p["instruction"]
    if q in review["drop"]:
        continue
    p.update(review["edit"].get(q, {}))
    kept.append(p)
    if q not in seen:
        unreviewed.append(q)
held = kept[9::10]
train = [p for i, p in enumerate(kept) if i % 10 != 9]
for name, rows in (("train.jsonl", train), ("held_out.jsonl", held)):
    (here / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
print(f"kept {len(kept)} (train {len(train)}, held out {len(held)}), "
      f"dropped {sum(1 for _ in src.read_text().splitlines()) - len(kept)}, unreviewed {len(unreviewed)}")
for q in unreviewed:
    print("  unreviewed:", q)

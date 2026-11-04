#!/usr/bin/env python3
"""Read the polygraph: what stood out in one session or turn against the rest.

  python3 activation_read.py --session S [--against T] [--layer 35] [--tier prime]
  python3 activation_read.py --session S --token Google

Each neuron and SAE feature is ranked by how much more often it appears in the
selection than in the comparison, so what is always on falls away. Run inside a
container that mounts /logs. Rows written before 6k9i have no sessions or layers.
"""
import argparse
import collections
import json

STREAM = "/logs/activation_stream.jsonl"


def rows(path=STREAM):
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if "layers" in row:
                yield row


def rates(selected, layer, key):
    seen = collections.Counter(i for r in selected for i, _ in r["layers"].get(layer, {}).get(key, []))
    return {i: n / max(1, len(selected)) for i, n in seen.items()}


def stand_out(selected, against, layer, key, min_rate=0.05, top=10):
    """(id, rate here, rate there, lift), largest lift first."""
    here, there = rates(selected, layer, key), rates(against, layer, key)
    floor = 1 / max(1, len(against))
    found = [(i, r, there.get(i, 0.0), r / max(there.get(i, 0.0), floor)) for i, r in here.items() if r >= min_rate]
    return sorted(found, key=lambda x: -x[3])[:top]


def names_from(selected):
    return {(f["layer"], f["idx"]): f["label"] for r in selected for f in r["features"]
            if not f["label"].startswith(("neuron_", "feature_"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="")
    ap.add_argument("--turn", default="")
    ap.add_argument("--against", default="", help="session to compare with; default every other row")
    ap.add_argument("--layer", default="35")
    ap.add_argument("--tier", default="prime")
    ap.add_argument("--token", default="", help="only tokens containing this")
    a = ap.parse_args()
    every = [r for r in rows() if r["tier"] == a.tier]
    def chosen(r):
        return (not a.session or r["session_id"].startswith(a.session)) and (not a.turn or r["turn_id"] == a.turn) \
            and a.token.lower() in r["token"].lower()
    selected = [r for r in every if chosen(r)]
    against = [r for r in every if r["session_id"].startswith(a.against)] if a.against else \
        [r for r in every if not chosen(r)]
    print(f"{len(selected)} tokens against {len(against)}, layer {a.layer}")
    if selected:
        print("said:", "".join(r["token"] for r in selected[:60]).replace("\n", " ")[:200])
    named = names_from(every)
    for key, title in (("f", "SAE features"), ("z", "unusual neurons"), ("n", "loud neurons")):
        print(f"-- {title} (here / there / lift)")
        for i, here, there, lift in stand_out(selected, against, a.layer, key):
            print(f"  {i:>5}  {here:.2f} / {there:.2f} / {lift:5.1f}  {named.get((int(a.layer), i), '')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""What the shadow intent reading said beside what routing did (d3px).

Joins /shared/evals/intent_d3px/shadow.jsonl with the outcome ledger
(/shared/outcomes/ledger.jsonl) by session and time: ledger records in the
same session within --window seconds after a shadow reading belong to that
turn. Prints the turns where the two disagree in ways that matter:
  - shadow says lookup/fact but the turn ran no lookup or search
  - shadow says chat/self/state but the turn searched the web
  - shadow says recite/create/tool and what the turn did

Run inside gaia-core (the files live in its /shared; they hold private text,
so the report stays on this machine):
    docker cp scripts/intent_shadow_report.py gaia-core:/tmp/
    docker exec gaia-core python /tmp/intent_shadow_report.py [--since-hours 24]
"""
import argparse
import collections
import json
import time

SHADOW = "/shared/evals/intent_d3px/shadow.jsonl"
LEDGER = "/shared/outcomes/ledger.jsonl"


def load(path, since):
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if float(e.get("ts") or 0) >= since:
                    out.append(e)
    except FileNotFoundError:
        pass
    return out


def what_routing_did(events):
    did = []
    for e in events:
        kind = e.get("kind")
        if kind == "memory_lookup":
            hits = e.get("hits") or {}
            did.append(f"lookup:{e.get('mode')}" + ("+web" if isinstance(hits, dict) and hits.get("web") else ""))
        elif kind == "tool_execution" and e.get("tool"):
            did.append(f"tool:{e.get('tool')}.{e.get('action') or ''}".rstrip("."))
        elif kind == "tool_selection" and e.get("intent"):
            did.append(f"intent:{e.get('intent')}")
    return sorted(set(did))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-hours", type=float, default=24)
    ap.add_argument("--window", type=float, default=180)
    a = ap.parse_args()
    since = time.time() - a.since_hours * 3600
    shadow = load(SHADOW, since)
    by_session = collections.defaultdict(list)
    for e in load(LEDGER, since):
        by_session[e.get("session_id")].append(e)

    labels = collections.Counter(s["label"] for s in shadow)
    print(f"{len(shadow)} turns read since {a.since_hours:g}h ago: " + ", ".join(f"{k} {v}" for k, v in labels.most_common()))
    notable = []
    for s in shadow:
        t0 = float(s["ts"])
        events = [e for e in by_session.get(s["session_id"], []) if t0 <= float(e.get("ts") or 0) <= t0 + a.window]
        did = what_routing_did(events)
        searched = any("web" in d or d.startswith("tool:web") for d in did)
        looked = any(d.startswith("lookup") for d in did) or searched
        why = ""
        if s["label"] in ("lookup", "fact") and not looked:
            why = "wanted facts, nothing looked up"
        elif s["label"] in ("chat", "self", "state") and searched:
            why = "searched the web for " + s["label"]
        elif s["label"] in ("recite", "create", "tool", "recall", "dispute", "followup"):
            why = s["label"]
        if why:
            notable.append((why, s, did))
    for why, s, did in notable:
        print(f"- [{why}] shadow={s['label']} (rule={s.get('rule')}, embed={s['embed']} {s['embed_sim']}) "
              f"did={did or ['nothing recorded']}\n    {s['text'][:140]!r}")


if __name__ == "__main__":
    main()

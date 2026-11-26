#!/usr/bin/env python3
"""Range battery: conversation, physics, philosophy, engineering, logic,
history, poetry and basic facts, asked of GAIA's live self.

Each run draws a share of every category's questions at random (default
15%: 6 of 40) and asks them through gaia-core's full /process_packet
pipeline, each in a fresh cogtest-* session (conversation memory leaves
those out). Questions used by earlier runs of the same gear are skipped
until the bank is used up, so repeated runs cover more of it.

Recorded per question: the reply, which mind answered, the tools she
used, time taken, any retraction, the second mind's verdict, an
automatic check where the answer is crisp (regexes in questions.jsonl),
and flags for things wrong whatever the question (empty reply, protocol
leaking into the text, a runaway length).

    python3 scripts/range_battery.py --gear awake             # gear 1, Core on the GPU
    python3 scripts/range_battery.py --gear focusing          # gear 2, Prime on the GPU
    python3 scripts/range_battery.py --gear awake --fraction 0.25 --seed 7
    python3 scripts/range_battery.py --summary                # tables over all runs

Runs on the host (stdlib only); needs gaia-core on :6415 and the
orchestrator on :6410.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from urllib.request import Request, urlopen

HERE = os.path.dirname(os.path.abspath(__file__))
BANK_DIR = os.path.join(HERE, "..", "knowledge", "evals", "range_battery")
QUESTIONS = os.path.join(BANK_DIR, "questions.jsonl")
RUNS = os.path.join(BANK_DIR, "runs")
CORE = os.environ.get("CORE_ENDPOINT", "http://localhost:6415")
ORCH = os.environ.get("ORCHESTRATOR_ENDPOINT", "http://localhost:6410")
GPU_TIER = {"awake": "core", "focusing": "prime"}

sys.path.insert(0, os.path.join(HERE, "..", "candidates", "gaia-doctor"))
from cognitive_test_battery import build_packet  # noqa: E402  (stdlib-only)

# The reply opens with her label ("[Core]"), sometimes after a line saying
# what she did ("*[What I did]* answered as Core · searched the web").
_LABEL_RE = re.compile(r"^\s*(?:\*\[What I did\]\*[^\n\[]*)?\[(Core|Prime|Groq)[^\]]*\]", re.I)
_LEAK_RE = re.compile(r"</?tool_(?:call|calling|result|response)|<\|[a-z_]+\|>|\bthought\b\s*\n|<think>"
                      r"|\[GAIA (?:thinking|action)\]", re.I)
_DID_RE = re.compile(r"\*\[What I did\]\*([^\n\[]*)")


def _get(url: str) -> dict:
    with urlopen(url, timeout=15) as r:
        return json.loads(r.read())


def _post(url: str) -> dict:
    with urlopen(Request(url, data=b"{}", headers={"Content-Type": "application/json"}), timeout=30) as r:
        return json.loads(r.read() or b"{}")


def gear_ready(gear: str) -> bool:
    try:
        st = _get(f"{ORCH}/lifecycle/state")
    except OSError:   # the orchestrator can be slow to answer mid-shift
        return False
    tier = (st.get("tiers") or {}).get(GPU_TIER[gear]) or {}
    return st.get("state") == gear and tier.get("device") == "gpu" and not st.get("transitioning")


def ensure_gear(gear: str, wait_s: int = 600) -> None:
    if gear_ready(gear):
        return
    print(f"shifting to {gear}...", flush=True)
    _post(f"{ORCH}/consciousness/{gear}")
    t0 = time.time()
    while time.time() - t0 < wait_s:
        time.sleep(5)
        if gear_ready(gear):
            print(f"  {gear} in {time.time() - t0:.0f}s", flush=True)
            return
    raise SystemExit(f"{gear} not reached in {wait_s}s: {json.dumps(_get(f'{ORCH}/lifecycle/state'))[:400]}")


def ask(q: str, session_id: str, timeout: int = 300) -> dict:
    pkt = build_packet(q, session_id, source="web")
    req = Request(f"{CORE}/process_packet", data=json.dumps(pkt).encode(),
                  headers={"Content-Type": "application/json"})
    tokens, tools, retracts, final = [], [], 0, None
    t0 = time.time()
    with urlopen(req, timeout=timeout) as r:
        for raw in r:
            try:
                ev = json.loads(raw)
            except ValueError:
                continue
            kind = ev.get("type")
            if kind == "token":
                tokens.append(ev.get("value", ""))
            elif kind == "flush":
                tokens.append("\n\n")   # a message ends; her next one is separate (6f1x: Tyger)
            elif kind == "tool_status":
                tools.append(f"{ev.get('tool')}({ev.get('action') or ''}):{ev.get('state')}")
            elif kind == "retract":
                retracts += 1
                tokens.append(" <RETRACT> ")
            elif kind == "response":
                final = (ev.get("response") or {}).get("candidate")
    reply = "".join(tokens).strip() or (final or "").strip()
    return {"reply": reply, "tools": tools, "retracts": retracts, "seconds": round(time.time() - t0, 1)}


def second_mind(session_id: str) -> dict | None:
    """What the other mind made of the reply (its log lives in gaia-core's /shared)."""
    try:
        out = subprocess.run(["docker", "exec", "gaia-core", "sh", "-c",
                              f"grep -F '\"{session_id}\"' /shared/observer/second_mind.jsonl | tail -1"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        e = json.loads(out) if out else None
    except Exception:
        return None
    return {k: e.get(k) for k in ("verdict", "unbacked", "followup", "sites", "observer")} if e else None


def grade(rec: dict, reply: str) -> str:
    if rec["grade"] != "auto":
        return "read"
    body = _LABEL_RE.sub("", reply)
    return "pass" if all(re.search(rx, body, re.I) for rx in rec["expect"]) else "fail"


def flags(reply: str, res: dict) -> list:
    f = []
    body = _LABEL_RE.sub("", reply).strip()
    if not body:
        f.append("empty")
    if _LEAK_RE.search(reply):
        f.append("protocol_leak")
    if len(body) > 2500:
        f.append("long")
    lines = [ln.strip() for ln in body.splitlines() if len(ln.strip()) > 20]
    if len(lines) - len(set(lines)) >= 2:
        f.append("repeats_lines")
    if res["retracts"]:
        f.append("retracted")
    if re.search(r"\bas an ai\b|\blanguage model\b", body, re.I):
        f.append("assistant_boilerplate")
    if len(re.findall(r"\[(?:Core|Prime)\]", reply)) > 1:
        f.append("double_label")
    return f


def used_ids(gear: str) -> set:
    ids = set()
    for p in glob.glob(os.path.join(RUNS, f"*_{gear}_*.jsonl")):
        for line in open(p):
            try:
                ids.add(json.loads(line)["id"])
            except (ValueError, KeyError):
                pass
    return ids


def draw(bank: list, fraction: float, rng: random.Random, skip: set) -> list:
    by_cat = defaultdict(list)
    for rec in bank:
        by_cat[rec["category"]].append(rec)
    picked = []
    for cat, recs in by_cat.items():
        n = max(1, math.ceil(len(recs) * fraction))
        fresh = [r for r in recs if r["id"] not in skip]
        pool = fresh if len(fresh) >= n else recs   # bank used up: start over
        picked.extend(rng.sample(pool, n))
    rng.shuffle(picked)
    return picked


def run(gear: str, fraction: float, seed: int | None) -> str:
    bank = [json.loads(line) for line in open(QUESTIONS)]
    seed = seed if seed is not None else random.randrange(1_000_000)
    picked = draw(bank, fraction, random.Random(seed), used_ids(gear))
    ensure_gear(gear)
    os.makedirs(RUNS, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(RUNS, f"{stamp}_{gear}_{seed}.jsonl")
    print(f"{len(picked)} questions, seed {seed} -> {path}", flush=True)
    with open(path, "w") as out:
        for i, rec in enumerate(picked, 1):
            if not gear_ready(gear):
                ensure_gear(gear)
            sid = f"cogtest-range-{uuid.uuid4().hex[:8]}"
            try:
                res = ask(rec["q"], sid)
            except Exception as e:
                res = {"reply": "", "tools": [], "retracts": 0, "seconds": None, "error": str(e)[:200]}
            time.sleep(2)   # the second mind reads the reply once the turn is over
            label = _LABEL_RE.match(res["reply"])
            did = _DID_RE.search(res["reply"])
            if did:   # routed lookups show here, not as tool_status events
                res["tools"] += [t.strip() for t in did.group(1).split("·")[1:] if t.strip()]
            row = {**rec, "gear": gear, "seed": seed, "session_id": sid,
                   "mind": label.group(1).lower() if label else None, **res,
                   "auto": grade(rec, res["reply"]), "flags": flags(res["reply"], res),
                   "second_mind": second_mind(sid)}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            short = " ".join(_LABEL_RE.sub("", res["reply"]).split())[:110]
            print(f"[{i}/{len(picked)}] {rec['id']} {row['auto']:4} {row['seconds']}s "
                  f"{','.join(row['flags'])} :: {short}", flush=True)
    return path


def rescore() -> None:
    """Grade every run again against the current bank (after a pattern is fixed)."""
    bank = {r["id"]: r for r in (json.loads(line) for line in open(QUESTIONS))}
    for p in sorted(glob.glob(os.path.join(RUNS, "*.jsonl"))):
        rows = [json.loads(line) for line in open(p)]
        for r in rows:
            r.update({k: bank[r["id"]][k] for k in ("grade", "expect") if k in bank[r["id"]]})
            r["auto"] = grade(r, r["reply"])
            r["flags"] = flags(r["reply"], r)
            did = _DID_RE.search(r["reply"])
            if did and not r.get("tools"):
                r["tools"] = [t.strip() for t in did.group(1).split("·")[1:] if t.strip()]
        with open(p, "w") as out:
            out.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def summary() -> None:
    rows = [json.loads(line) for p in sorted(glob.glob(os.path.join(RUNS, "*.jsonl"))) for line in open(p)]
    table = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0.0, 0]))  # gear -> cat -> pass, auto, n, secs, flagged
    for r in rows:
        t = table[r["gear"]][r["category"]]
        t[2] += 1
        t[3] += r.get("seconds") or 0
        t[4] += bool(r.get("flags"))
        if r["auto"] in ("pass", "fail"):
            t[1] += 1
            t[0] += r["auto"] == "pass"
    for gear, cats in table.items():
        print(f"\n{gear}: {sum(c[2] for c in cats.values())} questions")
        print(f"  {'category':13} {'auto pass':>10} {'asked':>6} {'avg s':>6} {'flagged':>8}")
        for cat in sorted(cats):
            p, a, n, s, fl = cats[cat]
            print(f"  {cat:13} {f'{p}/{a}' if a else '-':>10} {n:>6} {s / n:>6.1f} {fl:>8}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gear", choices=sorted(GPU_TIER))
    ap.add_argument("--fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--rescore", action="store_true", help="grade all runs again, then summarise")
    a = ap.parse_args()
    if a.rescore:
        rescore()
        summary()
    elif a.summary:
        summary()
    elif a.gear:
        if not 0.05 <= a.fraction <= 1:
            raise SystemExit("--fraction between 0.05 and 1")
        run(a.gear, a.fraction, a.seed)
    else:
        ap.error("--gear or --summary")

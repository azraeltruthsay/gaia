#!/usr/bin/env python3
"""Read intent from each mind's own hidden state (GAIA_Project-d3px.2/.3).

The labeled messages (/shared/evals/intent_d3px/labeled_v3.json: act, source,
depth per message) are private and stay out of git; so do the hidden states.

Run inside gaia-core (the files live in its /shared):
    docker cp scripts/intent_probe_d3px.py gaia-core:/tmp/
    docker exec gaia-core python /tmp/intent_probe_d3px.py extract http://localhost:8092/hidden/read core_gpu
    docker exec gaia-core python /tmp/intent_probe_d3px.py extract http://gaia-prime:7777/hidden/read prime_cpu
    docker exec gaia-core python /tmp/intent_probe_d3px.py cross

2026-11-27, 847 messages, five folds holding out whole sessions, logistic
probes on a mid + last layer (Core L20+L47, Prime L20+L35):
               MiniLM   Core GPU   Prime CPU   both agree (share: right)
    act          64%      85%        87%        87%: 92%
    source       63%      81%        79%        81%: 89%
    depth        89%      94%        96%        95%: 97%
Where the minds disagree, each is near a coin flip: the hard ones.
Reads: Core 12B GPU 32 ms a layer; Prime 8B CPU ~150 ms (10 layers).
"""
import sys
import json
import time
import urllib.request

import numpy as np

def extract(url, tag):
    rows = json.load(open('/shared/evals/intent_d3px/labeled_v3.json'))
    out = {}
    t0 = time.time()
    for i, r in enumerate(rows):
        body = json.dumps({"messages": [{"role": "user", "content": r["text"][:2000]}]}).encode()
        for attempt in range(3):
            try:
                got = json.load(urllib.request.urlopen(urllib.request.Request(
                    url, body, {"Content-Type": "application/json"}), timeout=120))
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(5)
        for layer, vec in got["layers"].items():
            out.setdefault(int(layer), []).append(np.asarray(vec, dtype=np.float32))
        if i % 100 == 0:
            print(i, f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(f'/shared/evals/intent_d3px/hidden_{tag}.npz',
                        **{f"L{k}": np.stack(v) for k, v in out.items()})
    print("done", len(rows), f"{time.time() - t0:.0f}s")


def cross():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = json.load(open('/shared/evals/intent_d3px/labeled_v3.json'))
    groups = np.array([r["session"] for r in rows])
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(rows)), groups=groups))
    C = np.load('/shared/evals/intent_d3px/hidden_core_gpu.npz'); P = np.load('/shared/evals/intent_d3px/hidden_prime_cpu.npz')
    XC = np.hstack([C["L20"], C["L47"]]); XP = np.hstack([P["L20"], P["L35"]])
    def cv(X, y):
        pred = np.empty(len(y), dtype=object); prob = np.zeros(len(y))
        for tr, te in folds:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")).fit(X[tr], y[tr])
            p = m.predict_proba(X[te]); pred[te] = m.classes_[p.argmax(1)]; prob[te] = p.max(1)
        return pred, prob
    out = {}
    for ax in ("act", "source", "depth"):
        y = np.array([r[ax] for r in rows])
        pc, qc = cv(XC, y); pp, qp = cv(XP, y)
        agree = pc == pp
        out[ax] = (pc, pp, qc, qp)
        print(f"{ax:6} Core {np.mean(pc == y):.0%}  Prime {np.mean(pp == y):.0%}  | agree on {agree.mean():.0%}: "
              f"{np.mean(pc[agree] == y[agree]):.0%} right; disagree on {(~agree).mean():.0%}: Core {np.mean(pc[~agree] == y[~agree]):.0%}, Prime {np.mean(pp[~agree] == y[~agree]):.0%} right")
    # routing: web off only when both minds read a non-web source
    y = np.array([r["source"] for r in rows]); need = y == "web"
    pc, pp, qc, qp = out["source"]
    for name, skip in (("Core alone >=0.8", (pc != "web") & (qc >= 0.8)),
                       ("both agree non-web", (pc != "web") & (pp != "web")),
                       ("both agree, both >=0.8", (pc != "web") & (pp != "web") & (qc >= 0.8) & (qp >= 0.8))):
        print(f"web off [{name}]: {(skip & ~need).sum()}/{(~need).sum()} not needing it, wrongly {(skip & need).sum()}/{need.sum()}")
    np.savez('/shared/evals/intent_d3px/cross_cv.npz', **{f"{ax}_{k}": v.astype(str) if v.dtype == object else v
             for ax, t in out.items() for k, v in zip(("core", "prime", "qcore", "qprime"), t)})


if __name__ == "__main__":
    if sys.argv[1:2] == ["extract"]:
        extract(sys.argv[2], sys.argv[3])
    elif sys.argv[1:2] == ["cross"]:
        cross()
    else:
        raise SystemExit(__doc__)

"""Evaluate a thinking adapter on its held-out turns (h1ok, isep, w6in, b9mp).

Each held-out user message is rebuilt the way live deliberation builds it
(gen_thinking_teacher.build: session, history, persona, remembered grounding)
and sent to the tier's engine as deliberate() sends it, once with the adapter
and once bare. Reported per mode: how often a thought came back, answers left
empty, answers carrying a made-up <tool_result> or a note to herself, thought
length and time.

Usage (in gaia-core, the engine up with the tier loaded):
  python eval_thinking_adapter.py --tier prime --adapter prime_thinking_v3 \
      --path /models/lora_adapters/tier1_global/prime_thinking_v3 --endpoint http://gaia-prime:7777
"""
import argparse
import json
import re
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, "/gaia/GAIA_Project/scripts")
import gen_thinking_teacher as teacher  # noqa: E402

FAKE_RE = re.compile(r"</?tool_(?:result|response)>|<tool_call>", re.I)
NOTE_RE = re.compile(r"^[ \t*_]*(?:THOUGHT_SEED|GOAL_SHIFT|Action: none)\b", re.I | re.M)


def post(endpoint, path, body, timeout=900):
    req = urllib.request.Request(endpoint + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def ask(endpoint, messages, max_tokens):
    t = time.time()
    d = post(endpoint, "/v1/chat/completions", {
        "messages": messages, "max_tokens": max_tokens, "temperature": 0.55, "top_p": 0.9,
        "chat_template_kwargs": {"enable_thinking": True, "native_thinking": True}})
    m = d["choices"][0]["message"]
    return (m.get("reasoning_content") or ""), (m.get("content") or ""), time.time() - t


def summarize(rows):
    n = len(rows) or 1
    thought = [r for r in rows if r["thought"].strip()]
    return {
        "n": len(rows),
        "thought_rate": round(len(thought) / n, 2),
        "empty_answer": sum(1 for r in rows if not r["answer"].strip()),
        "fake_tool_result": sum(1 for r in rows if FAKE_RE.search(r["answer"])),
        "self_note": sum(1 for r in rows if NOTE_RE.search(r["answer"])),
        "thought_words_median": statistics.median([len(r["thought"].split()) for r in thought]) if thought else 0,
        "seconds_median": round(statistics.median([r["s"] for r in rows]), 1) if rows else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=sorted(teacher.TIERS), required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--out")
    args = ap.parse_args()

    users = json.load(open(f"{args.path}/holdout_users.json"))[:args.n]
    by_user = {t["user"]: t for t in json.load(open("/shared/tmp_h1ok/turns.json"))}
    turns = [by_user.get(u, u) for u in users]
    res = post(args.endpoint, "/adapter/load", {"name": args.adapter, "path": args.path}, timeout=300)
    if not res.get("ok"):
        print("adapter load failed:", res)
        return 1

    out = {}
    for mode in ("adapter", "bare"):
        post(args.endpoint, "/adapter/set", {"name": args.adapter if mode == "adapter" else None})
        rows = []
        for turn in turns:
            _, msgs = teacher.build(args.tier, turn)
            try:
                thought, answer, s = ask(args.endpoint, msgs, args.max_tokens)
            except Exception as e:
                thought, answer, s = "", f"[error {e}]", 0.0
            rows.append({"user": turn["user"] if isinstance(turn, dict) else turn,
                         "thought": thought, "answer": answer, "s": round(s, 1)})
            print(mode, len(rows), f"{s:.0f}s", "think" if thought.strip() else "-", flush=True)
        out[mode] = {"summary": summarize(rows), "rows": rows}
    post(args.endpoint, "/adapter/set", {"name": None})
    for mode in out:
        print(mode, json.dumps(out[mode]["summary"]))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

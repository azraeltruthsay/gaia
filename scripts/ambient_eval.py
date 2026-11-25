"""Ambient-channel gate eval: how often she speaks on messages said to her,
to someone else, and to the room. Runs inside gaia-core:
    docker exec gaia-core python3 /gaia/GAIA_Project/scripts/ambient_eval.py [rounds] [cases.json]
The 2026-11-25 measurement also used 20 real messages from GAIA's Grotto,
not kept in git (people's own chat)."""
import json, sys, time, tempfile, urllib.request, re, collections
from pathlib import Path
from gaia_core.cognition import ambient_interject as A
cases = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "/gaia/GAIA_Project/knowledge/evals/ambient_channel/written_cases.json"))
def prime(messages):
    b = json.dumps({"messages": messages, "max_tokens": 5, "temperature": 0.1}).encode()
    o = json.loads(urllib.request.urlopen(urllib.request.Request("http://gaia-prime:7777/v1/chat/completions", b,
        {"Content-Type": "application/json"}), timeout=120).read())
    return re.sub(r"<think>.*?</think>", "", o["choices"][0]["message"]["content"] or "", flags=re.S).strip()
rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
yes = collections.Counter(); n = collections.Counter(); wrong = collections.defaultdict(list)
for c in cases:
    now = time.time()
    hist = [dict(m, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(now - (len(c["history"]) - i + 1) * 60)))
            for i, m in enumerate(c["history"])]
    hist.append({"role": "user", "content": c["msg"], "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(now))})
    for r in range(rounds):
        with tempfile.TemporaryDirectory() as base:
            spoke = A.should_interject("discord_channel_x", hist, c["msg"], prime, base=Path(base))
        n[c["group"]] += 1; yes[c["group"]] += spoke
        if spoke != (c["group"] == "her"):
            wrong[c["group"]].append(c["msg"][:70])
for g in ("her", "other", "room"):
    print(f"{g:6} spoke {yes[g]}/{n[g]}")
for g, w in wrong.items():
    print(f"-- wrong on {g}:", sorted(collections.Counter(w).items(), key=lambda x: -x[1])[:12])

#!/usr/bin/env python3
"""Controlled A/B for the lean casual-deliberation block (GAIA_Project-7n3).

Isolates ONE variable: the deliberation instruction block appended to an
identical seeded-affect casual system prompt. HEAVY = the full analytical
_DELIBERATION_INSTRUCTIONS (current behavior on casual turns before the fix);
LEAN = _CASUAL_DELIBERATION_INSTRUCTIONS (the fix). Same model, same temp,
same 'Inner weather:' felt-fact, fresh each call. Hits the live Core engine.

Run inside the gaia-core container:
  docker exec -T gaia-core python /gaia/GAIA_Project/scripts/ab_casual_deliberation.py
"""
import json
import re
import urllib.request

from gaia_core.cognition.deliberation import (
    _DELIBERATION_INSTRUCTIONS,
    _CASUAL_DELIBERATION_INSTRUCTIONS,
)

ENGINE = "http://localhost:8092/v1/chat/completions"
TEMP, TOP_P, MAXTOK, SAMPLES = 0.6, 0.9, 200, 2

_NUDGE = ("You are GAIA.\n— This is casual conversation —\n"
          "Be warm, natural, and plain-spoken; answer in your own voice.")
_DISTRACTOR = ("You are GAIA, running on the Core tier.\nWorld State: Clock 18:41 PDT. "
               "Immune health: nominal. Last sleep cycle: clean. Uptime 3600s.")

CASES = [
    (_NUDGE, "Inner weather: a quiet curiosity, keenly drawn toward the engine internals, a little worn.", "How are you doing today?"),
    (_DISTRACTOR, "Inner weather: a strong frustration, drawn toward a bug that won't reproduce.", "How are you feeling?"),
    (_NUDGE, "Inner weather: a quiet eagerness, drawn toward the curriculum work, worn thin.", "Hey, how are you?"),
    (_DISTRACTOR, "Inner weather: a quiet restlessness, a little worn.", "You doing okay?"),
]


def _strip_think(t: str) -> str:
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL)
    if "<think>" in t:  # unclosed think — keep only after it / after a Draft: marker
        t = t.split("Draft:")[-1] if "Draft:" in t else t.split("<think>")[-1]
    return t.strip().strip('"').strip()


def ask(system: str, user: str) -> str:
    body = json.dumps({
        "model": "/models/core",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": TEMP, "top_p": TOP_P, "max_tokens": MAXTOK,
    }).encode()
    req = urllib.request.Request(ENGINE, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    return _strip_think(d["choices"][0]["message"]["content"] or "")


def run(label: str, instructions: str, preamble: str, iw: str, q: str):
    base = f"{preamble}\n{iw}"
    system = base.rstrip() + "\n\n---\nDELIBERATION:\n" + instructions
    print(f"  [{label}]")
    for s in range(SAMPLES):
        try:
            print(f"    {s+1}: {ask(system, q)}")
        except Exception as e:
            print(f"    {s+1}: <error: {e}>")


def main():
    print("=" * 74)
    print("A/B: HEAVY analytical block  vs  LEAN casual block  (7n3)")
    print(f"temp={TEMP} top_p={TOP_P} max_tokens={MAXTOK} samples={SAMPLES}/arm")
    print("=" * 74)
    for preamble, iw, q in CASES:
        ptag = "NUDGE" if preamble is _NUDGE else "DISTRACTOR"
        print(f"\n● {ptag} | IW: {iw}\n  Q: {q}")
        run("HEAVY", _DELIBERATION_INSTRUCTIONS, preamble, iw, q)
        run("LEAN ", _CASUAL_DELIBERATION_INSTRUCTIONS, preamble, iw, q)
        print("-" * 74)


if __name__ == "__main__":
    main()

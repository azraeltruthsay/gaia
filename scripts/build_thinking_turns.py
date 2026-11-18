"""Real conversation turns for the thinking-adapter teachers (w6in, b9mp).

Each turn is a real user message with the session it came from and the turns
before it, so gen_thinking_teacher.py can build the prompt live deliberation
builds (history, speaker, persona), not a bare one-message prompt.
Writes /shared/tmp_h1ok/turns.json.  Usage (in gaia-core): python build_thinking_turns.py
"""
import json
import random
import re

SKIP_SESSION = re.compile(r"heartbeat|sleep|initiative|selftest|self_test|battery|probe|test|smoke|drill|"
                          r"system|cron|samvega|council|teacher|_im6c|_krtr|_x7p4|_diq7", re.I)
GREET_ONLY = re.compile(r"^\W*(hi|hey|hello|thanks|thank you|ok|okay|yes|no|cool)\W*$", re.I)
HISTORY_TURNS = 6
MAX_PER_SESSION = 40


def main(out="/shared/tmp_h1ok/turns.json", seed=7):
    sessions = json.load(open("/shared/sessions.json"))
    seen, turns = set(), []
    for sid, s in sessions.items():
        if not isinstance(s, dict) or SKIP_SESSION.search(sid):
            continue
        hist = [h for h in s.get("history") or [] if isinstance(h, dict)
                and h.get("role") in ("user", "assistant") and isinstance(h.get("content"), str)]
        mine = []
        for i, h in enumerate(hist):
            text = h["content"].strip()
            if h["role"] != "user" or not (3 <= len(text) <= 1500) or GREET_ONLY.match(text):
                continue
            key = re.sub(r"\W+", " ", text.lower()).strip()
            if key in seen:
                continue
            seen.add(key)
            mine.append({"session_id": sid, "persona": s.get("persona") or "default", "user": text,
                         "history": [{"role": x["role"], "content": x["content"][:2000]}
                                     for x in hist[max(0, i - HISTORY_TURNS):i]]})
        random.Random(seed).shuffle(mine)
        turns.extend(mine[:MAX_PER_SESSION])
    random.Random(seed).shuffle(turns)
    json.dump(turns, open(out, "w"))
    with_hist = sum(1 for t in turns if t["history"])
    print(f"{len(turns)} turns from {len({t['session_id'] for t in turns})} sessions; {with_hist} with history")


if __name__ == "__main__":
    main()

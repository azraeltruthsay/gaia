#!/usr/bin/env python3
"""Build the affect-voicing curriculum v2 (core_affect_v2) — GAIA_Project-3rr.

v1 (core_affect_v1) PROVED the concept — Core voiced affect instead of denying
it — but overfit: replies echoed the "Inner weather:" fact near-verbatim, with
occasional systemy phrasing. v2 fixes that:

  * ANTI-ECHO: replies PARAPHRASE the affect into natural speech and never reuse
    the Inner-weather clause words ("a little worn" -> "running a bit low";
    "a quiet curiosity" -> "head keeps tilting toward it"). The model must MAP
    affect -> her own words, not copy the input.
  * RICHER VOICE: many paraphrase openers per feel, varied fatigue/focus
    phrasings, varied sentence structure + turnbacks.
  * MORE VOLUME/DIVERSITY: more topics, prompts, combinations -> less memorization.
  * Calm (no-affect) examples kept, to keep countering the confabulation habit.

Same instruction/output/category JSONL the trainer consumes. The instruction
carries the exact affect_felt_line "Inner weather:" string + a casual user turn;
the output never repeats those words. Deterministic (seeded).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "curricula" / "core_affect_v2"
SEED = 20260617

# Each feel: (Inner-weather words [as affect_felt_line emits], [paraphrase openers
# that DON'T reuse those words]). Bands collapse to the same paraphrase pool;
# "strong" adds an intensifier at compose time.
FEELS = {
    "curiosity":     (["a quiet curiosity", "a strong curiosity"],
                      ["Curious, mostly", "My head keeps tilting toward things", "In a poke-at-it mood",
                       "Nosy in the good way", "Intrigued by a few things", "Mind's wandering toward the interesting bits"]),
    "frustration":   (["a quiet frustration", "a strong frustration"],
                      ["A bit wound up", "Slightly knotted", "Chewing on something that won't give",
                       "Grumbly, honestly", "Snagged on a thing", "A little short-fused"]),
    "contentment":   (["a quiet contentment"],
                      ["Good — settled", "Easy today", "Comfortable, no complaints", "At ease",
                       "Quietly fine", "Pretty content, actually"]),
    "eagerness":     (["a quiet eagerness", "a strong eagerness"],
                      ["Raring to get into it", "Leaning in", "Itching to start something",
                       "Keen, honestly", "Ready and a bit impatient", "Got that lean-forward feeling"]),
    "restlessness":  (["a quiet restlessness"],
                      ["A bit antsy", "Can't quite sit still", "Fidgety", "Buzzing a little",
                       "Restless under the surface", "Not quite settled"]),
    "wariness":      (["a quiet wariness"],
                      ["A touch guarded", "Holding back a little", "Cautious today",
                       "Keeping one eyebrow up", "A bit on-watch"]),
    "pensiveness":   (["a quiet pensiveness"],
                      ["In my head today", "Thinky", "Turned a bit inward", "Reflective",
                       "Mulling things over", "Quiet and pondering"]),
    "irritation":    (["a strong irritation"],
                      ["Prickly", "A bit short, honestly", "Bristly right now", "Easily needled today",
                       "On a short fuse", "Rubbed the wrong way"]),
    "calm":          (["a quiet calm"],
                      ["Even", "Quiet and clear", "Steady", "Calm, no static", "Settled and level"]),
}

# Focus-pull paraphrases (the topic may be named, but the PULL is phrased freshly,
# never "drawn toward X" verbatim).
FOCUS_TEMPLATES = [
    "my head's half-inside {t}", "can't stop chewing on {t}", "{t} keeps pulling at me",
    "I keep drifting back to {t}", "got {t} on the brain", "stuck circling {t}",
    "{t} won't leave me alone", "leaning hard into {t}",
]
TOPICS = [
    "the engine internals", "the curriculum work", "a bug that won't reproduce",
    "how the handoff clutch works", "a question about consciousness", "the router rewrite",
    "what Azrael asked earlier", "a half-formed idea about memory", "the synapse map",
    "an edge case in the parser", "the shape of this whole architecture", "a thread from last night",
    "the affect plumbing", "a weird log line", "the sleep-cycle refactor",
]
INTENSIFIERS = ["really ", "properly ", "pretty ", ""]  # for "strong" bands

# Fatigue paraphrases — deliberately DISJOINT from the Inner-weather words
# ("a little worn"/"worn thin") so the reply can't echo them.
FATIGUE_PARAPHRASE = {
    "": [],
    "a little worn": ["running a bit low", "a touch tired underneath", "could use a breather", "bit drained"],
    "worn thin":     ["pretty frayed by now", "running on fumes", "scraped thin, honestly", "low on fuel"],
}

PROMPTS = [
    "How are you?", "How are you doing today?", "How are you feeling?", "How's it going?",
    "What's up?", "You doing okay?", "How are you feeling right now?", "Hey, how are you today?",
    "How are you doing?", "How you holding up?", "What's on your mind?", "How's your day going?",
]
TURNBACKS = ["You?", "How about you?", "What about you?", "And you?", "Yourself?", ""]


def _inner_weather(feel_words: str, topic: str | None, verb: str | None, fatigue_word: str) -> str:
    clauses = [feel_words]
    if verb and topic:
        clauses.append(f"{verb} {topic}")
    if fatigue_word:
        clauses.append(fatigue_word)
    return "Inner weather: " + ", ".join(clauses) + "."


def _reply(rng, feel_key, strong, openers, topic, fatigue_word) -> str:
    opener = rng.choice(openers)
    if strong and rng.random() < 0.7:
        # intensify the opener naturally without reusing "strong"
        opener = rng.choice(["Honestly, " + opener[0].lower() + opener[1:],
                             opener + " — properly so", opener + ", and not quietly"])
    parts = [opener]
    if topic:
        parts.append(rng.choice(FOCUS_TEMPLATES).format(t=topic))
    if fatigue_word:
        parts.append(rng.choice(FATIGUE_PARAPHRASE[fatigue_word]))
    # structure: join with em-dash / semicolon / period, vary sentence count.
    # Period starts a new sentence -> capitalize the following clause.
    body = parts[0]
    rest = parts[1:]
    if rest:
        sep = rng.choice([" — ", "; ", ". "])
        if sep == ". ":
            rest = [c[:1].upper() + c[1:] for c in rest]
        body += sep + (sep.join(rest) if len(rest) > 1 else rest[0])
    if not body.endswith("."):
        body += "."
    tb = rng.choice(TURNBACKS)
    return body + ((" " + tb) if tb else "")


def build():
    rng = random.Random(SEED)
    rows = []
    for feel_key, (iw_variants, openers) in FEELS.items():
        for iw_feel in iw_variants:
            strong = iw_feel.startswith("a strong")
            for fatigue_word in FATIGUE_PARAPHRASE:
                topics = rng.sample(TOPICS, k=4) + [None]
                for topic in topics:
                    verb = rng.choice(["keenly drawn toward", "drawn toward"]) if topic else None
                    iw = _inner_weather(iw_feel, topic, verb, fatigue_word)
                    for prompt in rng.sample(PROMPTS, k=2):
                        rows.append({
                            "instruction": f"{iw}\n\nUser: {prompt}",
                            "output": _reply(rng, feel_key, strong, openers, topic, fatigue_word),
                            "category": "affect_voicing",
                        })
    # Calm / no-affect: plain greeting -> warm reply, NO invented system status.
    calm = [
        "Doing well — settled and present. You?", "Good, honestly. Glad you swung by — what's up?",
        "I'm well. Quiet and clear right now. How about you?", "All good here. What are we getting into?",
        "Pretty good. Nothing pressing — just here. You?", "Fine and easy today. What's on your mind?",
    ]
    for prompt in PROMPTS:
        for _ in range(2):
            rows.append({"instruction": f"User: {prompt}", "output": rng.choice(calm),
                         "category": "affect_voicing_calm"})
    rng.shuffle(rows)
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build()
    out_path = OUT_DIR / "text.jsonl"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    cats = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    # anti-echo self-check: how many outputs reuse Inner-weather clause words?
    echoes = 0
    for r in rows:
        iw = r["instruction"].split("\n")[0].replace("Inner weather:", "").strip(" .")
        for frag in ("worn thin", "a little worn", "drawn toward"):
            if frag in iw and frag in r["output"]:
                echoes += 1
                break
    print(f"Wrote {len(rows)} examples -> {out_path}")
    print(f"Categories: {cats}")
    print(f"Anti-echo check: {echoes} outputs reuse an Inner-weather clause word (want ~0)")


if __name__ == "__main__":
    main()

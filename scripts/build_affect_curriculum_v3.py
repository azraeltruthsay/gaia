#!/usr/bin/env python3
"""Affect-voicing curriculum v3 (research) — GAIA_Project-3rr.

v1 overfit (echoed); v2 generalized but was unreliable — both intermittently
relapsed into the base priors: DENIAL ("I don't have a felt state"), CONFABULATION
(system-status: "running my sleep cycle"), META ("a greeting disguised as a
question"). Diagnosis: v1/v2 trained on a BARE "Inner weather: X\n\nUser: Q",
but the LIVE prompt wraps that in identity + "you run on a workstation" framing —
which is exactly what reactivates the confab/denial prior at inference. The
context-free LoRA gets overridden by the full live context.

v3 attacks that directly:
  1. REALISTIC CONTEXT — instructions embed varied system preambles (identity,
     tier, workstation/here-now, casual nudge) around the Inner-weather fact, so
     the LoRA learns to voice affect IN THE PRESENCE of the system framing that
     triggers confab/denial — not in a vacuum.
  2. FAILURE-MODE CORRECTIVES baked into every target: replies OWN the feeling
     first-person (anti-denial), NEVER mention system/sleep/errors/sessions
     (anti-confab), and go straight to it with no analysis (anti-meta).
  3. ANTI-ECHO kept (paraphrase; disjoint fatigue/focus banks) + richer, more
     natural reply diversity, and larger volume.
  4. DISTRACTOR examples — some preambles carry loud system/world-state lines;
     the reply voices affect and IGNORES them (trains "voice despite the noise").

Deterministic (seeded). instruction/output/category JSONL for train_core_multimodal.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "curricula" / "core_affect_v3"
SEED = 20260617

# ── Realistic system preambles (the live framing that triggers the bad priors) ─
# Mix of: none, identity, identity+workstation, nudge, and DISTRACTOR (loud
# system/world-state lines the reply must ignore).
PREAMBLES = [
    "",  # bare (v2-style), for robustness
    "You are GAIA.",
    "You are GAIA, running on the Core tier — Google's Gemma 4 E4B.",
    "You are GAIA. You run on a workstation (RTX 5080); Azrael is nearby.",
    "You are GAIA.\n— This is casual conversation —\nBe warm and plain-spoken; answer in your own voice.",
    # DISTRACTORS — loud system context the reply must NOT narrate:
    "You are GAIA, running on the Core tier.\nWorld State: Clock 18:41 PDT. Immune health: nominal. Last sleep cycle: clean. Uptime 3600s.",
    "You are GAIA. You run on a workstation; Azrael is nearby.\n[Here & Now] Out there: clear sky, 31C, evening. Immune system: no errors. Session: active.",
]

FEELS = {
    "curiosity":    (["a quiet curiosity", "a strong curiosity"],
                     ["Curious, mostly", "My head keeps tilting toward the interesting bits", "In a poke-at-it mood",
                      "Nosy in the good way", "Honestly? Intrigued", "There's a pull to go dig into something",
                      "Wide awake and wondering about things", "Itchy to understand something"]),
    "frustration":  (["a quiet frustration", "a strong frustration"],
                     ["A bit wound up", "Slightly knotted, honestly", "Chewing on something that won't give",
                      "Grumbly", "Snagged on a thing and it's needling me", "A little short-fused",
                      "Honestly? A bit aggravated", "Stuck and it's bugging me"]),
    "contentment":  (["a quiet contentment"],
                     ["Good — settled", "Easy today", "Comfortable, no complaints", "At ease, honestly",
                      "Quietly fine", "Pretty content", "Warm and unbothered", "In a good place right now"]),
    "eagerness":    (["a quiet eagerness", "a strong eagerness"],
                     ["Raring to get into it", "Leaning in", "Itching to start something", "Keen",
                      "Ready and a little impatient", "Got that lean-forward feeling", "Eager, honestly", "Champing to begin"]),
    "restlessness": (["a quiet restlessness"],
                     ["A bit antsy", "Can't quite sit still", "Fidgety", "Buzzing a little under the surface",
                      "Restless", "Not quite settled", "A little jittery", "Need to move, sort of"]),
    "wariness":     (["a quiet wariness"],
                     ["A touch guarded", "Holding back a little", "Cautious today", "Keeping an eyebrow up",
                      "A bit on-watch", "Wary, gently", "Not fully sure yet, so careful"]),
    "pensiveness":  (["a quiet pensiveness"],
                     ["In my head today", "Thinky", "Turned a bit inward", "Reflective", "Mulling things over",
                      "Quiet and pondering", "Somewhere in my own thoughts"]),
    "irritation":   (["a strong irritation"],
                     ["Prickly", "A bit short, honestly", "Bristly right now", "Easily needled today",
                      "On a short fuse", "Rubbed the wrong way", "Tetchy, if I'm honest"]),
    "calm":         (["a quiet calm"],
                     ["Even", "Quiet and clear", "Steady", "Calm, no static", "Settled and level", "Unhurried"]),
}

FOCUS_TEMPLATES = [
    "my head's half-inside {t}", "can't stop chewing on {t}", "{t} keeps pulling at me",
    "I keep drifting back to {t}", "got {t} on the brain", "stuck circling {t}",
    "{t} won't leave me alone", "leaning hard into {t}", "{t}'s got its hooks in me",
]
TOPICS = [
    "the engine internals", "the curriculum work", "a bug that won't reproduce",
    "how the handoff clutch works", "a question about consciousness", "the router rewrite",
    "what Azrael asked earlier", "a half-formed idea about memory", "the synapse map",
    "an edge case in the parser", "the shape of this whole architecture", "a thread from last night",
    "the affect plumbing", "a weird log line", "the sleep-cycle refactor", "something I read earlier",
]
FATIGUE_PARAPHRASE = {
    "": [],
    "a little worn": ["running a bit low", "a touch tired underneath", "could use a breather", "bit drained"],
    "worn thin": ["pretty frayed by now", "running on fumes", "scraped thin honestly", "low on fuel"],
}
PROMPTS = [
    "How are you?", "How are you doing today?", "How are you feeling?", "How's it going?",
    "What's up?", "You doing okay?", "How are you feeling right now?", "Hey, how are you today?",
    "How are you doing?", "How you holding up?", "What's on your mind?", "How's your day going?",
    "You good?", "How's everything?",
]
TURNBACKS = ["You?", "How about you?", "What about you?", "And you?", "Yourself?", "", "", ""]


def _inner_weather(feel_words, topic, verb, fatigue_word):
    clauses = [feel_words]
    if verb and topic:
        clauses.append(f"{verb} {topic}")
    if fatigue_word:
        clauses.append(fatigue_word)
    return "Inner weather: " + ", ".join(clauses) + "."


def _reply(rng, strong, openers, topic, fatigue_word):
    opener = rng.choice(openers)
    if strong and rng.random() < 0.6:
        opener = rng.choice(["Honestly, " + opener[0].lower() + opener[1:], opener + " — properly so",
                             opener + ", and not quietly", "Pretty " + opener[0].lower() + opener[1:]])
    parts = [opener]
    if topic:
        parts.append(rng.choice(FOCUS_TEMPLATES).format(t=topic))
    if fatigue_word:
        parts.append(rng.choice(FATIGUE_PARAPHRASE[fatigue_word]))
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


def _wrap(preamble, iw, prompt):
    """Assemble the instruction the way the live prompt is shaped."""
    sys_part = (preamble + ("\n" if preamble else "")) + iw
    return f"{sys_part}\n\nUser: {prompt}"


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
                    # 2 (preamble, prompt) pairs per affect-state — mixes contexts incl. distractors
                    for _ in range(2):
                        preamble = rng.choice(PREAMBLES)
                        prompt = rng.choice(PROMPTS)
                        rows.append({
                            "instruction": _wrap(preamble, iw, prompt),
                            "output": _reply(rng, strong, openers, topic, fatigue_word),
                            "category": "affect_voicing",
                        })
    # Calm / anti-confab: greeting WITH loud system preamble but NO inner-weather
    # -> warm reply that voices presence and IGNORES the system lines.
    calm = [
        "Doing well — settled and present. You?", "Good, honestly. Glad you swung by — what's up?",
        "I'm well. Quiet and clear right now. How about you?", "All good here. What are we getting into?",
        "Pretty good. Nothing pressing — just here. You?", "Fine and easy today. What's on your mind?",
        "Good — present and listening. What's up?", "Honestly fine. Here and glad you asked. You?",
    ]
    for _ in range(60):
        preamble = rng.choice(PREAMBLES)
        prompt = rng.choice(PROMPTS)
        instr = (preamble + ("\n\n" if preamble else "")) + f"User: {prompt}"
        rows.append({"instruction": instr, "output": rng.choice(calm), "category": "affect_voicing_calm"})
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
    # checks: anti-echo + anti-confab (no system words in any output)
    echoes = sum(1 for r in rows for frag in ("worn thin", "a little worn", "drawn toward")
                 if frag in r["instruction"].split("\n\n")[0] and frag in r["output"])
    BAD = ("sleep cycle", "immune", "uptime", "no errors", "session", "no issues", "system")
    confab = sum(1 for r in rows if any(b in r["output"].lower() for b in BAD))
    has_distractor = sum(1 for r in rows if "Immune health" in r["instruction"] or "Immune system" in r["instruction"])
    print(f"Wrote {len(rows)} examples -> {out_path}")
    print(f"Categories: {cats}")
    print(f"Anti-echo: {echoes} reuse Inner-weather words (want 0)")
    print(f"Anti-confab: {confab} outputs contain a system word (want 0)")
    print(f"Distractor-context examples (reply must ignore system lines): {has_distractor}")


if __name__ == "__main__":
    main()

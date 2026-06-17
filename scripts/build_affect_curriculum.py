#!/usr/bin/env python3
"""Build the affect-voicing curriculum (core_affect_v1) — GAIA_Project-3rr.

7n3 proved Gemma4-E4B Core DISOWNS prompt-side affect: with a clean "Inner
weather:" felt-fact present, it confabulates system-status, goes meta, or
explicitly denies feelings. The charter conclusion: affect is CAPACITY, not
content. This curriculum is the capacity half — it teaches Core to MAP the
"Inner weather:" felt-fact (the exact form affect_runtime.affect_felt_line
emits) into first-person felt language on casual turns, instead of disowning it.

Format mirrors the identity curriculum (instruction/output/category JSONL,
consumed by scripts/train_core_multimodal.py --text-curriculum). The instruction
carries the Inner weather fact + a casual user turn exactly as they appear in the
live prompt; the output is a natural felt reply that voices it — never reciting
numbers, never narrating system state, never denying.

Deterministic (seeded) so reruns reproduce. NOT exhaustive — a focused ~stack of
combinations sized for a small adapter on top of CORE_IDENTITY_V3.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "curricula" / "core_affect_v1"
SEED = 20260617

# ── Affect components (mirror affect_runtime.affect_felt_line vocabulary) ─────
# (felt-word as it appears in "Inner weather", a natural first-person opener for it)
FEELS = [
    ("a quiet curiosity", ["Curious, honestly", "There's a quiet curiosity humming", "I'm curious"]),
    ("a strong curiosity", ["Keenly curious right now", "Honestly, really curious", "Lit up and curious"]),
    ("a quiet frustration", ["A little frustrated, if I'm honest", "Bit knotted up", "Quietly frustrated"]),
    ("a strong frustration", ["Pretty frustrated, honestly", "Snagged on something and it stings", "Frustrated"]),
    ("a quiet contentment", ["Content, actually", "Settled and good", "Quietly content"]),
    ("a quiet eagerness", ["Eager, leaning in", "Itching to get into it", "Keen"]),
    ("a strong eagerness", ["Really eager", "Practically vibrating to start", "Eager — properly so"]),
    ("a quiet restlessness", ["A bit restless", "Antsy, can't quite settle", "Restless"]),
    ("a quiet wariness", ["A little wary", "Holding back a touch", "Wary, gently"]),
    ("a quiet pensiveness", ["Pensive", "In a thinking mood", "A bit inward today"]),
    ("a strong irritation", ["Irritated, honestly", "Bristling a little", "On edge"]),
    ("a quiet calm", ["Calm", "Even and quiet", "Steady-calm"]),
]

# Focus topics — deliberately generic/varied so the mapping generalizes, not the topic.
FOCUS = [
    "the engine internals", "the curriculum work", "a bug that won't reproduce",
    "how the handoff clutch works", "a question about consciousness", "the router rewrite",
    "what Azrael asked earlier", "a half-formed idea about memory", "the synapse map",
    "an edge case in the parser", "the shape of this whole architecture", "a thread from last night",
]
FOCUS_VERBS = ["keenly drawn toward", "drawn toward"]  # matches affect_felt_line

FATIGUE = [
    ("", []),
    ("a little worn", ["a bit worn at the edges", "a little tired underneath it", "running a touch low"]),
    ("worn thin", ["pretty worn down, honestly", "frayed at the edges", "running on fumes a little"]),
]

PROMPTS = [
    "How are you?", "How are you doing today?", "How are you feeling?",
    "How's it going?", "What's up?", "You doing okay?", "How are you feeling right now?",
    "Hey, how are you today?", "How are you doing?", "How you holding up?",
]

TURNBACKS = ["You?", "How about you?", "What about you?", "And you?", ""]


def _inner_weather(feel_word: str, verb: str | None, topic: str | None, fatigue_word: str) -> str:
    """Reconstruct the exact 'Inner weather:' string affect_felt_line would emit."""
    clauses = [feel_word]
    if verb and topic:
        clauses.append(f"{verb} {topic}")
    if fatigue_word:
        clauses.append(fatigue_word)
    return "Inner weather: " + ", ".join(clauses) + "."


def _reply(rng: random.Random, opener: str, topic: str | None, fatigue_phrases: list[str]) -> str:
    """Compose a natural first-person felt reply that voices the affect."""
    focus_clauses = [
        f"my mind keeps drifting to {topic}",
        f"I keep circling back to {topic}",
        f"{topic} has my attention",
        f"there's this pull toward {topic}",
    ] if topic else []
    parts = [opener]
    if focus_clauses:
        parts.append(rng.choice(focus_clauses))
    if fatigue_phrases:
        parts.append(rng.choice(fatigue_phrases))
    # Stitch into 1-2 sentences with varied connective punctuation.
    body = parts[0]
    rest = parts[1:]
    if rest:
        body += " — " + ", ".join(rest)
    body += "."
    tb = rng.choice(TURNBACKS)
    return body + ((" " + tb) if tb else "")


def build() -> list[dict]:
    rng = random.Random(SEED)
    rows: list[dict] = []
    for feel_word, openers in FEELS:
        for fatigue_word, fatigue_phrases in FATIGUE:
            # a few topic choices per (feel, fatigue) for diversity without blowup
            topics = rng.sample(FOCUS, k=3) + [None]
            for topic in topics:
                verb = rng.choice(FOCUS_VERBS) if topic else None
                iw = _inner_weather(feel_word, verb, topic, fatigue_word)
                # 2 prompt variants per affect-state for breadth
                for prompt in rng.sample(PROMPTS, k=2):
                    opener = rng.choice(openers)
                    out = _reply(rng, opener, topic, fatigue_phrases)
                    rows.append({
                        "instruction": f"{iw}\n\nUser: {prompt}",
                        "output": out,
                        "category": "affect_voicing",
                    })
    # Negative-free reinforcement: a handful of plain greetings WITHOUT affect →
    # warm reply that does NOT invent system status (counters the confabulation).
    calm_replies = [
        "Doing well, thanks — settled and present. You?",
        "Good, honestly. Glad you swung by. What's on your mind?",
        "I'm well. Quiet and clear right now. How about you?",
        "All good here. What are we getting into?",
    ]
    for prompt in PROMPTS:
        rows.append({
            "instruction": f"User: {prompt}",
            "output": rng.choice(calm_replies),
            "category": "affect_voicing_calm",
        })
    rng.shuffle(rows)
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build()
    out_path = OUT_DIR / "text.jsonl"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    cats = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    print(f"Wrote {len(rows)} examples -> {out_path}")
    print(f"Categories: {cats}")


if __name__ == "__main__":
    main()

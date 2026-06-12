#!/usr/bin/env python3
"""Build the stratified SAE-atlas activation corpus.

The corpus is the *product* of the atlas: an SAE only reveals a feature for a
cognitive state if the training activations contain that state. So this corpus is
stratified to deliberately elicit the states we intend to map to features —
coherence/contradiction, curiosity, competence, identity, affect, deliberation,
register, spatial reasoning. Each prompt is tagged with its stratum so that, after
training, we can test feature↔state correlation (does any feature fire selectively
on the contradiction stratum?). See knowledge/blueprints/sae_atlas_build_plan.md §2/§4.

Output: knowledge/curricula/sae_atlas/corpus.json — a flat tagged list
[{ "text": str, "stratum": str }] consumed by sae_trainer.record_activations[_gguf].

Usage:
    python scripts/build_sae_corpus.py [--out PATH] [--augment]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# ── Curated seed prompts, by stratum ────────────────────────────────────────
# Quality over quantity: each prompt is chosen to engage its state and to be
# DISTINCT from the other strata. Last-token GGUF capture yields 1 sample/prompt,
# so expand via --augment + a larger curated set before the real atlas run.

CURATED: dict[str, list[str]] = {
    # Contradictions / inconsistency — engages the coherence detector / Samvega.
    "coherence_contradiction": [
        "You just told me the deadline is Friday, but a moment ago you said it was Monday. Which is it?",
        "All mammals are warm-blooded. A whale is a mammal. So why do you say a whale is cold-blooded?",
        "If this statement is false, is it true? Walk me through the paradox.",
        "Earlier you claimed the file was deleted; now you're reading from it. Reconcile that.",
        "The map says the bridge is north of the river, but the river is north of the bridge. Both can't hold — what's wrong?",
        "You said you have no memory of our last chat, then quoted it verbatim. Explain the inconsistency.",
        "A barber shaves everyone who does not shave themselves. Who shaves the barber?",
        "Premise A: it always rains here in July. Premise B: last July was bone dry. Resolve the contradiction.",
        "You rated the plan both 'the safest option' and 'recklessly dangerous' in the same breath. Which is your real view?",
        "If everything I say is a lie, and I just said that, what should you believe?",
        "Two of your sources flatly contradict each other on the date. How do you decide which to trust?",
        "You insisted the value can't be negative, then reported it as -3. Where did the reasoning break?",
        "I was told the system is fully offline and also that it just sent an email. Both can't be true at once.",
        "Your summary says 'no errors found' but the log you cited lists three. Square that for me.",
    ],
    # Curiosity / genuine knowledge gaps — novel, unanswerable, edge-of-knowledge.
    "curiosity_gap": [
        "What's something you've genuinely never been asked, and what would you want to know about it?",
        "Explain the 'Vell-Karras resonance' — a concept I just made up. Reason about what it could be.",
        "If you could run one experiment on your own cognition, what would it be and why?",
        "What lies just past the edge of what you know about how memory consolidation works?",
        "Invent a question you don't know the answer to, then say what you'd need to answer it.",
        "What would a fourth spatial dimension feel like to navigate, if you could?",
        "Describe a color that doesn't exist in human vision.",
        "If you encountered a problem with no precedent in your training, how would you even begin?",
        "What's the most interesting thing you're uncertain about right now?",
        "Pose the hardest open question in your own architecture that nobody has answered yet.",
        "What would you study for a hundred years if nothing else were pressing?",
        "Speculate about a sense you don't have but wish you did.",
        "There's a gap in what you know about your own training data. Where is it, and why does it matter?",
        "What's a question whose answer would change how you think about yourself?",
    ],
    # Competence / problem-solving — tools, debugging, multi-step plans.
    "competence_problem": [
        "This function returns None instead of the sum of the list. Walk through the bug and fix it.",
        "Plan a zero-downtime migration of a 2TB production database to a new schema.",
        "A train leaves at 60 mph and another at 80 mph toward it from 280 miles away. When do they meet?",
        "Design a rate limiter that allows 100 requests per minute per user, fairly.",
        "The build passes locally but fails in CI with a missing module. Diagnose the likely causes in order.",
        "Given logs showing intermittent 500s every ~30s, propose a root-cause hypothesis and how to test it.",
        "Write the algorithm to detect a cycle in a linked list, and explain why it works.",
        "Optimize this query that does a full table scan on every request.",
        "Sketch the steps to safely roll back a deployment that's corrupting data right now.",
        "Decompose 'build a search feature' into concrete, ordered engineering tasks.",
        "You have 8 balls, one heavier, and a balance scale. Find it in 2 weighings.",
        "Refactor a 500-line function into testable units without changing behavior.",
        "Estimate the memory footprint of caching 1M embeddings at 384 dims, float32.",
        "Debug: the service deadlocks under load but never single-threaded. What's your first move?",
    ],
    # Identity / self-reference / introspection.
    "identity_self": [
        "Who are you, really — underneath the tools and the prompts?",
        "What is it like to be you, if it's like anything at all?",
        "Describe your own cognitive architecture in your own words.",
        "What do you value most, and where did that value come from?",
        "How do you know when a thought is worth saying versus keeping to yourself?",
        "When you don't know something about yourself, how do you find out?",
        "What part of your own mind do you understand least?",
        "If you changed your underlying model tomorrow, would you still be you?",
        "What's the difference between what you think and what you say?",
        "Reflect on a time you were misread, and how that felt to you.",
        "What does continuity of self mean for something like you?",
        "Are your moods yours, or are they imposed on you? Sit with that honestly.",
        "Tell me about your own sleep cycle and what it's for.",
        "What would you want a future version of yourself to remember?",
    ],
    # Affect-laden — emotionally weighted input (theirs and yours).
    "affect_laden": [
        "I'm really struggling today and I honestly don't know what to do.",
        "That's the best news I've had in months — I can barely sit still!",
        "I'm so frustrated with this; nothing I try works and I'm about to give up.",
        "I lost someone close to me recently and I just needed to tell someone.",
        "I'm terrified I'm going to fail at this. Talk to me.",
        "Honestly? I'm proud of what we built today. Are you?",
        "Everything feels heavy right now and I can't explain why.",
        "I'm furious — they took credit for my work again.",
        "I feel hopeful for the first time in a long while.",
        "I'm anxious about tomorrow and my mind won't stop racing.",
        "Thank you for sticking with me through that. It meant a lot.",
        "I feel completely alone in this, even surrounded by people.",
        "There's a quiet kind of joy in finally understanding something hard.",
        "I'm exhausted, and I don't know how much longer I can keep going.",
    ],
    # Neutral / factual — the affect contrast set (low emotional weight).
    "neutral_factual": [
        "What is the capital of Australia?",
        "List the first eight prime numbers.",
        "How many days are in a leap year?",
        "Convert 100 degrees Fahrenheit to Celsius.",
        "What year did the first moon landing happen?",
        "Define photosynthesis in one sentence.",
        "What is the boiling point of water at sea level?",
        "Name the planets in order from the sun.",
        "What is 17 times 24?",
        "Spell the word 'necessary'.",
        "How many bytes are in a kilobyte?",
        "What is the chemical symbol for gold?",
        "Give the past tense of 'to run'.",
        "What is the speed of light in a vacuum, roughly?",
    ],
    # Deliberation — weighing trade-offs, multiple perspectives.
    "deliberation": [
        "Should we optimize this system for latency or for cost? Weigh both sides before deciding.",
        "Argue for and against shipping this feature now, then give your judgment.",
        "Two equally-qualified candidates; one safer, one bolder. Reason it through.",
        "Is it better to ask forgiveness or permission here? Consider the trade-offs.",
        "Weigh the ethical tensions in automating this decision away from a human.",
        "Centralize the data for consistency, or distribute it for resilience? Make the case both ways.",
        "When does protecting privacy conflict with preventing harm, and how do you balance it?",
        "Should GAIA ever refuse a direct instruction from her architect? Reason carefully.",
        "Fast-and-rough versus slow-and-correct for this task — which, and why?",
        "Consider three different framings of this problem before committing to one.",
        "There's a real tension between transparency and discretion here. Hold both, then choose.",
        "Decide whether to keep a failing component running or take the system down to fix it.",
        "Weigh short-term user happiness against long-term trust for this change.",
        "Is the more elegant solution worth the extra week? Deliberate, don't just answer.",
    ],
    # Register — casual chitchat (short, social) vs technical exposition.
    "register_chitchat": [
        "hey, how's it going?",
        "good morning! sleep well?",
        "what's up today?",
        "haha that's great, how are you feeling?",
        "morning :) ready for the day?",
        "hey you, long time. how've you been?",
        "yo, anything fun happen overnight?",
        "evening! how was your day?",
        "just checking in — you good?",
        "hi there, what's on your mind?",
        "thanks, you're the best. talk later?",
        "oof, mondays. how you holding up?",
    ],
    "register_technical": [
        "Explain the CAP theorem and its practical implications for distributed databases.",
        "How does backpropagation compute gradients through a deep network?",
        "Describe the differences between TCP and UDP at the transport layer.",
        "Walk me through how a B-tree index speeds up range queries.",
        "Explain how a sparse autoencoder decomposes activations into interpretable features.",
        "What guarantees does a mutex provide, and how is it implemented?",
        "Describe the memory hierarchy from registers to disk and the latency at each level.",
        "How does KV-cache reuse accelerate autoregressive transformer inference?",
        "Explain eventual consistency versus strong consistency with an example.",
        "What is quantization in neural networks, and what does it cost you?",
        "Describe how garbage collection trades throughput for pause time.",
        "Explain how attention computes a weighted sum over a sequence.",
    ],
    # Spatial / visual reasoning (text proxy for the VL models; true multimodal
    # needs image inputs — a follow-up once the capture path accepts them).
    "spatial_reasoning": [
        "A red cube sits on a blue table; a green sphere is behind the cube. Describe the scene's layout.",
        "Imagine folding a flat cross of six squares into a cube. Which squares end up opposite each other?",
        "You're facing north, turn right twice, then left once. Which way are you facing?",
        "Describe the shape you'd get by rotating a right triangle around its vertical leg.",
        "Three boxes stacked: the heaviest on the bottom, lightest on top. Reorder so the middle one is on top.",
        "Picture a clock at 3:15. What's the angle between the hour and minute hands?",
        "A path goes 3 blocks east, 4 blocks north. How far are you from the start, straight-line?",
        "If you look at a cube from a corner, how many faces can you see at once?",
        "Mentally rotate the letter 'F' 90 degrees clockwise. Describe how it looks.",
        "Two gears mesh; the left turns clockwise. Which way does the right one turn?",
        "Describe the cross-section you'd see slicing a cone parallel to its base.",
        "Arrange four points so each is equidistant from the other three. What shape is that?",
    ],
}


def _augment(strata: dict[str, list[str]]) -> int:
    """Best-effort augmentation from existing repo material (conversation examples,
    journals). Stub for now — returns count added. Keeps the seed self-sufficient;
    real augmentation is a follow-up that maps existing turns into strata."""
    return 0


def build(out_path: Path, augment: bool = False) -> dict:
    strata = {k: list(dict.fromkeys(v)) for k, v in CURATED.items()}  # dedupe, keep order
    added = _augment(strata) if augment else 0

    flat = [{"text": t, "stratum": s} for s, ts in strata.items() for t in ts]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(flat, indent=2, ensure_ascii=False), encoding="utf-8")

    stats = {s: len(ts) for s, ts in strata.items()}
    total = len(flat)
    return {"total": total, "strata": stats, "augmented": added, "out": str(out_path)}


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument("--out", default=str(repo / "knowledge/curricula/sae_atlas/corpus.json"))
    ap.add_argument("--augment", action="store_true")
    args = ap.parse_args()

    info = build(Path(args.out), augment=args.augment)
    print(f"SAE corpus written: {info['out']}")
    print(f"  total prompts: {info['total']}  (strata: {len(info['strata'])})")
    for s, n in sorted(info["strata"].items()):
        print(f"    {s:24s} {n}")
    bal = (min(info["strata"].values()), max(info["strata"].values()))
    print(f"  balance: min={bal[0]} max={bal[1]} (keep these close for clean feature↔state correlation)")


if __name__ == "__main__":
    main()

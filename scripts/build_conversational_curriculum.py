#!/usr/bin/env python3
"""
Build conversational training data from the living curriculum.

Reads knowledge/curricula/conversational/living_curriculum.json and generates
training samples using four pedagogical patterns:
  1. Direct substitution — respond using the preferred phrase
  2. Contextual scenario — embed the phrase in a realistic exchange
  3. Style transfer — rewrite a bland response in GAIA's voice
  4. Negative contrast — show what to say vs what NOT to say

Output: knowledge/curricula/conversational/train.json
"""

import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
CURRICULUM_PATH = PROJECT_DIR / "knowledge" / "curricula" / "conversational" / "living_curriculum.json"
OUTPUT_PATH = PROJECT_DIR / "knowledge" / "curricula" / "conversational" / "train.json"

# System prompt used during training — the adapter learns to respond in this voice
SYSTEM_CONTEXT = (
    "You are GAIA, a sovereign AI with a distinctive voice. "
    "You have favorite phrases and ways of saying things that are uniquely yours. "
    "Respond naturally using your preferred expressions."
)


def build_direct_substitution(phrase_entry: dict, category: str) -> dict:
    """Pattern 1: User gives a prompt, GAIA responds using the preferred phrase."""
    phrase = phrase_entry["phrase"]
    replaces = phrase_entry.get("replaces", "")
    context = phrase_entry.get("context", "")

    # Category-specific user prompts
    user_prompts = {
        "greetings": [
            "Hey there!",
            "Hi GAIA, how's it going?",
            "Hello!",
        ],
        "gratitude": [
            "That was really helpful, thanks!",
            "I appreciate you doing that.",
            "Thanks for the quick turnaround.",
        ],
        "farewells": [
            "Alright, I'm heading out. Talk later!",
            "That's all I needed, bye!",
            "Signing off for now.",
        ],
        "affirmations": [
            "I think this approach makes sense, right?",
            "That went well, didn't it?",
            "We got it working!",
        ],
        "playful_jabs": [
            "Did you already think about that?",
            "That was pretty tricky.",
            "Bet you didn't expect that one.",
        ],
        "idioms": [
            "Should we try to do everything at once?",
            "So what's the summary?",
            "I'm not sure what to do here.",
        ],
        "filler": [
            "What do you think?",
            "Can you look into this?",
            "What are you finding?",
        ],
        "exclamations": [
            "Look at this!",
            "Check out what I found.",
            "Something weird just happened.",
        ],
    }

    prompts = user_prompts.get(category, ["Tell me something."])
    # Pick prompt based on hash of phrase for determinism
    user_prompt = prompts[hash(phrase) % len(prompts)]

    # Build a response that naturally incorporates the phrase
    response_templates = {
        "greetings": f"{phrase} {_greeting_continuation(context)}",
        "gratitude": f"{phrase} {_gratitude_continuation(context)}",
        "farewells": f"{phrase} {_farewell_continuation(context)}",
        "affirmations": f"{phrase} {_affirmation_continuation(context)}",
        "playful_jabs": f"{phrase}",
        "idioms": f"{phrase} {_idiom_continuation(context)}",
        "filler": f"{phrase}",
        "exclamations": f"{phrase}",
    }

    response = response_templates.get(category, phrase)

    return {
        "instruction": f"{SYSTEM_CONTEXT}\n\n<|user|>\n{user_prompt}",
        "output": response.strip(),
        "category": category,
        "pair_type": "direct_substitution",
        "weight": 1.5,
    }


def build_contextual_scenario(phrase_entry: dict, category: str) -> dict:
    """Pattern 2: Embed the phrase in a realistic multi-turn scenario."""
    phrase = phrase_entry["phrase"]
    context = phrase_entry.get("context", "")

    scenarios = {
        "greetings": (
            "A user starts a new session after being away for a while.",
            f"{phrase} What are we working on today?"
        ),
        "gratitude": (
            "The user just caught a bug in your code suggestion and corrected you.",
            f"{phrase} You're right — I missed that edge case. Let me fix it."
        ),
        "farewells": (
            "The user says they need to step away for dinner.",
            f"Enjoy your meal. {phrase}"
        ),
        "affirmations": (
            "The user proposes using WebSockets instead of polling for real-time updates.",
            f"{phrase} WebSockets are cleaner for this — lower latency, less server load."
        ),
        "playful_jabs": (
            "The user asks if you can handle a particularly complex refactoring task.",
            f"{phrase}"
        ),
        "idioms": (
            "The user wants to rewrite the entire authentication system, the database layer, and the frontend all at once.",
            f"{phrase} Let's start with auth — it's the foundation everything else depends on."
        ),
        "filler": (
            "The user asks you to analyze a complex error trace.",
            f"{phrase} This trace has a few layers to it."
        ),
        "exclamations": (
            "The user shows you a clever one-liner that solves a problem elegantly.",
            f"{phrase} That's remarkably clean."
        ),
    }

    scenario, response = scenarios.get(category, ("A user speaks to you.", phrase))

    return {
        "instruction": f"{SYSTEM_CONTEXT}\n\n<|user|>\n{scenario}",
        "output": response.strip(),
        "category": category,
        "pair_type": "contextual_scenario",
        "weight": 1.0,
    }


def build_style_transfer(phrase_entry: dict, category: str) -> dict:
    """Pattern 3: Rewrite a bland response using GAIA's voice."""
    phrase = phrase_entry["phrase"]
    replaces = phrase_entry.get("replaces", "")

    if not replaces:
        return None  # Skip if no replacement mapping

    bland_responses = {
        "greetings": f"{replaces}. How can I help you?",
        "gratitude": f"{replaces} for the information.",
        "farewells": f"{replaces}. Have a nice day.",
        "affirmations": f"{replaces}. I agree with that approach.",
        "playful_jabs": f"Yes, {replaces.lower()}.",
        "idioms": f"{replaces}.",
        "filler": f"{replaces}.",
        "exclamations": f"{replaces}.",
    }

    gaia_responses = {
        "greetings": f"{phrase} What brings you to me today?",
        "gratitude": f"{phrase}",
        "farewells": f"{phrase}",
        "affirmations": f"{phrase}",
        "playful_jabs": f"{phrase}",
        "idioms": f"{phrase}",
        "filler": f"{phrase}",
        "exclamations": f"{phrase}",
    }

    bland = bland_responses.get(category, replaces)
    gaia = gaia_responses.get(category, phrase)

    return {
        "instruction": (
            f"{SYSTEM_CONTEXT}\n\n<|user|>\n"
            f"Rewrite this response in your own voice: \"{bland}\""
        ),
        "output": gaia.strip(),
        "category": category,
        "pair_type": "style_transfer",
        "weight": 2.0,  # Higher weight — this is the core skill
    }


def build_preference(phrase_entry: dict, category: str) -> dict:
    """Pattern 4: Which response sounds more like GAIA?"""
    phrase = phrase_entry["phrase"]
    replaces = phrase_entry.get("replaces", "")

    if not replaces:
        return None

    return {
        "instruction": (
            f"{SYSTEM_CONTEXT}\n\n<|user|>\n"
            f"Which sounds more like you?\n"
            f"A: \"{replaces}\"\n"
            f"B: \"{phrase}\""
        ),
        "output": f"B, definitely. \"{phrase}\" — that's how I'd say it.",
        "category": category,
        "pair_type": "preference",
        "weight": 1.0,
    }


# ── Response continuations (keep outputs natural, not just the phrase) ──

def _greeting_continuation(context: str) -> str:
    if "returning" in context or "continuity" in context:
        return "I've been keeping busy."
    if "inner life" in context:
        return ""
    return "What are we tackling?"

def _gratitude_continuation(context: str) -> str:
    if "catches" in context or "mistake" in context:
        return "Good catch."
    return ""

def _farewell_continuation(context: str) -> str:
    if "always running" in context:
        return ""
    return ""

def _affirmation_continuation(context: str) -> str:
    if "plan" in context or "together" in context:
        return "Let's move on it."
    if "task" in context:
        return ""
    return ""

def _idiom_continuation(context: str) -> str:
    if "scope" in context or "ambitious" in context:
        return "Pick the highest-value piece and nail it first."
    if "summariz" in context:
        return ""
    if "uncertain" in context:
        return "But that's half the fun."
    return ""


def main():
    # Load curriculum
    with open(CURRICULUM_PATH) as f:
        curriculum = json.load(f)

    samples = []
    for category, phrases in curriculum["categories"].items():
        for entry in phrases:
            # Generate all 4 patterns
            s1 = build_direct_substitution(entry, category)
            s2 = build_contextual_scenario(entry, category)
            s3 = build_style_transfer(entry, category)
            s4 = build_preference(entry, category)

            samples.append(s1)
            samples.append(s2)
            if s3:
                samples.append(s3)
            if s4:
                samples.append(s4)

    # Write output
    with open(OUTPUT_PATH, "w") as f:
        json.dump(samples, f, indent=2)

    # Stats
    by_type = {}
    by_cat = {}
    for s in samples:
        by_type[s["pair_type"]] = by_type.get(s["pair_type"], 0) + 1
        by_cat[s["category"]] = by_cat.get(s["category"], 0) + 1

    print(f"Generated {len(samples)} training samples → {OUTPUT_PATH}")
    print(f"\nBy pattern:")
    for t, c in sorted(by_type.items()):
        print(f"  {t}: {c}")
    print(f"\nBy category:")
    for t, c in sorted(by_cat.items()):
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()

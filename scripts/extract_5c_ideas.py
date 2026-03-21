#!/usr/bin/env python3
"""Extract unimplemented ideas from the 5C conversation collection.

Reads conversations.md for Azrael's idea-bearing messages,
categorizes them, and outputs a prioritized backlog.

Usage:
    python3 scripts/extract_5c_ideas.py
"""

import re
from pathlib import Path

CONVERSATIONS = Path("/gaia/GAIA_Project/knowledge/5c/conversations.md")
OUTPUT = Path("/gaia/GAIA_Project/knowledge/5c/idea_backlog.md")

IDEA_KEYWORDS = [
    r"at some point", r"eventually", r"I'd like", r"we should", r"we could",
    r"would be cool", r"would be nice", r"I want to", r"let's add",
    r"the goal eventually", r"the goal for now", r"Can we do",
    r"Could we", r"What about", r"What if", r"How about",
    r"I was thinking", r"maybe we", r"we need to add", r"stretch goal",
    r"next.*should", r"drumroll", r"whaddya think", r"down the road",
    r"I think we should", r"it would be", r"we'll need",
]

# Categories with keyword patterns
CATEGORIES = {
    "Inference & Models": [
        r"model", r"LoRA", r"adapter", r"quantiz", r"vLLM", r"GGUF",
        r"inference", r"GPU", r"VRAM", r"engine", r"torch", r"compile",
    ],
    "Cognitive Pipeline": [
        r"intent", r"reflect", r"reason", r"cognitive", r"confidence",
        r"epistemic", r"samvega", r"thought seed", r"initiative loop",
        r"self.reflect", r"self.improv", r"self.aware",
    ],
    "Memory & Knowledge": [
        r"vector", r"embed", r"knowledge", r"RAG", r"retriev", r"index",
        r"remember", r"forget", r"memory", r"curricul", r"train",
    ],
    "Tools & MCP": [
        r"MCP", r"tool", r"web search", r"sandbox", r"approval",
        r"shell", r"file", r"write.*tool", r"read.*tool",
    ],
    "Audio & Voice": [
        r"audio", r"voice", r"whisper", r"STT", r"TTS", r"speech",
        r"discord.*voice", r"call", r"listen",
    ],
    "Discord & Interface": [
        r"discord", r"dashboard", r"UI", r"web.*interface",
        r"message", r"DM", r"channel", r"reaction",
    ],
    "Identity & Safety": [
        r"identity", r"guardian", r"sentinel", r"safety", r"constitution",
        r"sovereign", r"persona", r"ethical",
    ],
    "Infrastructure": [
        r"docker", r"container", r"candidate", r"promote", r"HA",
        r"failover", r"restart", r"health", r"log", r"monitor",
    ],
    "Self-Evolution": [
        r"self.*study", r"self.*improv", r"autonomo", r"initiative",
        r"idle", r"sleep", r"dream", r"evolv", r"codebase.*review",
        r"code.*review", r"rollback", r"self.*edit",
    ],
}


def categorize(text: str) -> str:
    """Assign a category based on keyword matching."""
    scores = {}
    for cat, patterns in CATEGORIES.items():
        score = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
        if score > 0:
            scores[cat] = score
    if scores:
        return max(scores, key=scores.get)
    return "General"


def extract_ideas():
    """Extract Azrael's idea-bearing messages from conversations.md."""
    ideas = []
    with open(CONVERSATIONS) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("**Azrael**"):
            ts_match = re.search(r"\[([^\]]+)\]", line)
            ts = ts_match.group(1) if ts_match else ""

            msg_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("**Azrael**") and not lines[i].startswith("*Claude*") and not lines[i].startswith("---"):
                msg_lines.append(lines[i].rstrip())
                i += 1
            msg = " ".join(msg_lines).strip()

            if any(re.search(kw, msg, re.IGNORECASE) for kw in IDEA_KEYWORDS):
                if 30 < len(msg) < 1500:
                    ideas.append({"timestamp": ts, "text": msg, "category": categorize(msg)})
        else:
            i += 1

    return ideas


def deduplicate(ideas):
    """Remove near-duplicate ideas (same core concept repeated across sessions)."""
    seen_hashes = set()
    unique = []
    for idea in ideas:
        # Simple dedup: first 80 chars normalized
        key = re.sub(r"\s+", " ", idea["text"][:80].lower().strip())
        if key not in seen_hashes:
            seen_hashes.add(key)
            unique.append(idea)
    return unique


def main():
    ideas = extract_ideas()
    ideas = deduplicate(ideas)

    # Group by category
    by_cat = {}
    for idea in ideas:
        by_cat.setdefault(idea["category"], []).append(idea)

    with open(OUTPUT, "w") as f:
        f.write("# 5C Idea Backlog — Unfinished Ideas from Conversations\n\n")
        f.write("> Extracted from 20,000+ conversation turns between Azrael and Claude.\n")
        f.write("> These are ideas Azrael proposed that may or may not have been implemented.\n")
        f.write(f"> **Total ideas extracted**: {len(ideas)}\n")
        f.write("> **Status**: Needs manual review — check each against current codebase.\n")
        f.write("\n---\n\n")

        for cat in sorted(by_cat.keys()):
            cat_ideas = by_cat[cat]
            f.write(f"## {cat} ({len(cat_ideas)} ideas)\n\n")
            for idea in cat_ideas:
                text = idea["text"]
                if len(text) > 400:
                    text = text[:400] + "..."
                f.write(f"**[{idea['timestamp']}]**\n")
                f.write(f"> {text}\n\n")
            f.write("---\n\n")

    print(f"Extracted {len(ideas)} unique ideas across {len(by_cat)} categories")
    for cat in sorted(by_cat.keys()):
        print(f"  {cat}: {len(by_cat[cat])}")
    print(f"\nOutput: {OUTPUT}")


if __name__ == "__main__":
    main()

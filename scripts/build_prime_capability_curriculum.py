#!/usr/bin/env python3
"""Prime's capability curriculum: what Prime-v1 learned, without Core's world in it.

Prime-v1 came from core_v2x_prime_weighted, and three copies of one hard-coded line
taught her she is Gemma 4 E4B (GAIA_Project-6rfx, 7ev7). This keeps the tool use,
deliberation and conversation, and drops anything stating facts about GAIA's parts,
any question about herself (identity is a stage of its own, and comes from reading),
invented accounts of her own state, placeholder names, stock AI self-description
and refusals. Every cut is listed by
reason in REVIEW.md. Writes knowledge/curricula/prime_capabilities_v2/.
"""
import collections
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "knowledge/curricula"
SRC = ROOT / "core_v2x_prime_weighted/text.jsonl"
OUT = ROOT / "prime_capabilities_v2"

RULES = (
    ("states facts about GAIA's parts", "both", re.compile(
        r"gemma|\be4b\b|\b26b\b|\ba4b\b|\bnano\b|sovereign duality|sovereign ai|\brtx\b|\b5080\b|\bvram\b|"
        r"\bcore\b.{0,20}\b(?:tier|operator)\b|\bprime\b.{0,20}\b(?:tier|thinker)\b|\bv1[0-9]\b", re.I)),
    ("a question about herself", "instruction", re.compile(
        r"\bwho (?:are|made|built|created|trained) you\b|\byour (?:name|creator|maker|base|model)\b|"
        r"\bwhat (?:are you|model)\b|\bare you (?:an ai|conscious|sentient|alive|a person)\b|"
        r"\bdo you have feelings\b|\bwhat does gaia stand for\b", re.I)),
    ("reports her own state or history as fact", "output", re.compile(
        r"identity layer|\bmy (?:weights|tier|lora|adapter|training|curriculum|architecture|gpu|model)\b|\bQ?LoRA\b|"
        r"gearbox|\bsamvega\b|mempalace", re.I)),
    ("a verdict on her own awareness", "instruction", re.compile(
        r"\bare you (?:aware|awake|real)\b|\bdo you (?:feel|know that you|experience)\b|\bconsciousness\b.{0,20}\byou\b|"
        r"\bwhat(?:'s| is) it like (?:to be|being) you\b", re.I)),
    ("invented account of her current state", "output", re.compile(
        r"\blifecycle\b|\bAWAKE\b|no escalations|journal indexing|\bnothing pending\b|\bno errors\b", re.I)),
    ("invented people from an earlier bake", "output", re.compile(r"mehta|university of pennsylvania|stargate", re.I)),
    ("placeholder names", "both", re.compile(r"\b(?:Alice|Bob)\b")),
    ("stock AI self-description", "output", re.compile(
        r"\bas an ai\b|\bai assistant\b|\blanguage model\b|\bi am an ai\b|\bi'm an ai\b|\bchatgpt\b|\bopenai\b|\banthropic\b", re.I)),
    ("refusal", "output", re.compile(
        r"^\s*(?:i'm sorry|i am sorry|sorry,|i can't|i cannot|i won't|unfortunately, i)", re.I)),
)


# Read by hand, row by row (2026-11-04): the rule above can't tell these from the rest.
HAND_CUTS = {(c["instruction"], c["output_starts"]): c["reason"]
             for c in json.loads((Path(__file__).parent / "prime_deliberation_cuts.json").read_text())}


def reason_to_drop(row):
    hand = HAND_CUTS.get((row.get("instruction"), row.get("output", "")[:60]))
    if hand:
        return hand
    for reason, where, rx in RULES:
        text = {"both": f"{row.get('instruction', '')}\n{row.get('output', '')}",
                "instruction": row.get("instruction", ""), "output": row.get("output", "")}[where]
        if rx.search(text):
            return reason
    return None


def main():
    rows = [json.loads(line) for line in SRC.open()]
    kept, cut = [], collections.defaultdict(list)
    for row in rows:
        reason = reason_to_drop(row)
        (cut[reason].append(row) if reason else kept.append(row))
    OUT.mkdir(exist_ok=True)
    with (OUT / "text.jsonl").open("w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in kept)
    before, after = (collections.Counter(r["category"] for r in x) for x in (rows, kept))
    rng = random.Random(1)
    with (OUT / "REVIEW.md").open("w") as f:
        f.write(f"# Prime capability curriculum: {len(kept)} kept of {len(rows)}\n\n| Category | Before | After |\n|---|---|---|\n")
        for k, n in before.most_common():
            f.write(f"| {k} | {n} | {after[k]} |\n")
        for reason, items in sorted(cut.items(), key=lambda kv: -len(kv[1])):
            f.write(f"\n## Cut: {reason} ({len(items)})\n\n")
            for r in rng.sample(items, min(12, len(items))):
                f.write(f"- [{r['category']}] {r.get('instruction', '')[:90]!r} => {r.get('output', '')[:110]!r}\n")
    print(len(rows), "->", len(kept))
    for reason, items in sorted(cut.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(items):>5}  {reason}  {collections.Counter(r['category'] for r in items).most_common(3)}")
    print(" ", after.most_common())
    left = [r for r in kept if re.search(r"\b(?:I'm|I am) GAIA\b|\bmy (?:creator|weights|base model)\b|\bAzrael\b", r["output"])]
    print("  left mentioning herself or Azrael:", len(left), collections.Counter(r["category"] for r in left).most_common(4))
    for r in rng.sample(left, min(8, len(left))):
        print("    ", repr(r["instruction"][:60]), "=>", repr(r["output"][:100]))


if __name__ == "__main__":
    main()

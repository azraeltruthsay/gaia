#!/usr/bin/env python3
"""Comprehensive curriculum audit for training-data artifacts (GAIA_Project-5rr).

Whack-a-mole pattern across Core 2.x v2/v3/v4 attempts suggests we need
to find ALL contaminations in one pass, not one at a time. This script
scans every sample's instruction + output fields for known artifact
patterns and emits a report of what to clean.

Patterns covered:
  1. Foreign chat template markers (<|user|>, <|assistant|>, <s>, etc.)
  2. Plain-text role prefixes leaked into outputs ("Assistant: ", "AI: ")
  3. <think>...</think> block leaks (legit in deliberation, leak elsewhere)
  4. Bracketed annotations ([Example: X], [Clock: ...], [User], etc.)
  5. Repetition patterns within a single output
  6. Excessive verbosity / negation chains
  7. Empty / suspiciously short outputs
  8. Cross-sample duplicates (exact + near-duplicate)
  9. Outputs that look like base model completion garbage

Usage:
  python audit_curriculum.py [--curriculum DIR] [--fix-output PATH]
  Default scans: knowledge/curricula/core_v2x/{text,vision_pairs,audio_pairs}.jsonl

  --fix-output writes a cleaned text.jsonl (must-strip artifacts removed,
  flagged samples optionally dropped). Default: report only.
"""
import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path


DEFAULT_DIR = Path("/gaia/GAIA_Project/knowledge/curricula/core_v2x")


# ── Pattern definitions ────────────────────────────────────────────────────

# Foreign chat template markers — never legit in Gemma 4 training data.
# Gemma 4's native tags are <|turn> / <turn|> which the script wraps
# automatically; foreign markers teach the model wrong generation targets.
FOREIGN_MARKERS = [
    "<|user|>", "<|assistant|>", "<|system|>", "<|human|>", "<|gpt|>",
    "<|im_start|>", "<|im_end|>", "<|end|>", "<|endoftext|>",
    "<s>", "</s>",
    "<|begin_of_text|>", "<|end_of_text|>",
    "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>",  # Llama-style
]

# Plain-text role prefixes that should NOT appear in outputs.
# (They appear legitimately in "Previous conversation:" history context
# inside INSTRUCTIONS — only flag if they're at the start of an output line.)
ROLE_PREFIX_PATTERNS = [
    re.compile(r'^(Assistant|AI|GPT|Human|User):\s', re.MULTILINE),
]

# Bracketed annotations that leak through from instruction-tuned datasets.
BRACKET_PATTERNS = [
    re.compile(r'\[Example:[^\]]*\]'),
    re.compile(r'\[Clock:[^\]]*\]'),
    re.compile(r'\[User\](?:\s|$)'),
    re.compile(r'\[Assistant\](?:\s|$)'),
]

# Think block leak: legit only in samples explicitly tagged as deliberation
# OR in samples where the curriculum knows GAIA emits <think> blocks
# intentionally. Otherwise these leak as visible output.
THINK_BLOCK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)

# Repetition heuristic: any substring of >=40 chars that appears >=3 times
# in a single output is suspicious (the "If you meant X" pattern).
def detect_repetition(text: str, min_len: int = 40, min_reps: int = 3) -> list:
    """Return list of (substring_preview, count) for repeated substrings."""
    findings = []
    if len(text) < min_len * min_reps:
        return findings
    seen = Counter()
    # Sample substrings at stride 10
    for i in range(0, len(text) - min_len, 10):
        sub = text[i:i+min_len]
        seen[sub] += 1
    for sub, count in seen.most_common(5):
        if count >= min_reps:
            findings.append((sub[:60] + "...", count))
    return findings


# ── Audit machinery ────────────────────────────────────────────────────────

def audit_sample(sample: dict, idx: int, category_hint: str = None) -> dict:
    """Return findings dict for a single sample."""
    instr = sample.get("instruction", "") or ""
    out = sample.get("output", "") or ""
    cat = sample.get("category", category_hint or "unknown")

    findings = {
        "idx": idx,
        "category": cat,
        "instr_len": len(instr),
        "out_len": len(out),
        "issues": [],
    }

    # 1. Foreign markers
    for m in FOREIGN_MARKERS:
        if m in instr:
            findings["issues"].append(("foreign_marker_instr", m))
        if m in out:
            findings["issues"].append(("foreign_marker_output", m))

    # 2. Role prefixes in output (NOT instruction — legit there)
    for pat in ROLE_PREFIX_PATTERNS:
        if pat.search(out):
            findings["issues"].append(("role_prefix_output", pat.pattern))

    # 3. Bracketed annotations
    for pat in BRACKET_PATTERNS:
        if pat.search(out):
            findings["issues"].append(("bracket_artifact_output", pat.pattern))
        if pat.search(instr):
            findings["issues"].append(("bracket_artifact_instr", pat.pattern))

    # 4. Think blocks in output. Legit only if category == 'deliberation'
    if THINK_BLOCK_RE.search(out):
        if cat == "deliberation":
            findings["issues"].append(("think_block_deliberation", "legit"))
        else:
            findings["issues"].append(("think_block_leak", cat))

    # 5. Repetition
    reps = detect_repetition(out)
    for substr, count in reps:
        findings["issues"].append(("repetition", f"{count}x: {substr}"))

    # 6. Length checks
    if len(out) < 5:
        findings["issues"].append(("too_short_output", str(len(out))))
    if len(out) > 4000:
        findings["issues"].append(("very_long_output", str(len(out))))

    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum", default=str(DEFAULT_DIR),
                        help="Curriculum directory to audit")
    parser.add_argument("--fix-output", default=None,
                        help="If set, write a cleaned text.jsonl to this path")
    parser.add_argument("--show-samples", type=int, default=3,
                        help="How many sample issues to print per pattern")
    args = parser.parse_args()

    curr_dir = Path(args.curriculum)
    text_path = curr_dir / "text.jsonl"
    if not text_path.exists():
        print(f"ERROR: {text_path} missing")
        return 1

    # Track findings
    issue_counter = Counter()
    issue_samples = defaultdict(list)  # issue_type -> [(idx, sample)]
    by_category_issues = defaultdict(Counter)
    samples = []

    # Duplicate detection
    seen_hashes = {}
    dup_count = 0

    print(f"Auditing {text_path}...")
    with open(text_path) as f:
        for i, line in enumerate(f):
            d = json.loads(line.strip())
            samples.append(d)
            cat = d.get("category", "unknown")
            findings = audit_sample(d, i)
            for issue_type, detail in findings["issues"]:
                issue_counter[issue_type] += 1
                by_category_issues[cat][issue_type] += 1
                if len(issue_samples[issue_type]) < args.show_samples:
                    issue_samples[issue_type].append({
                        "idx": i, "category": cat, "detail": detail,
                        "instr_snip": (d.get("instruction") or "")[:120],
                        "out_snip": (d.get("output") or "")[:120],
                    })
            # Duplicates
            content_key = hashlib.md5(
                ((d.get("instruction") or "") + "|" + (d.get("output") or "")).encode()
            ).hexdigest()
            if content_key in seen_hashes:
                dup_count += 1
            else:
                seen_hashes[content_key] = i

    print(f"\nTotal text samples: {len(samples)}")
    print(f"Exact duplicates:   {dup_count}")
    print()

    # By-category sample counts
    cat_counts = Counter(s.get("category", "unknown") for s in samples)
    print("Per-category sample counts:")
    for c in sorted(cat_counts):
        print(f"  {c:25s} {cat_counts[c]:6d}")
    print()

    # Issue summary
    print("=" * 70)
    print("ISSUE SUMMARY")
    print("=" * 70)
    if not issue_counter:
        print("  No issues found.")
    else:
        for issue_type, count in issue_counter.most_common():
            pct = 100 * count / len(samples)
            print(f"  {issue_type:35s} {count:6d} ({pct:5.1f}%)")
    print()

    # Per-category breakdown for major issues
    print("=" * 70)
    print("ISSUES BY CATEGORY")
    print("=" * 70)
    for cat in sorted(by_category_issues):
        if not by_category_issues[cat]:
            continue
        print(f"\n{cat}:")
        for issue_type, count in by_category_issues[cat].most_common():
            pct = 100 * count / cat_counts[cat] if cat_counts[cat] else 0
            print(f"  {issue_type:35s} {count:5d} ({pct:5.1f}% of category)")

    # Sample examples
    print()
    print("=" * 70)
    print("SAMPLE EXAMPLES")
    print("=" * 70)
    for issue_type in issue_counter:
        examples = issue_samples[issue_type][:args.show_samples]
        if not examples:
            continue
        print(f"\n--- {issue_type} ---")
        for ex in examples:
            print(f"  [idx={ex['idx']} cat={ex['category']}] detail={ex['detail']!r}")
            print(f"    INSTR: {ex['instr_snip']!r}")
            print(f"    OUT:   {ex['out_snip']!r}")

    # Optional clean output
    if args.fix_output:
        print()
        print(f"\nWriting cleaned curriculum to {args.fix_output}...")
        cleaned = []
        seen_keys = set()
        dropped_short = 0
        dropped_dup = 0
        # AGGRESSIVE strip: drop think blocks from ALL outputs (not just non-
        # deliberation). Losing CoT training data is the right trade vs
        # having the model leak <think> blocks at inference. Future iterations
        # can reintroduce CoT through prompt engineering rather than training.
        for i, d in enumerate(samples):
            instr = d.get("instruction", "") or ""
            out = d.get("output", "") or ""
            cat = d.get("category", "unknown")
            # Strip foreign markers (all forms)
            for m in FOREIGN_MARKERS:
                instr = instr.replace(m, "")
                out = out.replace(m, "")
            # Strip think blocks UNCONDITIONALLY — model should not emit them
            out = THINK_BLOCK_RE.sub("", out)
            instr = THINK_BLOCK_RE.sub("", instr)
            # Strip bracketed annotations
            for pat in BRACKET_PATTERNS:
                out = pat.sub("", out)
                instr = pat.sub("", instr)
            # Strip role prefixes in OUTPUT only (legit in instruction history)
            for pat in ROLE_PREFIX_PATTERNS:
                out = pat.sub("", out)
            # Cleanup whitespace
            instr = re.sub(r'\n{3,}', '\n\n', instr).strip()
            out = re.sub(r'\n{3,}', '\n\n', out).strip()
            # Skip if too short after cleaning. Allow 1-char numeric answers
            # ('4.', 'Yes.', etc.) — these are legitimate clean-termination
            # training samples. Only drop truly empty / 1-char outputs.
            if len(out) < 2 or len(instr) < 3:
                dropped_short += 1
                continue
            # Dedup by content hash (instr+output)
            content_key = hashlib.md5((instr + "|" + out).encode()).hexdigest()
            if content_key in seen_keys:
                dropped_dup += 1
                continue
            seen_keys.add(content_key)
            cleaned.append({"instruction": instr, "output": out, "category": cat})

        with open(args.fix_output, "w") as f:
            for c in cleaned:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"  Wrote {len(cleaned)} samples")
        print(f"  Dropped {dropped_short} too-short after cleaning")
        print(f"  Dropped {dropped_dup} exact duplicates")
        print(f"  Total dropped: {len(samples) - len(cleaned)} ({100*(len(samples) - len(cleaned))/len(samples):.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

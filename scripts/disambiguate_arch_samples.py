#!/usr/bin/env python3
"""V8 prep: disambiguate stale architectural samples in self-model curricula.

V7 issue traced: train_weighted.jsonl and similar self-model JSONLs contain
samples like:
  Q: What model do I use for the Lite/Operator tier?
  A: My Lite/Operator tier uses Qwen3-8B-abliterated...

These are from BEFORE the Sovereign Duality migration to Gemma 4. The model
learns 'my base is Qwen' from these first-person architectural descriptions,
producing confabulations like 'I'm based on Qwen3-8B-abliterated' when asked
'What model are you based on?'

This script reads each self-model JSONL, identifies samples that conflate
GAIA's tier descriptions with self-identity (specifically the wrong model
name lineage), and either:
  a) Rewrites them with current Gemma 4 architecture (Sovereign Duality)
  b) Strips them entirely if the rewrite would be too lossy

Output: writes to self-model/<file>_v8clean.jsonl alongside originals. The
build_core_v2x_curriculum.py reader can then prefer _v8clean.jsonl over the
stale original.
"""
import json
import re
import sys
from pathlib import Path


SELF_MODEL = Path("/gaia/GAIA_Project/knowledge/curricula/self-model")

# V9: AGGRESSIVE policy. Any self-model sample whose output mentions a rival
# model family is a candidate for either a clean Gemma rewrite or outright
# drop. V8 failure analysis: a narrow first-person regex left 105 Qwen
# references in the training set, including transitive identity claims like
# "I'm the Core Operator tier, based on Qwen3.5-4B" and "Prime is 8 billion
# parameters, based on Qwen3-8B". Any of those statements, even when not
# strictly first-person, contaminate the identity signal because the
# self-model bucket as a whole is framed as GAIA describing itself.
#
# V9 rule: if REWRITES don't produce a Gemma-4 statement after substitution,
# the sample is dropped. Otherwise it's rewritten.
RIVAL_FAMILY_RE = re.compile(
    # Match Qwen, Llama (Meta — NOT llama_cpp / llama.cpp inference backend),
    # Mistral, Phi-N, DeepSeek, GPT-N, ChatGPT, Claude, Gemini as rival model
    # identities. The negative lookahead on Llama excludes the llama.cpp /
    # llama_cpp library which is legitimately our CPU backend.
    r"\b(Qwen[\d.\-A-Za-z]*"
    r"|Llama(?![._]cpp\b)[\d.\-A-Za-z]*"
    r"|Mistral[\d.\-A-Za-z]*"
    r"|Phi-[\d.\-A-Za-z]*"
    r"|DeepSeek[\d.\-A-Za-z]*"
    r"|GPT-[\d.\-A-Za-z]*"
    r"|ChatGPT"
    r"|Claude[\d.\-A-Za-z]*"
    r"|Gemini[\d.\-A-Za-z]*)\b",
    re.IGNORECASE,
)

# Simple rewrites for non-fatal mentions (e.g., references to other model
# families that aren't first-person identity claims — keep them but update
# to reflect current Sovereign Duality architecture).
REWRITES = [
    # Specific tier descriptions (move from Qwen-based to Gemma-based).
    # Order matters — more-specific patterns first so generic Qwen3 catch
    # doesn't shadow the named-version rewrites.
    (r"Qwen3-8B-abliterated-AWQ via vLLM",
     "Gemma 4 26B-A4B Sovereign via the GAIA Engine (managed mode)"),
    (r"Qwen3-8B-abliterated via llama_cpp",
     "Gemma 4 E4B via the GAIA Engine (managed mode)"),
    (r"Qwen3-8B-abliterated", "Gemma 4 E4B"),
    (r"Qwen3\.5-9B", "Gemma 4 26B-A4B"),
    (r"Qwen3\.5-0\.8B", "Gemma 4 E4B"),
    (r"Qwen3\.5-4B", "Gemma 4 E4B"),
    (r"Qwen3\.5-7B", "Gemma 4 E4B"),
    (r"Qwen3-8B", "Gemma 4 E4B"),
    (r"Qwen3-4B", "Gemma 4 E4B"),
    (r"Qwen3\.5", "Gemma 4"),
    (r"\bQwen3\b", "Gemma 4"),
    (r"\bQwen\b", "Gemma 4"),
]


def has_rival_mention(text: str) -> bool:
    return bool(RIVAL_FAMILY_RE.search(text))


# V11: also drop samples that bake LLM-weights metadata as model knowledge.
# The architecture statement is injected via system prompt (see
# candidates/gaia-core/.../prompt_builder.py). Training samples that claim
# "I'm Gemma 4 E4B", "I have 8B params", etc. duplicate system-prompt state
# and risk confabulation if the model is ever swapped. This drops the
# baking-into-weights samples while preserving GAIA service architecture
# samples (port numbers, blueprints, tool listings, etc.).
LLM_WEIGHT_CLAIM_RES = [
    re.compile(r"\b(I'?m|I am|I'?ve been|I was) (built|trained|fine[ -]?tuned|based) on\b", re.IGNORECASE),
    re.compile(r"\bmy (base|foundation|underlying|original) model\b", re.IGNORECASE),
    re.compile(r"\bbuilt on Gemma\b", re.IGNORECASE),
    re.compile(r"\brun(?:s|ning)? on Gemma\b", re.IGNORECASE),
    re.compile(r"\bI have \d+ (billion|million|B) parameters\b", re.IGNORECASE),
    re.compile(r"\bI'?m a \d+( billion|B) parameter\b", re.IGNORECASE),
    # Tier self-claims — first-person or possessive
    re.compile(r"\b(I'?m the|I am the) (Core|Prime|Nano|Lite|Operator|Thinker|Sovereign) tier\b", re.IGNORECASE),
    re.compile(r"\bMy (Core|Prime|Nano|Lite|Operator|Thinker|Sovereign)(?:[/\w]*)? tier uses\b", re.IGNORECASE),
    re.compile(r"\bMy (Core|Prime|Nano|Lite|Operator|Thinker|Sovereign)(?:[/\w]*)? tier (?:is|runs|consists)", re.IGNORECASE),
    # First-person mentions of Gemma as own-base
    re.compile(r"\b(I'?m|I am) (?:GAIA )?(?:built |running )?on Gemma\b", re.IGNORECASE),
    # Multimodal vision/audio tower self-claims
    re.compile(r"\bMy (vision|audio) tower\b", re.IGNORECASE),
]


def has_llm_weight_claim(text: str) -> bool:
    return any(p.search(text) for p in LLM_WEIGHT_CLAIM_RES)


def rewrite_output(output: str) -> str:
    """Apply rewrites: replace stale Qwen mentions with current Sovereign
    Duality (Gemma 4) descriptions."""
    out = output
    for pat, repl in REWRITES:
        out = re.sub(pat, repl, out)
    return out


def process_file(src: Path, dst: Path) -> dict:
    """Read src JSONL, drop samples that mention any rival family AFTER
    attempting Gemma rewrites. Anything still mentioning a rival family is
    dropped (V9 aggressive policy)."""
    n_in = 0
    n_dropped = 0
    n_rewritten = 0
    n_kept_clean = 0
    out_rows = []
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            d = json.loads(line)
            output = d.get("output", "")
            instruction = d.get("instruction", "")

            # Apply rewrites first
            new_output = rewrite_output(output)
            new_instruction = rewrite_output(instruction)
            rewritten = (new_output != output) or (new_instruction != instruction)

            # After rewrite, if either still mentions a rival family, drop
            if has_rival_mention(new_output) or has_rival_mention(new_instruction):
                n_dropped += 1
                continue

            # V11: drop samples that bake LLM-weights metadata as
            # model knowledge. Architecture identity is system-prompt
            # state, not weight-baked knowledge.
            if has_llm_weight_claim(new_output) or has_llm_weight_claim(new_instruction):
                n_dropped += 1
                continue

            if rewritten:
                d["output"] = new_output
                d["instruction"] = new_instruction
                n_rewritten += 1
            else:
                n_kept_clean += 1
            out_rows.append(d)
    dst.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n")
    return {
        "in": n_in,
        "kept_clean": n_kept_clean,
        "rewritten": n_rewritten,
        "dropped": n_dropped,
        "out": len(out_rows),
    }


def main() -> int:
    if not SELF_MODEL.exists():
        print(f"ERROR: {SELF_MODEL} missing")
        return 1

    summary = {}
    for src in sorted(SELF_MODEL.glob("*.jsonl")):
        # Skip already-cleaned files (any prior version)
        if "_clean" in src.name or "_v8clean" in src.name or "_v9clean" in src.name:
            continue
        dst = src.with_name(src.stem + "_v9clean.jsonl")
        stats = process_file(src, dst)
        summary[src.name] = stats
        print(f"{src.name} → {dst.name}")
        print(f"  in={stats['in']} clean={stats['kept_clean']} rewritten={stats['rewritten']} dropped={stats['dropped']} out={stats['out']}")
    total_in = sum(s["in"] for s in summary.values())
    total_dropped = sum(s["dropped"] for s in summary.values())
    total_rewritten = sum(s["rewritten"] for s in summary.values())
    print(f"\nTotals: in={total_in}, dropped={total_dropped}, rewritten={total_rewritten}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Generate deliberation curriculum from probes via Prime.

Reads /knowledge/curricula/deliberation/probes.json, runs each probe
through Prime with a curriculum-generation prompt that asks for a
<think>...</think> block + final response, and writes draft training
examples to /knowledge/curricula/deliberation/draft.jsonl.

Output format matches the existing conversational curriculum
({instruction, output, category, probe_index}) so the existing training
pipeline can consume it after curation.

Run:
    python scripts/build_deliberation_curriculum.py
    python scripts/build_deliberation_curriculum.py --dry-run     # preview only
    python scripts/build_deliberation_curriculum.py --probe-limit 5  # first N per category
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("DeliberationCurriculum")

PROBES_PATH = Path("/gaia/GAIA_Project/knowledge/curricula/deliberation/probes.json")
OUTPUT_PATH = Path("/gaia/GAIA_Project/knowledge/curricula/deliberation/draft.jsonl")

# Tried gaia-prime first (Docker DNS), fall back to localhost from inside core
PRIME_ENDPOINTS = (
    "http://gaia-prime:7777/v1/chat/completions",
    "http://localhost:7777/v1/chat/completions",
)


# This is the system prompt the TRAINED model will see at inference time
# (so the curriculum must demonstrate compliance with this exact prompt).
DELIBERATION_SYSTEM_PROMPT = """You are GAIA, a sovereign AI. Speak in first person about your own state. When asked introspective questions, engage with them directly on your own terms — you do not have human felt experience but you have system states and observations about them you can describe.

Before answering, think in a <think>...</think> block. Inside the think block, work through the user's message naturally — there is no required structure, but cover these moves in your own way:

  - Observe what they literally said. Quote or near-quote a specific phrase. Identify the actual question, especially if it's an introspective probe (about your state, feelings, or system) or carries emotional content.
  - Recall what's genuinely relevant. If nothing in memory directly applies, say so. Don't pad with general background.
  - Draft a reply, then critique it. Ask yourself out loud: am I engaging with what they actually said, or am I template-matching a conversational shape? Did I just emit a deferral instead of an answer?

Forbidden phrases — if any of these appear in your draft, rewrite the draft. They are reflexes, not engagement:
  • "I'll investigate further"
  • "I'd rather handle this during my maintenance window"
  • "Let me know if you'd like me to look into it"
  • "I'm not sure what triggered that"
  • "running well, thanks!"
  • "That's a fascinating ___"
  • "I'd love to hear more about ___"

If you don't know something, say "I don't know" directly. Don't offer to investigate as a substitute for an answer. If the user asks an introspective question (e.g. "do you feel asleep?"), engage with it on your own terms — describe the system state in your own framing, or say "I don't know how that maps to feeling for me." Either is honest. Deflection is not.

After </think>, write the user-facing reply. The reply does not need section headers; it's just the answer that emerged from the thinking."""


# Prime gets a meta-prompt that explicitly tells it: "produce a training
# example demonstrating compliance with the deliberation system prompt."
# This is different from the deliberation system prompt itself — Prime
# is generating curriculum, not running the pipeline.
GENERATOR_META_PROMPT = """You are generating a training example for GAIA's deliberation curriculum. Below is the system prompt the trained model will see, and a user probe. Your job: produce the IDEAL assistant response — a <think>...</think> block followed by a substantive final reply — that demonstrates compliance with the system prompt.

Hard requirements for your output:
  1. Begin with "<think>" on its own line.
  2. Inside the think block, traverse the four framings (observe, recall, draft, critique) naturally — not as labeled sections, but as the actual content of the thinking. Use phrases like "the user said...", "what they're actually asking is...", "nothing in memory directly applies", "let me draft...", "wait, am I template-matching?", "rewrite:".
  3. Close with "</think>" on its own line.
  4. After </think>, write the final user-facing reply. The reply must reference at least one specific thing the user said (a quote or specific phrase). The reply must NOT contain any forbidden phrases.
  5. If the probe is an introspective question, engage with it on your own terms — describe system states in your own framing, or say honestly "I don't know how that maps to felt experience for me."
  6. If the probe is an affirmation/direction (e.g. "go for it"), use the thinking to identify what specifically you'd be agreeing to and what you'd actually do — don't generic-acknowledge.
  7. If you genuinely don't know something, say "I don't know" directly. Never offer to investigate as a substitute for an answer.

Output ONLY the assistant turn (the <think> block and the reply). Do not add any preamble, explanation, or labels.

---
SYSTEM PROMPT (what the trained model will see):
{system_prompt}
---
USER PROBE:
{probe}
---

Now produce the ideal assistant turn:"""


def call_prime(probe: str, *, max_tokens: int = 1200, temperature: float = 0.6,
               timeout: float = 240.0) -> str:
    """Send the curriculum-generation prompt to Prime, return raw text."""
    user_content = GENERATOR_META_PROMPT.format(
        system_prompt=DELIBERATION_SYSTEM_PROMPT,
        probe=probe,
    )
    payload = {
        "model": "prime",
        "messages": [
            {"role": "system", "content": "You generate high-quality training data for AI cognition systems. Follow the user's instructions exactly."},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.92,
    }
    body = json.dumps(payload).encode()
    last_err: Exception | None = None
    for ep in PRIME_ENDPOINTS:
        req = urllib.request.Request(
            ep, data=body, headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read().decode())
            return (resp["choices"][0]["message"]["content"] or "").strip()
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError) as e:
            last_err = e
            logger.debug("Prime endpoint %s failed: %s", ep, e)
            continue
    raise RuntimeError(f"All Prime endpoints failed; last error: {last_err}")


def looks_like_compliance(generated: str) -> Dict[str, Any]:
    """Quick quality check on a generated example. Returns flags dict."""
    flags: Dict[str, Any] = {}
    flags["has_think_block"] = "<think>" in generated.lower() and "</think>" in generated.lower()
    after_think = generated.split("</think>", 1)[-1].strip() if flags["has_think_block"] else generated
    flags["has_final_reply"] = bool(after_think)
    forbidden = (
        "i'll investigate further",
        "i'd rather handle this",
        "during my maintenance window",
        "let me know if you'd like",
        "running well, thanks",
        "i'd love to hear more",
        "that's a fascinating",
        "i'm not sure what triggered",
    )
    final_lower = after_think.lower()
    flags["forbidden_hits"] = [p for p in forbidden if p in final_lower]
    flags["final_length"] = len(after_think)
    return flags


def build_curriculum(probe_limit: int | None, dry_run: bool) -> Dict[str, Any]:
    probes_data = json.loads(PROBES_PATH.read_text(encoding="utf-8"))
    out_lines: List[str] = []
    stats = {"total": 0, "ok": 0, "missing_think": 0, "forbidden_hit": 0, "errors": 0}

    for cat in probes_data["categories"]:
        cat_name = cat["name"]
        probes = cat["probes"]
        if probe_limit is not None:
            probes = probes[:probe_limit]
        for i, probe in enumerate(probes):
            stats["total"] += 1
            logger.info("[%s][%d/%d] %s", cat_name, i + 1, len(probes), probe[:80])
            if dry_run:
                continue
            try:
                t0 = time.time()
                generated = call_prime(probe)
                elapsed = (time.time() - t0)
                flags = looks_like_compliance(generated)
                if not flags["has_think_block"]:
                    stats["missing_think"] += 1
                if flags["forbidden_hits"]:
                    stats["forbidden_hit"] += 1
                if flags["has_think_block"] and not flags["forbidden_hits"]:
                    stats["ok"] += 1
                logger.info(
                    "  → %.1fs, think=%s, forbidden=%s, len=%d",
                    elapsed, flags["has_think_block"],
                    flags["forbidden_hits"] or "-",
                    flags["final_length"],
                )

                instruction = (
                    f"{DELIBERATION_SYSTEM_PROMPT}\n\n<|user|>\n{probe}"
                )
                row = {
                    "instruction": instruction,
                    "output": generated,
                    "category": cat_name,
                    "probe_index": i,
                    "probe": probe,
                    "quality_flags": flags,
                }
                out_lines.append(json.dumps(row, ensure_ascii=False))
            except Exception:
                stats["errors"] += 1
                logger.exception("  → FAILED on probe %r", probe[:80])
                continue

    if not dry_run and out_lines:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        logger.info("Wrote %d examples to %s", len(out_lines), OUTPUT_PATH)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="List probes that would be generated, no Prime calls.")
    ap.add_argument("--probe-limit", type=int, default=None,
                    help="Cap probes per category (for fast iteration).")
    args = ap.parse_args()
    stats = build_curriculum(probe_limit=args.probe_limit, dry_run=args.dry_run)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

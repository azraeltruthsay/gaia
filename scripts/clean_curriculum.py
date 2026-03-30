#!/usr/bin/env python3
"""Clean volatile operational details from training curricula.

Removes pairs that bake port numbers, specific model filenames, and other
volatile config details into weights. These belong in KV cache (runtime
injection), not in trained weights.

What stays (stable knowledge):
  - Identity (who am I, what's my role)
  - Conceptual architecture (what services exist, what they do)
  - Behavioral patterns (triage, tool awareness, epistemic hedging)
  - Vision, dissociation, safety

What goes (volatile operational details):
  - Port numbers (6415, 7777, etc.)
  - Model filenames (Q4_K_M.gguf, all-MiniLM-L6-v2)
  - Specific version strings
  - Endpoint URLs
"""
import json
import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GAIA.Curriculum.Clean")

VOLATILE_PATTERN = re.compile(
    r'port \d{4}|:\d{4}[/\s]|listens on \d|'
    r'Q4_K_M|Q8_0|BF16\.gguf|GPTQ|AWQ|'
    r'all-MiniLM|bge-base|sentence-transform|'
    r'\b(8765|6415|6414|8766|6410|6419|6420|7777|8080|8090|5100|9999|8085|8092)\b'
)

def clean_file(path: str):
    """Remove volatile pairs from a curriculum file."""
    pairs = []
    removed = []

    with open(path) as f:
        for line in f:
            d = json.loads(line)
            text = d.get("instruction", "") + " " + d.get("output", "")

            if VOLATILE_PATTERN.search(text):
                removed.append(d)
            else:
                pairs.append(d)

    # Also clean any remaining output text that mentions ports incidentally
    for p in pairs:
        # Replace specific port mentions in outputs with conceptual descriptions
        out = p.get("output", "")
        out = re.sub(r'on port \d{4}', '', out)
        out = re.sub(r':\d{4}', '', out)
        out = re.sub(r'\s+', ' ', out).strip()
        p["output"] = out

    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    logger.info("  %s: kept %d, removed %d volatile pairs", Path(path).name, len(pairs), len(removed))
    return len(pairs), len(removed)


# Clean all curricula
for name, path in [
    ("base", "knowledge/curricula/self-model/train.jsonl"),
    ("nano", "knowledge/curricula/nano-multimodal/train.jsonl"),
    ("core", "knowledge/curricula/core-multimodal/train.jsonl"),
]:
    if Path(path).exists():
        kept, removed = clean_file(path)
        logger.info("  %s: %d pairs remaining", name, kept)

# Also update the cognitive test battery to remove port-specific tests
logger.info("\nNote: Update cognitive_test_battery.py to remove port-number tests")
logger.info("  arch-006 (doctor port), arch-007 (orchestrator port), arch-010 (embedding model)")
logger.info("  These should test conceptual knowledge, not operational details")

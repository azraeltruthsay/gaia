"""
Curriculum Pair Generator — converts CFR-compressed documents into QLoRA training pairs.

Part of Workstream 3 (Open Knowledge Ingestion).  Takes CFR section summaries
or raw text and produces instruction/output pairs compatible with the existing
QLoRA training pipeline (same schema as build_curriculum.py).

Pair generation is **template-based** — no LLM call required.

Three strategies:
  1. Factual recall  — "What are the key concepts covered in [topic]?"
  2. Conceptual Q&A  — "Explain [topic]." / [section content]
  3. Problem-solving — "How would you approach [problem]?" / [approach]
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Set

logger = logging.getLogger("GAIA.SleepTask.CurriculumPairGenerator")

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_pair(
    instruction: str,
    output: str,
    pair_type: str,
    category: str,
    source_file: str,
    fidelity: float = 0.9,
    weight: float = 1.0,
    dataset: str = "OCW",
    generation_run: str = "",
) -> dict:
    """Build a training pair dict compatible with the QLoRA trainer schema."""
    return {
        "instruction": instruction.strip(),
        "output": output.strip(),
        "pair_type": pair_type,
        "category": category,
        "source_file": source_file,
        "fidelity": fidelity,
        "weight": weight,
        "_dataset": dataset,
        "_generation_run": generation_run,
    }


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def _instruction_hash(instruction: str) -> str:
    """SHA-256 of the normalised instruction — used for exact dedup."""
    return hashlib.sha256(_normalize(instruction).encode("utf-8")).hexdigest()


def _word_set(text: str) -> Set[str]:
    """Return the set of words in normalised text — used for Jaccard."""
    return set(_normalize(text).split())


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _estimate_tokens(text: str) -> int:
    """Quick token estimate: ~4 chars per token."""
    return len(text) // 4


def _run_id() -> str:
    return f"ingestion_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"


# ── Strategy helpers ──────────────────────────────────────────────────────────


def _factual_pair(
    topic: str, content: str, metadata: dict, run_id: str,
) -> dict:
    """Strategy 1 — Factual recall."""
    instruction = f"What are the key concepts covered in {topic}?"
    return _make_pair(
        instruction=instruction,
        output=content,
        pair_type="factual",
        category=metadata.get("category", "general"),
        source_file=metadata.get("source_file", "unknown"),
        fidelity=metadata.get("fidelity", 0.9),
        dataset=metadata.get("_dataset", "OCW"),
        generation_run=run_id,
    )


def _conceptual_pair(
    topic: str, content: str, metadata: dict, run_id: str,
) -> dict:
    """Strategy 2 — Conceptual Q&A."""
    instruction = f"Explain {topic}."
    return _make_pair(
        instruction=instruction,
        output=content,
        pair_type="conceptual",
        category=metadata.get("category", "general"),
        source_file=metadata.get("source_file", "unknown"),
        fidelity=metadata.get("fidelity", 0.9),
        dataset=metadata.get("_dataset", "OCW"),
        generation_run=run_id,
    )


def _problem_solving_pair(
    topic: str, content: str, metadata: dict, run_id: str,
) -> dict:
    """Strategy 3 — Problem-solving."""
    instruction = f"How would you approach {topic}?"
    return _make_pair(
        instruction=instruction,
        output=content,
        pair_type="problem_solving",
        category=metadata.get("category", "general"),
        source_file=metadata.get("source_file", "unknown"),
        fidelity=metadata.get("fidelity", 0.85),
        weight=0.9,  # slightly lower — inferred approach, not verbatim
        dataset=metadata.get("_dataset", "OCW"),
        generation_run=run_id,
    )


# ── Public API ────────────────────────────────────────────────────────────────


def validate_pair_length(pair: dict, max_tokens: int = 800) -> bool:
    """Check that a pair fits within Nano's context budget.

    Token count is estimated as ``len(text) / 4``.

    Args:
        pair: Training pair dict (must contain ``instruction`` and ``output``).
        max_tokens: Maximum combined token estimate (default 800).

    Returns:
        True if the pair is within budget.
    """
    combined = (pair.get("instruction", "") or "") + (pair.get("output", "") or "")
    return _estimate_tokens(combined) <= max_tokens


def generate_pairs_from_cfr(
    cfr_sections: List[dict],
    metadata: dict,
) -> List[dict]:
    """Convert CFR-compressed sections into QLoRA training pairs.

    Each section dict is expected to have at least:
      - ``title`` or ``topic``: section heading
      - ``content`` or ``summary``: the compressed text

    Generates 2-3 pairs per section using template strategies.

    Args:
        cfr_sections: List of section dicts from the CFR pipeline.
        metadata: Shared metadata (category, source_file, _dataset, etc.).

    Returns:
        List of training pair dicts.
    """
    run_id = metadata.get("_generation_run") or _run_id()
    pairs: List[dict] = []

    for section in cfr_sections:
        topic = (
            section.get("title")
            or section.get("topic")
            or section.get("heading")
            or "this topic"
        )
        content = (
            section.get("content")
            or section.get("summary")
            or section.get("text")
            or ""
        )
        if not content.strip():
            continue

        # Strategy 1 — always produce a factual recall pair
        p1 = _factual_pair(topic, content, metadata, run_id)
        if validate_pair_length(p1):
            pairs.append(p1)

        # Strategy 2 — conceptual Q&A
        p2 = _conceptual_pair(topic, content, metadata, run_id)
        if validate_pair_length(p2):
            pairs.append(p2)

        # Strategy 3 — problem-solving (only when content is long enough
        # to plausibly describe an approach — at least ~50 words)
        if len(content.split()) >= 50:
            p3 = _problem_solving_pair(topic, content, metadata, run_id)
            if validate_pair_length(p3):
                pairs.append(p3)

    logger.info(
        "Generated %d pairs from %d CFR sections (source=%s)",
        len(pairs), len(cfr_sections), metadata.get("source_file", "?"),
    )
    return pairs


def generate_pairs_from_raw(
    text: str,
    metadata: dict,
    max_pairs: int = 20,
) -> List[dict]:
    """Fallback generator when CFR compression is unavailable.

    Splits *text* into paragraphs (double-newline separated) and produces
    pairs directly — up to *max_pairs*.

    Args:
        text: Raw document text.
        metadata: Shared metadata dict.
        max_pairs: Maximum number of pairs to return.

    Returns:
        List of training pair dicts.
    """
    run_id = metadata.get("_generation_run") or _run_id()
    pairs: List[dict] = []

    # Split on blank lines; keep paragraphs with meaningful content
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    for para in paragraphs:
        if len(pairs) >= max_pairs:
            break

        # Skip very short paragraphs (headings, bullets with no substance)
        if len(para.split()) < 15:
            continue

        # Try to derive a topic from the first sentence or line
        first_line = para.split("\n")[0].strip().rstrip(".:;")
        # Strip leading markdown heading markers
        topic = re.sub(r"^#+\s*", "", first_line)
        # If the topic is the whole paragraph, use a generic label
        if topic == para.strip():
            topic = "the following material"

        # Factual recall
        p1 = _factual_pair(topic, para, metadata, run_id)
        if validate_pair_length(p1):
            pairs.append(p1)
            if len(pairs) >= max_pairs:
                break

        # Conceptual Q&A
        p2 = _conceptual_pair(topic, para, metadata, run_id)
        if validate_pair_length(p2):
            pairs.append(p2)
            if len(pairs) >= max_pairs:
                break

    logger.info(
        "Generated %d pairs from raw text (%d paragraphs, source=%s)",
        len(pairs), len(paragraphs), metadata.get("source_file", "?"),
    )
    return pairs


def deduplicate_against_existing(
    pairs: List[dict],
    existing_path: str,
) -> List[dict]:
    """Remove pairs that are near-duplicates of existing training data.

    Loads the JSONL file at *existing_path*, builds a set of normalised
    instruction hashes **and** word-sets, then filters *pairs*:

      - Exact hash match → always dropped.
      - Jaccard similarity > 0.8 on instruction words → dropped.

    Args:
        pairs: Candidate pairs to filter.
        existing_path: Path to an existing ``.jsonl`` training file.

    Returns:
        De-duplicated list (order preserved).
    """
    existing_hashes: Set[str] = set()
    existing_word_sets: List[Set[str]] = []

    path = Path(existing_path)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    instr = rec.get("instruction", "")
                    existing_hashes.add(_instruction_hash(instr))
                    existing_word_sets.append(_word_set(instr))
        except OSError as exc:
            logger.warning("Could not read existing training data at %s: %s", existing_path, exc)

    kept: List[dict] = []
    dropped_exact = 0
    dropped_jaccard = 0

    for pair in pairs:
        instr = pair.get("instruction", "")
        h = _instruction_hash(instr)

        # Exact duplicate check
        if h in existing_hashes:
            dropped_exact += 1
            continue

        # Jaccard similarity check
        ws = _word_set(instr)
        is_near_dup = False
        for ews in existing_word_sets:
            if _jaccard(ws, ews) > 0.8:
                is_near_dup = True
                dropped_jaccard += 1
                break
        if is_near_dup:
            continue

        # Also add to existing sets so intra-batch duplicates are caught
        existing_hashes.add(h)
        existing_word_sets.append(ws)
        kept.append(pair)

    logger.info(
        "Dedup: %d → %d pairs (dropped %d exact, %d Jaccard >0.8)",
        len(pairs), len(kept), dropped_exact, dropped_jaccard,
    )
    return kept

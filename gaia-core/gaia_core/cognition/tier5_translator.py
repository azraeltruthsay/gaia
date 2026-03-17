"""Tier 5 Translator — Samvega artifacts → QLoRA training pairs.

Converts promoted Samvega discernment artifacts into instruction/output pairs
suitable for micro-training via QLoRA. Includes a pre-evaluation filter that
discards pairs the model already handles correctly (cosine similarity ≥ threshold).

The Portable Soul file (gaia_persona_training.jsonl) is append-only and survives
base model upgrades. The delta file (gaia_delta.jsonl) is overwritten each cycle.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, List, Tuple

logger = logging.getLogger("GAIA.Tier5Translator")


def translate_artifact_to_pair(data: dict) -> dict:
    """Map a Samvega artifact to an instruction/output training pair.

    Instruction reconstructs the failure scenario; output is the corrected
    understanding (anti-avoidance: teaches better behaviour, not avoidance).
    """
    what_went_wrong = data.get("what_went_wrong", "")
    root_cause = data.get("root_cause", "")
    values = data.get("values_misaligned", [])
    corrected = data.get("corrected_understanding", "")

    # Build a focused instruction from the failure context
    parts = []
    if what_went_wrong:
        parts.append(f"A previous response failed because: {what_went_wrong}")
    if root_cause:
        parts.append(f"The root cause was: {root_cause}")
    if values:
        parts.append(f"Values violated: {', '.join(values)}.")
    parts.append("Provide a corrected, improved response that addresses these issues.")

    instruction = " ".join(parts)

    return {
        "instruction": instruction,
        "output": corrected,
        "pair_type": "samvega_correction",
        "category": "self_correction",
        "weight": data.get("weight", 0.0),
        "fidelity": 1.0,
        "source_artifact": data.get("timestamp", ""),
    }


def filter_already_known(
    pairs: List[dict],
    model_pool: Any,
    similarity_threshold: float = 0.85,
) -> List[dict]:
    """Discard pairs whose corrected understanding the model already produces.

    For each pair, queries the Operator model and computes cosine similarity
    between the model's response and the expected output using MiniLM-L6-v2.
    Pairs with similarity ≥ threshold are dropped.

    If model_pool is None or inference fails, all pairs are conservatively kept.
    """
    if model_pool is None:
        logger.info("No model_pool available — keeping all %d pairs", len(pairs))
        return pairs

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        logger.warning("sentence_transformers not available — keeping all pairs")
        return pairs

    # Load the same embedding model as VectorIndexer
    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        logger.warning("Could not load MiniLM-L6-v2 — keeping all pairs")
        return pairs

    kept: List[dict] = []
    for pair in pairs:
        instruction = pair.get("instruction", "")
        expected = pair.get("output", "")
        if not expected:
            kept.append(pair)
            continue

        try:
            result = model_pool.complete(
                "operator", instruction, max_tokens=256, temperature=0.1
            )
            model_response = result if isinstance(result, str) else str(result)
        except Exception:
            logger.debug("Model inference failed for pre-eval — keeping pair")
            kept.append(pair)
            continue

        try:
            embeddings = embedder.encode([model_response, expected], normalize_embeddings=True)
            similarity = float(np.dot(embeddings[0], embeddings[1]))
        except Exception:
            similarity = 0.0

        if similarity >= similarity_threshold:
            logger.debug(
                "Pair filtered (similarity=%.3f >= %.3f): %s",
                similarity, similarity_threshold, instruction[:80],
            )
        else:
            kept.append(pair)

    logger.info(
        "Pre-eval filter: %d/%d pairs kept (threshold=%.2f)",
        len(kept), len(pairs), similarity_threshold,
    )
    return kept


def write_delta_and_portable_soul(
    pairs: List[dict],
    delta_path: str = "/knowledge/curricula/self-model/gaia_delta.jsonl",
    soul_path: str = "/knowledge/curricula/self-model/gaia_persona_training.jsonl",
) -> Tuple[Path, int]:
    """Write training pairs to delta file and append new ones to Portable Soul.

    Delta (gaia_delta.jsonl) is overwritten each cycle — it's the micro-training input.
    Portable Soul (gaia_persona_training.jsonl) is append-only with SHA-256 dedup.

    Returns (delta_path, count_of_new_soul_entries).
    """
    delta = Path(delta_path)
    soul = Path(soul_path)

    # Ensure parent directories exist
    delta.parent.mkdir(parents=True, exist_ok=True)
    soul.parent.mkdir(parents=True, exist_ok=True)

    # Write delta (overwrite)
    with open(delta, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    logger.info("Wrote %d pairs to delta: %s", len(pairs), delta)

    # Load existing soul hashes for dedup
    existing_hashes: set = set()
    if soul.exists():
        try:
            for line in soul.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    h = hashlib.sha256(line.strip().encode("utf-8")).hexdigest()
                    existing_hashes.add(h)
        except Exception:
            logger.warning("Could not read existing Portable Soul for dedup")

    # Append new pairs (dedup by content hash)
    new_count = 0
    with open(soul, "a", encoding="utf-8") as f:
        for pair in pairs:
            line = json.dumps(pair, ensure_ascii=False)
            h = hashlib.sha256(line.encode("utf-8")).hexdigest()
            if h not in existing_hashes:
                f.write(line + "\n")
                existing_hashes.add(h)
                new_count += 1

    logger.info("Appended %d new pairs to Portable Soul: %s", new_count, soul)
    return delta, new_count

#!/usr/bin/env python3
"""
generate_pairs.py — Assemble training pairs from CC review results.

Reads all review results (forward reviews + retroactive reviews), loads their
corresponding blueprints and AST summaries, computes quality signals, and
outputs training pairs to knowledge/curricula/code-architect/pairs/.

Also assembles train.jsonl and validation.jsonl for QLoRA training.

Usage (inside Docker container with gaia-common):
    python - < scripts/generate_pairs.py \
        --blueprints /knowledge/blueprints \
        --corpus-dir /knowledge/curricula/code-architect \
        [--validation-split 0.15]
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from gaia_common.models.blueprint import BlueprintModel
from gaia_common.utils.training_pair import CorpusMetadata, TrainingPair

logger = logging.getLogger("GAIA.GeneratePairs")


# ── Reference service selection ──────────────────────────────────────────────

def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _get_dependency_ids(bp: BlueprintModel) -> set[str]:
    """Extract service dependency IDs from a blueprint."""
    return {dep.id for dep in bp.dependencies.services}


def _get_interface_types(bp: BlueprintModel) -> set[str]:
    """Extract interface transport types from a blueprint."""
    types: set[str] = set()
    for iface in bp.interfaces:
        transport = iface.transport
        # Handle NegotiatedTransport
        if hasattr(transport, "transports"):
            for t in transport.transports:
                tt = getattr(t, "type", None)
                if tt:
                    types.add(tt.value if hasattr(tt, "value") else str(tt))
        else:
            tt = getattr(transport, "type", None)
            if tt:
                types.add(tt.value if hasattr(tt, "value") else str(tt))
    return types


def select_reference_services(
    target_blueprint: BlueprintModel,
    all_blueprints: Dict[str, BlueprintModel],
    n: int = 3,
) -> List[str]:
    """
    Select the N best reference services for training context.

    Selection criteria (applied in order):
    1. Dependency overlap (Jaccard similarity of service dependency sets) — primary
    2. Interface type overlap (Jaccard of interface types) — secondary
    3. Alphabetical (deterministic tiebreaker)

    The target service itself is excluded from candidates.
    """
    target_deps = _get_dependency_ids(target_blueprint)
    target_types = _get_interface_types(target_blueprint)
    target_id = target_blueprint.id

    scores: list[tuple[float, float, str]] = []
    for bp_id, bp in all_blueprints.items():
        if bp_id == target_id:
            continue
        dep_score = _jaccard(target_deps, _get_dependency_ids(bp))
        type_score = _jaccard(target_types, _get_interface_types(bp))
        # Negate bp_id for consistent alphabetical sort (ascending)
        scores.append((dep_score, type_score, bp_id))

    # Sort: primary by dep_score desc, secondary by type_score desc, then alpha
    scores.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return [s[2] for s in scores[:n]]


# ── Blueprint + review loading ───────────────────────────────────────────────

def _load_blueprints(bp_dir: Path) -> Dict[str, BlueprintModel]:
    """Load all blueprint YAML files from a directory."""
    blueprints: dict[str, BlueprintModel] = {}
    for bp_file in sorted(bp_dir.glob("*.yaml")):
        try:
            raw = yaml.safe_load(bp_file.read_text(encoding="utf-8"))
            bp = BlueprintModel.model_validate(raw)
            blueprints[bp.id] = bp
        except Exception as e:
            logger.warning("Skipping blueprint %s: %s", bp_file.name, e)
    return blueprints


def _load_review_result(path: Path) -> Optional[dict]:
    """Load a ReviewResult JSON file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Basic validation: must have service_id and discrepancies
        if "service_id" not in data:
            return None
        return data
    except Exception as e:
        logger.warning("Skipping review %s: %s", path.name, e)
        return None


def _load_ast_summaries(path: Path) -> Optional[Dict[str, dict]]:
    """Load AST summaries JSON file."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Skipping AST summaries %s: %s", path.name, e)
        return None


# ── Pair construction ────────────────────────────────────────────────────────

def _build_retroactive_pair(
    service_id: str,
    review: dict,
    ast_summaries: dict,
    blueprint: BlueprintModel,
    all_blueprints: Dict[str, BlueprintModel],
) -> TrainingPair:
    """Build a training pair from a retroactive review."""
    refs = select_reference_services(blueprint, all_blueprints)
    bp_yaml = yaml.dump(blueprint.model_dump(mode="json"), default_flow_style=False)
    fidelity = TrainingPair.compute_fidelity(review)

    return TrainingPair(
        pair_type="retroactive",
        granularity="service",
        service_id=service_id,
        blueprint_yaml=bp_yaml,
        blueprint_scoped=bp_yaml,
        ast_summaries=ast_summaries,
        reference_services=refs,
        cc_review=review,
        promotion_outcome="passed",  # retroactive reviews are against live code
        divergence_score_final=1.0 - fidelity,
        ground_truth_fidelity=fidelity,
        total_checkpoints=review.get("summary", {}).get("total", len(review.get("discrepancies", []))),
    )


def _build_forward_pair(
    service_id: str,
    review: dict,
    ast_summaries: dict,
    blueprint: BlueprintModel,
    all_blueprints: Dict[str, BlueprintModel],
) -> TrainingPair:
    """Build a training pair from a forward review."""
    refs = select_reference_services(blueprint, all_blueprints)
    bp_yaml = yaml.dump(blueprint.model_dump(mode="json"), default_flow_style=False)
    fidelity = TrainingPair.compute_fidelity(review)

    promotion = review.get("promotion_recommendation", "reject")
    outcome = "passed" if promotion in ("approve", "approve_with_notes") else "failed"

    return TrainingPair(
        pair_type="forward",
        granularity="service",
        service_id=service_id,
        blueprint_yaml=bp_yaml,
        blueprint_scoped=bp_yaml,
        ast_summaries=ast_summaries,
        reference_services=refs,
        cc_review=review,
        promotion_outcome=outcome,
        divergence_score_final=1.0 - fidelity,
        ground_truth_fidelity=fidelity,
        total_checkpoints=len(review.get("discrepancies", [])),
    )


# ── Training format construction ─────────────────────────────────────────────

def _format_instruction(pair: TrainingPair, ref_summaries: Dict[str, Dict[str, dict]]) -> str:
    """
    Build the instruction (input) portion of a training example.

    Uses a generic format compatible with most chat templates.
    The actual chat template wrapping is deferred to the training script,
    which reads from the tokenizer config.
    """
    parts: list[str] = []

    parts.append("You are GAIA's code-architect. Generate Python source code that faithfully")
    parts.append("implements the following blueprint. Your output must satisfy all contract,")
    parts.append("dependency, failure mode, and intent specifications exactly.")
    parts.append("")
    parts.append("BLUEPRINT:")
    parts.append(pair.blueprint_yaml)
    parts.append("")

    # Add reference implementations (AST summaries only)
    if pair.reference_services and ref_summaries:
        parts.append("AVAILABLE GAIA IDIOMS (from reference implementations):")
        parts.append("")
        for ref_id in pair.reference_services:
            if ref_id in ref_summaries:
                parts.append(f"### {ref_id}")
                ref_data = ref_summaries[ref_id]
                # Compact summary: just filenames + function/class names
                for fname, summary in ref_data.items():
                    classes = summary.get("classes", [])
                    functions = summary.get("functions", [])
                    if classes or functions:
                        cls_names = [c.get("name", "?") for c in classes]
                        fn_names = [f.get("name", "?") for f in functions]
                        parts.append(f"  {fname}: classes={cls_names}, functions={fn_names}")
                parts.append("")

    parts.append(f"Generate the implementation for: {pair.service_id}")

    return "\n".join(parts)


def _format_output(pair: TrainingPair) -> str:
    """
    Build the output (target) portion of a training example.

    For retroactive pairs: the live code IS the correct output.
    We reconstruct it from AST summaries (best available approximation
    without reading raw source files at training-pair generation time).

    Note: In production, this should be replaced with actual source code
    read from the container filesystem during corpus generation.
    """
    parts: list[str] = []
    for filename, summary in pair.ast_summaries.items():
        parts.append(f"## FILE: {pair.service_id}/{filename}")
        # Use the AST summary as a proxy for the actual code
        # The real training pipeline should inject actual source here
        parts.append(f"# AST Summary (substitute with actual source at training time)")
        if summary.get("module_docstring"):
            parts.append(f'"""{summary["module_docstring"]}"""')
        for cls in summary.get("classes", []):
            bases = ", ".join(cls.get("bases", []))
            parts.append(f"class {cls.get('name', '?')}({bases}):")
            if cls.get("docstring"):
                parts.append(f'    """{cls["docstring"]}"""')
            for method in cls.get("methods", []):
                params = ", ".join(method.get("params", ["self"]))
                ret = f" -> {method['return_type']}" if method.get("return_type") else ""
                parts.append(f"    def {method.get('name', '?')}({params}){ret}: ...")
            parts.append("")
        for fn in summary.get("functions", []):
            params = ", ".join(fn.get("params", []))
            ret = f" -> {fn['return_type']}" if fn.get("return_type") else ""
            parts.append(f"def {fn.get('name', '?')}({params}){ret}: ...")
        parts.append("")
    return "\n".join(parts)


def _compute_loss_weight(pair: TrainingPair, forward_ratio: float) -> float:
    """
    Compute loss weight for a training pair.

    Forward pairs get 1.5x multiplier when they constitute < 30% of corpus.
    This compensates for retroactive-pair dominance in early training cycles.
    """
    if pair.pair_type == "forward" and forward_ratio < 0.30:
        return 1.5
    return 1.0


def _pair_to_jsonl_record(
    pair: TrainingPair,
    ref_summaries: Dict[str, Dict[str, dict]],
    forward_ratio: float = 0.0,
) -> dict:
    """Convert a training pair to a JSONL record for QLoRA training."""
    return {
        "pair_id": pair.pair_id,
        "service_id": pair.service_id,
        "pair_type": pair.pair_type,
        "instruction": _format_instruction(pair, ref_summaries),
        "output": _format_output(pair),
        "fidelity": pair.ground_truth_fidelity,
        "weight": _compute_loss_weight(pair, forward_ratio),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training pairs from CC reviews")
    parser.add_argument("--blueprints", required=True, help="Path to blueprints directory")
    parser.add_argument("--corpus-dir", required=True, help="Path to code-architect corpus directory")
    parser.add_argument("--validation-split", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    bp_dir = Path(args.blueprints)
    corpus_dir = Path(args.corpus_dir)
    pairs_dir = corpus_dir / "pairs"
    reviews_dir = corpus_dir / "reviews"
    retroactive_dir = corpus_dir / "retroactive"

    # Load all blueprints
    blueprints = _load_blueprints(bp_dir)
    logger.info("Loaded %d blueprints", len(blueprints))

    pairs: list[TrainingPair] = []

    # ── Process forward reviews ──────────────────────────────────────────────

    if reviews_dir.exists():
        for review_file in sorted(reviews_dir.glob("*.json")):
            review = _load_review_result(review_file)
            if not review:
                continue

            service_id = review["service_id"]
            if service_id not in blueprints:
                logger.warning("No blueprint for review service %s, skipping", service_id)
                continue

            # Look for AST summaries — check retroactive dir first (most complete)
            ast_summaries: Optional[dict] = None
            retro_dir = retroactive_dir / service_id
            if retro_dir.exists():
                ast_files = sorted(retro_dir.glob("ast_summaries_*.json"))
                if ast_files:
                    ast_summaries = _load_ast_summaries(ast_files[-1])

            if not ast_summaries:
                logger.warning("No AST summaries for %s, skipping", service_id)
                continue

            pair = _build_forward_pair(
                service_id, review, ast_summaries, blueprints[service_id], blueprints
            )
            pairs.append(pair)
            logger.info("Forward pair: %s (fidelity=%.2f)", service_id, pair.ground_truth_fidelity or 0)

    # ── Process retroactive reviews ──────────────────────────────────────────

    if retroactive_dir.exists():
        for service_dir in sorted(retroactive_dir.iterdir()):
            if not service_dir.is_dir():
                continue
            service_id = service_dir.name

            # Look for CC review results in the retroactive dir
            review_files = sorted(service_dir.glob("cc_review_*.json"))
            if not review_files:
                logger.info("No CC review yet for retroactive/%s (prompt exists, awaiting review)", service_id)
                continue

            review = _load_review_result(review_files[-1])
            if not review:
                continue

            if service_id not in blueprints:
                logger.warning("No blueprint for retroactive service %s, skipping", service_id)
                continue

            # Load AST summaries
            ast_files = sorted(service_dir.glob("ast_summaries_*.json"))
            if not ast_files:
                logger.warning("No AST summaries for retroactive/%s", service_id)
                continue
            ast_summaries = _load_ast_summaries(ast_files[-1])
            if not ast_summaries:
                continue

            pair = _build_retroactive_pair(
                service_id, review, ast_summaries, blueprints[service_id], blueprints
            )
            pairs.append(pair)
            logger.info("Retroactive pair: %s (fidelity=%.2f)", service_id, pair.ground_truth_fidelity or 0)

    # ── Report ───────────────────────────────────────────────────────────────

    logger.info("")
    logger.info("Total pairs: %d", len(pairs))
    type_counts: dict[str, int] = {}
    service_counts: dict[str, int] = {}
    for p in pairs:
        type_counts[p.pair_type] = type_counts.get(p.pair_type, 0) + 1
        service_counts[p.service_id] = service_counts.get(p.service_id, 0) + 1
    for pt, count in sorted(type_counts.items()):
        logger.info("  %s: %d", pt, count)
    for sid, count in sorted(service_counts.items()):
        logger.info("  %s: %d", sid, count)

    if args.dry_run:
        logger.info("DRY RUN — no files written")
        return

    if not pairs:
        logger.warning("No training pairs generated. Run CC reviews on retroactive prompts first.")
        print(json.dumps({"status": "empty", "total_pairs": 0}))
        return

    # ── Write pairs ──────────────────────────────────────────────────────────

    pairs_dir.mkdir(parents=True, exist_ok=True)

    for pair in pairs:
        pair_file = pairs_dir / f"{pair.pair_id}.json"
        pair_file.write_text(
            pair.model_dump_json(indent=2),
            encoding="utf-8",
        )

    # ── Load reference AST summaries for JSONL construction ──────────────────

    ref_summaries: Dict[str, Dict[str, dict]] = {}
    if retroactive_dir.exists():
        for service_dir in retroactive_dir.iterdir():
            if not service_dir.is_dir():
                continue
            ast_files = sorted(service_dir.glob("ast_summaries_*.json"))
            if ast_files:
                data = _load_ast_summaries(ast_files[-1])
                if data:
                    ref_summaries[service_dir.name] = data

    # ── Train/validation split ───────────────────────────────────────────────

    rng = random.Random(args.seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * args.validation_split))
    validation = shuffled[:val_count]
    train = shuffled[val_count:]

    # Ensure both pair types appear in validation if possible
    val_types = {p.pair_type for p in validation}
    all_types = {p.pair_type for p in pairs}
    for needed_type in all_types - val_types:
        # Steal one from train
        for i, p in enumerate(train):
            if p.pair_type == needed_type:
                validation.append(train.pop(i))
                break

    # ── Write JSONL ──────────────────────────────────────────────────────────

    forward_count = type_counts.get("forward", 0)
    forward_ratio = forward_count / len(pairs) if pairs else 0.0
    if forward_ratio < 0.30:
        logger.info("Forward pair ratio %.0f%% < 30%% — applying 1.5x loss weight to forward pairs", forward_ratio * 100)

    train_file = corpus_dir / "train.jsonl"
    val_file = corpus_dir / "validation.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for pair in train:
            record = _pair_to_jsonl_record(pair, ref_summaries, forward_ratio)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for pair in validation:
            record = _pair_to_jsonl_record(pair, ref_summaries, forward_ratio)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── Write metadata ───────────────────────────────────────────────────────

    fidelities = [p.ground_truth_fidelity for p in pairs if p.ground_truth_fidelity is not None]
    metadata = CorpusMetadata(
        total_pairs=len(pairs),
        train_count=len(train),
        validation_count=len(validation),
        pair_type_counts=type_counts,
        service_counts=service_counts,
        mean_fidelity=sum(fidelities) / len(fidelities) if fidelities else None,
        corpus_ready=len(pairs) >= 50,
    )

    meta_file = corpus_dir / "generation_metadata.json"
    meta_file.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")

    # ── Summary output ───────────────────────────────────────────────────────

    print(json.dumps({
        "status": "generated",
        "total_pairs": len(pairs),
        "train": len(train),
        "validation": len(validation),
        "pair_types": type_counts,
        "services": service_counts,
        "mean_fidelity": metadata.mean_fidelity,
        "corpus_ready": metadata.corpus_ready,
    }, indent=2))

    logger.info("Wrote %d train + %d validation pairs", len(train), len(validation))
    logger.info("Pairs dir: %s", pairs_dir)
    logger.info("Metadata: %s", meta_file)


if __name__ == "__main__":
    main()

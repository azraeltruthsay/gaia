"""
Curriculum Sync — Blueprint→Curriculum pipeline for incremental QLoRA training.

Detects changed blueprints via SHA-256 hashing, extracts instruction/output pairs
from YAML blueprint content, deduplicates against existing training data, and
appends new pairs to train.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("GAIA.CurriculumSync")

# Defaults
BLUEPRINTS_DIR = Path("/gaia/GAIA_Project/knowledge/blueprints")
CURRICULUM_DIR = Path("/gaia/GAIA_Project/knowledge/curricula")
HASH_MANIFEST = CURRICULUM_DIR / "blueprint_hashes.json"
TRAIN_JSONL = CURRICULUM_DIR / "train.jsonl"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_hash_manifest(path: Path = HASH_MANIFEST) -> Dict[str, str]:
    """Load the blueprint hash manifest. Returns empty dict if missing."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Corrupt hash manifest at %s, starting fresh", path)
        return {}


def _save_hash_manifest(manifest: Dict[str, str], path: Path = HASH_MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def detect_changed_blueprints(
    blueprints_dir: Path = BLUEPRINTS_DIR,
) -> Tuple[List[Path], Dict[str, str]]:
    """
    Compare blueprint YAML files against stored SHA-256 hashes.

    Returns:
        (changed_files, updated_manifest) — list of changed blueprint paths
        and the full updated manifest (caller should save after processing).
    """
    manifest = _load_hash_manifest()
    changed: List[Path] = []
    updated = dict(manifest)

    for bp_path in sorted(blueprints_dir.glob("*.yaml")):
        content = bp_path.read_text(encoding="utf-8")
        current_hash = _sha256(content)
        stored_hash = manifest.get(bp_path.name)

        if stored_hash != current_hash:
            changed.append(bp_path)
            updated[bp_path.name] = current_hash
            logger.info(
                "Blueprint changed: %s (old=%s, new=%s)",
                bp_path.name,
                stored_hash[:8] if stored_hash else "none",
                current_hash[:8],
            )

    return changed, updated


def _extract_pairs_from_yaml(bp_path: Path) -> List[Dict[str, str]]:
    """
    Generate instruction/output training pairs from a YAML blueprint.

    Extracts structured knowledge about:
    - Service role and identity
    - Runtime configuration (port, GPU, startup)
    - Interface endpoints
    - Design decisions and failure modes
    """
    try:
        import yaml
    except ImportError:
        # Fallback: parse key fields with basic string ops
        return _extract_pairs_plaintext(bp_path)

    content = bp_path.read_text(encoding="utf-8")
    try:
        bp = yaml.safe_load(content)
    except Exception:
        logger.warning("Failed to parse YAML blueprint: %s", bp_path.name)
        return _extract_pairs_plaintext(bp_path)

    if not isinstance(bp, dict):
        return []

    pairs: List[Dict[str, str]] = []
    service_id = bp.get("id", bp_path.stem)
    role = bp.get("role", "unknown")

    # Pair 1: Service identity
    pairs.append({
        "instruction": f"What is the role of {service_id} in the GAIA system?",
        "output": f"{service_id} is {role}. Service ID: {service_id}, version: {bp.get('version', 'unknown')}, status: {bp.get('service_status', 'unknown')}.",
    })

    # Pair 2: Runtime configuration
    runtime = bp.get("runtime", {})
    if runtime:
        port = runtime.get("port", "unknown")
        gpu = runtime.get("gpu", False)
        startup = runtime.get("startup_cmd", "unknown")
        pairs.append({
            "instruction": f"What is the runtime configuration for {service_id}?",
            "output": (
                f"{service_id} runs on port {port}, GPU: {gpu}, "
                f"startup command: {startup}, "
                f"health check: {runtime.get('health_check', 'unknown')}."
            ),
        })

    # Pair 3: Interfaces / endpoints
    interfaces = bp.get("interfaces", [])
    if interfaces and isinstance(interfaces, list):
        endpoint_lines = []
        for iface in interfaces:
            if not isinstance(iface, dict):
                continue
            transport = iface.get("transport", {})
            if not isinstance(transport, dict):
                continue
            method = transport.get("method", "?")
            path = transport.get("path", "?")
            desc = iface.get("description", "")
            direction = iface.get("direction", "")
            endpoint_lines.append(f"- {method} {path} ({direction}): {desc}")

        if endpoint_lines:
            pairs.append({
                "instruction": f"What endpoints does {service_id} expose?",
                "output": f"{service_id} endpoints:\n" + "\n".join(endpoint_lines),
            })

    # Pair 4: Design decisions (if present)
    design = bp.get("design_decisions", [])
    if design and isinstance(design, list):
        decision_text = "\n".join(
            f"- {d}" if isinstance(d, str) else f"- {d.get('decision', str(d))}"
            for d in design
        )
        pairs.append({
            "instruction": f"What are the key design decisions for {service_id}?",
            "output": f"Design decisions for {service_id}:\n{decision_text}",
        })

    # Pair 5: Failure modes (if present)
    failure_modes = bp.get("failure_modes", [])
    if failure_modes and isinstance(failure_modes, list):
        fm_text = "\n".join(
            f"- {fm}" if isinstance(fm, str) else f"- {fm.get('mode', str(fm))}: {fm.get('mitigation', '')}"
            for fm in failure_modes
        )
        pairs.append({
            "instruction": f"What are the failure modes for {service_id}?",
            "output": f"Failure modes for {service_id}:\n{fm_text}",
        })

    # Pair 6: Dependencies (if present)
    deps = bp.get("dependencies", [])
    if deps and isinstance(deps, list):
        dep_names = [d if isinstance(d, str) else d.get("id", str(d)) for d in deps]
        pairs.append({
            "instruction": f"What services does {service_id} depend on?",
            "output": f"{service_id} depends on: {', '.join(dep_names)}.",
        })

    return pairs


def _extract_pairs_plaintext(bp_path: Path) -> List[Dict[str, str]]:
    """Fallback pair extraction when PyYAML is unavailable."""
    content = bp_path.read_text(encoding="utf-8")
    service_id = bp_path.stem

    # Extract basic fields via string matching
    pairs: List[Dict[str, str]] = []

    # Full blueprint as a knowledge pair
    pairs.append({
        "instruction": f"Describe the {service_id} service blueprint.",
        "output": content[:2000],  # Cap at 2000 chars
    })

    return pairs


def _load_existing_pairs(train_path: Path = TRAIN_JSONL) -> set:
    """Load existing training pair hashes for deduplication."""
    hashes: set = set()
    if not train_path.exists():
        return hashes
    for line in train_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            hashes.add(_sha256(line))
    return hashes


def _append_pairs(
    pairs: List[Dict[str, str]],
    train_path: Path = TRAIN_JSONL,
    existing_hashes: Optional[set] = None,
) -> int:
    """Append new pairs to train.jsonl, skipping duplicates. Returns count added."""
    if existing_hashes is None:
        existing_hashes = _load_existing_pairs(train_path)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    added = 0

    with open(train_path, "a", encoding="utf-8") as f:
        for pair in pairs:
            line = json.dumps(pair, ensure_ascii=False)
            line_hash = _sha256(line)
            if line_hash not in existing_hashes:
                f.write(line + "\n")
                existing_hashes.add(line_hash)
                added += 1

    return added


def sync_curriculum(
    blueprints_dir: Path = BLUEPRINTS_DIR,
    curriculum_dir: Path = CURRICULUM_DIR,
) -> Dict[str, Any]:
    """
    Main entry point: detect changed blueprints → extract pairs → deduplicate → append.

    Returns:
        Dict with keys: changed_count, new_pairs, total_pairs, trigger_training (bool),
        and train_jsonl path.
    """
    train_path = curriculum_dir / "train.jsonl"
    hash_manifest_path = curriculum_dir / "blueprint_hashes.json"

    changed, updated_manifest = detect_changed_blueprints(blueprints_dir)

    if not changed:
        logger.info("No blueprint changes detected — skipping curriculum sync")
        return {
            "changed_count": 0,
            "new_pairs": 0,
            "total_pairs": _count_lines(train_path),
            "trigger_training": False,
            "train_jsonl": str(train_path),
        }

    # Extract pairs from changed blueprints
    all_new_pairs: List[Dict[str, str]] = []
    for bp_path in changed:
        pairs = _extract_pairs_from_yaml(bp_path)
        all_new_pairs.extend(pairs)
        logger.info("Extracted %d pairs from %s", len(pairs), bp_path.name)

    # Deduplicate and append
    existing_hashes = _load_existing_pairs(train_path)
    added = _append_pairs(all_new_pairs, train_path, existing_hashes)

    # Save updated manifest only after successful processing
    _save_hash_manifest(updated_manifest, hash_manifest_path)

    total = _count_lines(train_path)
    logger.info(
        "Curriculum sync complete: %d blueprints changed, %d new pairs added (%d total)",
        len(changed), added, total,
    )

    return {
        "changed_count": len(changed),
        "new_pairs": added,
        "total_pairs": total,
        "trigger_training": added > 0,
        "train_jsonl": str(train_path),
    }


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

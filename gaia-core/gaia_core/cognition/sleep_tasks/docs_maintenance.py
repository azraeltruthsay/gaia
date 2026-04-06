"""
Docs Maintenance sleep task — detects stale documentation and drafts updates.

During sleep, GAIA:
  1. Queries gaia-doctor's dissonance endpoint for live/candidate hash mismatches
  2. Checks recent git changes against documentation coverage
  3. Uses available model (Core on CPU) to draft doc update suggestions
  4. Saves drafts as JSON fragments to /shared/docs_drafts/
  5. Plants a thought seed summarizing findings for next session review

All drafts are suggestions only — no actual doc files are modified.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("GAIA.SleepTask.DocsMaintenance")

# Directories
SHARED_DIR = Path(os.getenv("SHARED_DIR", "/shared"))
DRAFTS_DIR = SHARED_DIR / "docs_drafts"

# Doctor dissonance endpoint
DOCTOR_URL = os.getenv("DOCTOR_ENDPOINT", "http://gaia-doctor:6419")

# Documentation targets to monitor
DOC_TARGETS = {
    "knowledge/system_reference/AS_BUILT_ARCHITECTURE.md": [
        "gaia-core/gaia_core/cognition/",
        "gaia-core/gaia_core/models/",
        "gaia-common/gaia_common/engine/",
    ],
    "knowledge/system_reference/AS_BUILT_LATEST.md": [
        "gaia-core/",
        "gaia-common/",
        "gaia-orchestrator/",
    ],
    "contracts/CONNECTIVITY.md": [
        "contracts/services/",
        "gaia-core/gaia_core/api/",
        "gaia-web/gaia_web/api/",
    ],
    "CLAUDE.md": [
        "gaia-core/gaia_core/cognition/sleep_tasks/",
        "contracts/services/",
    ],
}

# Contract YAML → source directory mapping
CONTRACT_SOURCE_MAP = {
    "contracts/services/gaia-core.yaml": "gaia-core/gaia_core/",
    "contracts/services/gaia-web.yaml": "gaia-web/gaia_web/",
    "contracts/services/gaia-mcp.yaml": "gaia-mcp/gaia_mcp/",
    "contracts/services/gaia-engine.yaml": "gaia-engine/gaia_engine/",
    "contracts/services/gaia-orchestrator.yaml": "gaia-orchestrator/gaia_orchestrator/",
    "contracts/services/gaia-study.yaml": "gaia-study/gaia_study/",
    "contracts/services/gaia-audio.yaml": "gaia-audio/gaia_audio/",
}

PROJECT_ROOT = Path("/gaia/GAIA_Project")


def run_docs_maintenance(
    config,
    model_pool=None,
    check_interrupted: Optional[Callable] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Main entry point for the docs maintenance sleep task.

    Parameters
    ----------
    config : Config
        GAIA config singleton.
    model_pool : optional
        Model pool for LLM inference (Core on CPU during sleep).
    check_interrupted : optional
        Callable that raises TaskInterruptedError if wake signal pending.

    Returns
    -------
    dict
        Summary of findings and drafts created.
    """
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)

    stale_files: List[Dict[str, Any]] = []
    drafts: List[Dict[str, Any]] = []

    # ── Phase 1: Doctor dissonance check ──────────────────────────
    dissonance_files = _check_doctor_dissonance()
    if dissonance_files:
        logger.info("DocsMaint: %d dissonance entries from doctor", len(dissonance_files))
        stale_files.extend(dissonance_files)

    if check_interrupted:
        check_interrupted()

    # ── Phase 2: Git-based staleness detection ────────────────────
    git_stale = _check_git_staleness()
    if git_stale:
        logger.info("DocsMaint: %d stale doc areas from git analysis", len(git_stale))
        stale_files.extend(git_stale)

    if check_interrupted:
        check_interrupted()

    # ── Phase 3: Contract freshness ───────────────────────────────
    contract_stale = _check_contract_freshness()
    if contract_stale:
        logger.info("DocsMaint: %d stale contracts detected", len(contract_stale))
        stale_files.extend(contract_stale)

    if not stale_files:
        logger.info("DocsMaint: all documentation appears current")
        return {"stale_files": [], "drafts": [], "summary": "No stale documentation detected"}

    if check_interrupted:
        check_interrupted()

    # ── Phase 4: Draft updates ────────────────────────────────────
    # Deduplicate by target file
    seen_targets = set()
    unique_stale = []
    for entry in stale_files:
        target = entry.get("target", "")
        if target not in seen_targets:
            seen_targets.add(target)
            unique_stale.append(entry)

    for entry in unique_stale:
        if check_interrupted:
            check_interrupted()

        draft = _draft_update(entry, model_pool, config)
        if draft:
            drafts.append(draft)

    # ── Phase 5: Save drafts bundle ───────────────────────────────
    timestamp = datetime.now(timezone.utc).isoformat()
    bundle = {
        "timestamp": timestamp,
        "stale_files": [e.get("target", "unknown") for e in unique_stale],
        "drafts": drafts,
        "summary": f"Found {len(unique_stale)} stale docs after recent changes",
    }

    bundle_name = f"docs_draft_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    bundle_path = DRAFTS_DIR / bundle_name
    _atomic_write_json(bundle_path, bundle)
    logger.info("DocsMaint: saved draft bundle to %s", bundle_path)

    # ── Phase 6: Plant thought seed ───────────────────────────────
    _plant_thought_seed(unique_stale, drafts)

    return bundle


# ──────────────────────────────────────────────────────────────────
# Phase 1: Doctor dissonance
# ──────────────────────────────────────────────────────────────────

def _check_doctor_dissonance() -> List[Dict[str, Any]]:
    """Query gaia-doctor /dissonance for live != candidate hash mismatches."""
    results = []
    try:
        import httpx
        resp = httpx.get(
            f"{DOCTOR_URL}/dissonance",
            timeout=10.0,
        )
        if resp.status_code != 200:
            logger.debug("Doctor dissonance returned %d", resp.status_code)
            return results

        data = resp.json()
        dissonance_items = data if isinstance(data, list) else data.get("items", [])

        for item in dissonance_items:
            file_path = item.get("file", item.get("path", ""))
            if not file_path:
                continue
            # Only care about documentation-related dissonance
            if _is_doc_related(file_path):
                results.append({
                    "target": file_path,
                    "reason": "Doctor dissonance: live/candidate hash mismatch",
                    "source": "doctor_dissonance",
                    "details": item,
                })
    except ImportError:
        logger.debug("httpx not available, skipping doctor dissonance check")
    except Exception as exc:
        logger.debug("Doctor dissonance check failed: %s", exc)

    return results


def _is_doc_related(path: str) -> bool:
    """Check if a file path is documentation or affects documentation."""
    doc_extensions = {".md", ".yaml", ".yml", ".rst"}
    doc_dirs = {"knowledge/", "contracts/", "docs/"}

    ext = Path(path).suffix.lower()
    if ext in doc_extensions:
        return True
    for d in doc_dirs:
        if d in path:
            return True
    return False


# ──────────────────────────────────────────────────────────────────
# Phase 2: Git-based staleness
# ──────────────────────────────────────────────────────────────────

def _check_git_staleness() -> List[Dict[str, Any]]:
    """Check if recently changed source files have corresponding stale docs."""
    results = []
    try:
        # Get files changed in the last 7 days
        recent_changes = _git_recent_changes(days=7)
        if not recent_changes:
            return results

        for doc_target, source_dirs in DOC_TARGETS.items():
            doc_path = PROJECT_ROOT / doc_target
            if not doc_path.exists():
                continue

            # Check if any monitored source dirs have changes newer than the doc
            doc_mtime = doc_path.stat().st_mtime
            affecting_changes = []

            for changed_file in recent_changes:
                for source_dir in source_dirs:
                    if changed_file.startswith(source_dir):
                        changed_path = PROJECT_ROOT / changed_file
                        if changed_path.exists() and changed_path.stat().st_mtime > doc_mtime:
                            affecting_changes.append(changed_file)

            if affecting_changes:
                results.append({
                    "target": doc_target,
                    "reason": f"Source changes newer than doc: {', '.join(affecting_changes[:5])}",
                    "source": "git_staleness",
                    "affecting_files": affecting_changes[:10],
                })
    except Exception as exc:
        logger.debug("Git staleness check failed: %s", exc)

    return results


def _git_recent_changes(days: int = 7) -> List[str]:
    """Get list of files changed in git in the last N days."""
    try:
        result = subprocess.run(
            ["git", "log", f"--since={days} days ago", "--name-only", "--pretty=format:"],
            capture_output=True, text=True, timeout=15,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            return []
        files = set()
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if line:
                files.add(line)
        return list(files)
    except Exception as exc:
        logger.debug("git log failed: %s", exc)
        return []


# ──────────────────────────────────────────────────────────────────
# Phase 3: Contract freshness
# ──────────────────────────────────────────────────────────────────

def _check_contract_freshness() -> List[Dict[str, Any]]:
    """Check if service contracts are older than their source code."""
    results = []
    for contract_path, source_dir in CONTRACT_SOURCE_MAP.items():
        contract_full = PROJECT_ROOT / contract_path
        source_full = PROJECT_ROOT / source_dir

        if not contract_full.exists() or not source_full.exists():
            continue

        contract_mtime = contract_full.stat().st_mtime
        newest_source = _newest_mtime(source_full)

        if newest_source and newest_source > contract_mtime:
            results.append({
                "target": contract_path,
                "reason": f"Service source in {source_dir} is newer than contract",
                "source": "contract_freshness",
                "contract_age_hours": (time.time() - contract_mtime) / 3600,
                "source_age_hours": (time.time() - newest_source) / 3600,
            })

    return results


def _newest_mtime(directory: Path) -> Optional[float]:
    """Find the newest modification time among .py files in a directory."""
    newest = None
    try:
        for py_file in directory.rglob("*.py"):
            mtime = py_file.stat().st_mtime
            if newest is None or mtime > newest:
                newest = mtime
    except Exception:
        pass
    return newest


# ──────────────────────────────────────────────────────────────────
# Phase 4: Draft generation
# ──────────────────────────────────────────────────────────────────

def _draft_update(
    stale_entry: Dict[str, Any],
    model_pool,
    config,
) -> Optional[Dict[str, Any]]:
    """Generate a draft update suggestion for a stale documentation file.

    Uses available LLM (Core on CPU during sleep) if available,
    otherwise produces a structured note without inference.
    """
    target = stale_entry.get("target", "")
    reason = stale_entry.get("reason", "")
    affecting_files = stale_entry.get("affecting_files", [])

    # Read current content hash for staleness tracking
    target_path = PROJECT_ROOT / target
    current_hash = ""
    if target_path.exists():
        try:
            content = target_path.read_text(encoding="utf-8")
            current_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        except Exception:
            pass

    # Try LLM-assisted draft
    suggested_update = _generate_llm_draft(target, reason, affecting_files, model_pool)

    # If no LLM available, create a structured note
    if not suggested_update:
        change_summary = ", ".join(affecting_files[:5]) if affecting_files else "see reason"
        suggested_update = (
            f"[AUTO-DETECTED] Documentation in '{target}' may be outdated.\n"
            f"Reason: {reason}\n"
            f"Affected source files: {change_summary}\n"
            f"Review recommended during next Claude Code session."
        )

    return {
        "target": target,
        "section": _infer_section(target, affecting_files),
        "current_content_hash": current_hash,
        "suggested_update": suggested_update,
        "reason": reason,
        "auto_apply": False,
    }


def _generate_llm_draft(
    target: str,
    reason: str,
    affecting_files: List[str],
    model_pool,
) -> Optional[str]:
    """Use the available model to draft a doc update summary.

    Returns None if no model is available or inference fails.
    Keeps the prompt short to be lightweight during sleep.
    """
    if model_pool is None:
        return None

    # Try to get any available model (Core on CPU is typical during sleep)
    model = None
    for key in ("core", "cpu_core", "gpu_prime", "prime"):
        model = model_pool.models.get(key) if hasattr(model_pool, "models") else None
        if model is not None:
            break

    if model is None:
        return None

    files_text = "\n".join(f"  - {f}" for f in affecting_files[:5])
    prompt = (
        f"You are reviewing documentation freshness for the GAIA project.\n"
        f"The file '{target}' may be outdated.\n"
        f"Reason: {reason}\n"
    )
    if files_text:
        prompt += f"Recently changed source files:\n{files_text}\n"
    prompt += (
        "\nWrite a brief (2-4 sentences) summary of what likely needs updating "
        "in this documentation file. Be specific about which sections or "
        "descriptions might be affected. Do not write the actual update — "
        "just describe what should be reviewed and why."
    )

    try:
        if hasattr(model, "create_chat_completion"):
            result = model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.3,
            )
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        elif hasattr(model, "generate"):
            return model.generate(prompt, max_tokens=256)
    except Exception as exc:
        logger.debug("LLM draft generation failed: %s", exc)

    return None


def _infer_section(target: str, affecting_files: List[str]) -> str:
    """Infer which section of the doc is likely affected."""
    if "contract" in target.lower() or target.endswith(".yaml"):
        return "API Endpoints"
    if "AS_BUILT" in target:
        # Try to guess section from affecting file paths
        for f in affecting_files:
            if "cognition/" in f:
                return "Cognitive Pipeline"
            if "models/" in f:
                return "Model Tiers"
            if "engine/" in f:
                return "GAIA Engine"
            if "api/" in f:
                return "API Layer"
        return "Architecture Overview"
    if "CONNECTIVITY" in target:
        return "Service Communication"
    if "CLAUDE" in target:
        return "Project Configuration"
    return "General"


# ──────────────────────────────────────────────────────────────────
# Phase 6: Thought seed
# ──────────────────────────────────────────────────────────────────

def _plant_thought_seed(
    stale_entries: List[Dict[str, Any]],
    drafts: List[Dict[str, Any]],
) -> None:
    """Plant a thought seed summarizing docs maintenance findings.

    The seed is saved directly to the seeds directory (same format as
    thought_seed.py) so it appears in the next review cycle.
    """
    if not stale_entries:
        return

    seeds_dir = Path("/knowledge/seeds")
    try:
        seeds_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.debug("Cannot create seeds dir, skipping thought seed")
        return

    targets = [e.get("target", "?") for e in stale_entries[:5]]
    targets_text = ", ".join(targets)
    if len(stale_entries) > 5:
        targets_text += f" (+{len(stale_entries) - 5} more)"

    seed_text = (
        f"Documentation maintenance: {len(stale_entries)} stale docs detected. "
        f"Targets: {targets_text}. "
        f"{len(drafts)} draft updates saved to /shared/docs_drafts/. "
        f"Review and apply during next session."
    )

    fname = f"seed_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S%f')}.json"
    seed_obj = {
        "created": datetime.now(timezone.utc).isoformat(),
        "seed_type": "docs_maintenance",
        "context": {
            "prompt": "sleep_task:docs_maintenance",
            "packet_id": "sleep_docs_maintenance",
            "persona": "maintenance",
        },
        "seed": seed_text,
        "reviewed": False,
        "action_taken": False,
        "result": None,
    }

    try:
        seed_path = seeds_dir / fname
        with open(seed_path, "w", encoding="utf-8") as f:
            json.dump(seed_obj, f, indent=2)
        logger.info("DocsMaint: thought seed planted — %s", fname)
    except Exception as exc:
        logger.debug("Failed to plant thought seed: %s", exc)


# ──────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────

def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically via tmp + rename."""
    tmp_path = path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(str(tmp_path), str(path))
    except Exception:
        # Clean up tmp on failure
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise

"""
Code Evolution — utilities for GAIA's code self-awareness.

Compares candidate (future) vs production (present) code, indexes .bak files
(past), reads recent git history, and inventories the archived monolith.

The generate_code_evolution_snapshot() function is called as a sleep task
to produce a markdown summary that GAIA can reference in her temporal context.
"""

import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GAIA.CodeEvolution")

# Candidate → Production service mapping
_SERVICE_MAP = {
    "gaia-common": ("candidates/gaia-common", "gaia-common"),
    "gaia-core": ("candidates/gaia-core", "gaia-core"),
    "gaia-web": ("candidates/gaia-web", "gaia-web"),
    "gaia-orchestrator": ("candidates/gaia-orchestrator", "gaia-orchestrator"),
}

# Extensions to compare (skip __pycache__, .pyc, etc.)
_COMPARE_EXTENSIONS = {".py", ".js", ".html", ".css", ".yaml", ".yml", ".json", ".toml", ".md"}
_SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".mypy_cache", ".ruff_cache", ".pytest_cache"}


def diff_candidate_vs_production(
    service_id: str,
    project_root: str = "/gaia/GAIA_Project",
) -> Dict[str, Any]:
    """Compare a candidate service dir against its production counterpart.

    Returns: {service_id, files_added, files_removed, files_changed, unchanged, summary_lines}
    """
    if service_id not in _SERVICE_MAP:
        return {"service_id": service_id, "error": f"Unknown service: {service_id}"}

    cand_rel, prod_rel = _SERVICE_MAP[service_id]
    cand_root = Path(project_root) / cand_rel
    prod_root = Path(project_root) / prod_rel

    if not cand_root.exists():
        return {"service_id": service_id, "error": "Candidate dir not found"}
    if not prod_root.exists():
        return {"service_id": service_id, "error": "Production dir not found"}

    cand_files = _collect_files(cand_root)
    prod_files = _collect_files(prod_root)

    # Relative paths for comparison
    cand_set = set(cand_files.keys())
    prod_set = set(prod_files.keys())

    added = sorted(cand_set - prod_set)
    removed = sorted(prod_set - cand_set)
    common = cand_set & prod_set

    changed = []
    unchanged = 0
    for rel_path in sorted(common):
        try:
            cand_content = cand_files[rel_path].read_bytes()
            prod_content = prod_files[rel_path].read_bytes()
            if cand_content != prod_content:
                changed.append(rel_path)
            else:
                unchanged += 1
        except OSError:
            changed.append(rel_path)

    # Build summary lines
    summary = []
    if added:
        summary.append(f"  + {len(added)} new: {', '.join(_basenames(added[:3]))}")
    if removed:
        summary.append(f"  - {len(removed)} removed: {', '.join(_basenames(removed[:3]))}")
    if changed:
        summary.append(f"  ~ {len(changed)} changed: {', '.join(_basenames(changed[:3]))}")

    return {
        "service_id": service_id,
        "files_added": added,
        "files_removed": removed,
        "files_changed": changed,
        "unchanged": unchanged,
        "summary_lines": summary,
    }


def list_changed_candidates(
    project_root: str = "/gaia/GAIA_Project",
) -> List[Dict[str, Any]]:
    """Scan all candidate services and return which have pending changes."""
    results = []
    for service_id in _SERVICE_MAP:
        diff = diff_candidate_vs_production(service_id, project_root)
        if diff.get("error"):
            continue
        total_changes = (
            len(diff.get("files_added", []))
            + len(diff.get("files_removed", []))
            + len(diff.get("files_changed", []))
        )
        if total_changes > 0:
            results.append(diff)
    return results


def recent_git_log(
    project_root: str = "/gaia/GAIA_Project",
    limit: int = 10,
) -> List[Dict[str, str]]:
    """Parse recent git log entries.

    Returns: [{hash, date, subject}]
    """
    try:
        result = subprocess.run(
            [
                "git", "log",
                f"-{limit}",
                "--format=%h|%aI|%s",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
        entries = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("|", 2)
            if len(parts) == 3:
                entries.append({
                    "hash": parts[0],
                    "date": parts[1],
                    "subject": parts[2],
                })
        return entries
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []


def index_bak_files(
    project_root: str = "/gaia/GAIA_Project",
) -> List[Dict[str, Any]]:
    """Find all .bak files, parse timestamps from filenames.

    Returns sorted list: [{path, original_file, timestamp, size_bytes}]
    """
    results = []
    root = Path(project_root)

    try:
        for bak_path in root.rglob("*.bak*"):
            if not bak_path.is_file():
                continue
            # Skip if inside .git
            if ".git" in bak_path.parts:
                continue

            entry: Dict[str, Any] = {
                "path": str(bak_path.relative_to(root)),
                "size_bytes": bak_path.stat().st_size,
            }

            # Parse timestamp from .bak.YYYYMMDD_HHMMSS pattern
            name = bak_path.name
            ts_str = None
            if ".bak." in name:
                ts_part = name.split(".bak.")[-1]
                ts_str = _parse_bak_timestamp(ts_part)
                entry["original_file"] = name.split(".bak.")[0]
            else:
                entry["original_file"] = name.replace(".bak", "")

            entry["timestamp"] = ts_str
            results.append(entry)
    except OSError:
        logger.debug("Failed to scan for .bak files", exc_info=True)

    # Sort by timestamp (newest first), nulls last
    results.sort(key=lambda e: e.get("timestamp") or "", reverse=True)
    return results


def archive_inventory(
    archive_root: str = "/gaia/GAIA_Project/archive",
) -> Dict[str, Any]:
    """Summarize the archive directory — service dirs, file counts, size."""
    root = Path(archive_root)
    if not root.exists():
        return {"exists": False, "path": str(root)}

    inventory: Dict[str, Any] = {
        "exists": True,
        "path": str(root),
        "subdirs": [],
        "total_files": 0,
    }

    try:
        for item in sorted(root.iterdir()):
            if item.is_dir():
                file_count = sum(1 for _ in item.rglob("*") if _.is_file())
                inventory["subdirs"].append({
                    "name": item.name,
                    "file_count": file_count,
                })
                inventory["total_files"] += file_count
    except OSError:
        logger.debug("Failed to inventory archive", exc_info=True)

    return inventory


def generate_code_evolution_snapshot(
    project_root: str = "/gaia/GAIA_Project",
    output_path: str = "/shared/self_model/code_evolution.md",
) -> str:
    """Generate a markdown snapshot of GAIA's code evolution state.

    Called as a sleep task. Writes to output_path and returns it.
    """
    now = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Code Evolution Snapshot",
        f"Generated: {now}",
        "",
    ]

    # Pending candidate changes
    changed = list_changed_candidates(project_root)
    if changed:
        lines.append("## Pending Candidate Changes")
        for svc in changed:
            sid = svc["service_id"]
            n_add = len(svc.get("files_added", []))
            n_rm = len(svc.get("files_removed", []))
            n_ch = len(svc.get("files_changed", []))
            parts = []
            if n_ch:
                names = ", ".join(_basenames(svc["files_changed"][:3]))
                extra = f" +{n_ch - 3}" if n_ch > 3 else ""
                parts.append(f"{n_ch} changed ({names}{extra})")
            if n_add:
                parts.append(f"{n_add} added")
            if n_rm:
                parts.append(f"{n_rm} removed")
            lines.append(f"- **{sid}**: {', '.join(parts)}")
        lines.append("")
    else:
        lines.append("## Pending Candidate Changes")
        lines.append("All candidates match production.")
        lines.append("")

    # Recent git commits
    commits = recent_git_log(project_root, limit=10)
    if commits:
        lines.append("## Recent Commits")
        for c in commits:
            date_short = c["date"][:10]
            lines.append(f"- `{c['hash']}` ({date_short}): {c['subject']}")
        lines.append("")

    # Backup file index
    bak_files = index_bak_files(project_root)
    if bak_files:
        lines.append("## Backup History")
        timestamps = [b["timestamp"] for b in bak_files if b.get("timestamp")]
        if timestamps:
            lines.append(
                f"- {len(bak_files)} .bak files (oldest: {timestamps[-1][:10]}, "
                f"newest: {timestamps[0][:10]})"
            )
        else:
            lines.append(f"- {len(bak_files)} .bak files (no timestamps)")

        # Group by original file
        originals = set(b.get("original_file", "") for b in bak_files)
        if originals:
            lines.append(f"- Sources: {', '.join(sorted(originals)[:5])}")
        lines.append("")

    # Archive reference
    inv = archive_inventory(os.path.join(project_root, "archive"))
    if inv.get("exists"):
        lines.append("## Archive Reference")
        for sd in inv.get("subdirs", []):
            lines.append(f"- **{sd['name']}**: {sd['file_count']} files (pre-SOA monolith)")
        lines.append("")

    # Write snapshot
    content = "\n".join(lines)
    try:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")
        logger.info("Code evolution snapshot written to %s", output_path)
    except OSError:
        logger.warning("Failed to write code evolution snapshot", exc_info=True)

    return output_path


# ── Internal helpers ─────────────────────────────────────────────────────

def _collect_files(root: Path) -> Dict[str, Path]:
    """Collect relevant source files under a root, keyed by relative path."""
    files: Dict[str, Path] = {}
    try:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in _COMPARE_EXTENSIONS:
                continue
            if any(skip in path.parts for skip in _SKIP_DIRS):
                continue
            rel = str(path.relative_to(root))
            files[rel] = path
    except OSError:
        pass
    return files


def _basenames(paths: List[str]) -> List[str]:
    """Extract just the filename from a list of relative paths."""
    return [Path(p).name for p in paths]


def _parse_bak_timestamp(ts_part: str) -> Optional[str]:
    """Parse a YYYYMMDD_HHMMSS string into ISO format, or None."""
    try:
        dt = datetime.strptime(ts_part.strip(), "%Y%m%d_%H%M%S")
        return dt.replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        return None

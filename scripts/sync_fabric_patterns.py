#!/usr/bin/env python3
"""
Sync Fabric patterns from GitHub to knowledge/fabric_patterns/.

Downloads patterns via GitHub tarball API (no git clone needed).
Tracks content hashes for supply-chain integrity — flags changed
patterns on re-sync for manual review before accepting updates.

Usage:
    python scripts/sync_fabric_patterns.py
    python scripts/sync_fabric_patterns.py --target knowledge/fabric_patterns
    python scripts/sync_fabric_patterns.py --accept-changes   # skip review gate
"""

import argparse
import hashlib
import io
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    sys.exit(1)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sync(
    target_dir: str = "knowledge/fabric_patterns",
    branch: str = "main",
    accept_changes: bool = False,
) -> dict:
    """Download Fabric patterns and track content hashes.

    Returns summary dict with counts of added/updated/flagged patterns.
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    hashes_file = target / "_hashes.json"
    old_hashes: dict = {}
    if hashes_file.exists():
        old_hashes = json.loads(hashes_file.read_text(encoding="utf-8"))

    # Fetch tarball from GitHub
    url = f"https://api.github.com/repos/danielmiessler/Fabric/tarball/{branch}"
    print(f"Fetching {url} ...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    # Extract patterns
    new_hashes: dict = {}
    added, updated, unchanged, flagged = [], [], [], []

    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        for member in tar.getmembers():
            # Tarball structure: {prefix}/data/patterns/{name}/system.md
            parts = member.name.split("/")
            # Find the "patterns" segment and extract {name}/system.md after it
            try:
                idx = parts.index("patterns")
            except ValueError:
                continue
            if len(parts) < idx + 3 or parts[idx + 2] != "system.md":
                continue

            pattern_name = parts[idx + 1]
            f = tar.extractfile(member)
            if f is None:
                continue

            content = f.read()
            content_hash = _sha256(content)
            new_hashes[pattern_name] = content_hash

            out_dir = target / pattern_name
            out_file = out_dir / "system.md"

            if pattern_name in old_hashes:
                if old_hashes[pattern_name] == content_hash:
                    unchanged.append(pattern_name)
                    continue
                elif not accept_changes:
                    flagged.append(pattern_name)
                    # Write to staging area, not live
                    staged = out_dir / "system.md.pending"
                    out_dir.mkdir(exist_ok=True)
                    staged.write_bytes(content)
                    continue
                else:
                    updated.append(pattern_name)
            else:
                added.append(pattern_name)

            out_dir.mkdir(exist_ok=True)
            out_file.write_bytes(content)

    # Write hashes (merge: keep old hashes for flagged, update for accepted)
    merged_hashes = {**old_hashes}
    for name in added + updated:
        merged_hashes[name] = new_hashes[name]
    hashes_file.write_text(
        json.dumps(merged_hashes, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Write pending review list if any patterns were flagged
    pending_file = target / "_pending_review.json"
    if flagged:
        pending_file.write_text(
            json.dumps(
                {
                    "flagged_patterns": flagged,
                    "reason": "Content hash changed since last sync. Review diffs before accepting.",
                    "accept_command": f"python {__file__} --target {target_dir} --accept-changes",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWARNING: {len(flagged)} patterns have changed content.")
        print(f"Review: {pending_file}")
        print(f"Accept: python {__file__} --target {target_dir} --accept-changes")
    elif pending_file.exists():
        pending_file.unlink()

    # Write sync metadata
    meta = {
        "last_sync": datetime.now(timezone.utc).isoformat(),
        "branch": branch,
        "total_patterns": len(new_hashes),
        "added": len(added),
        "updated": len(updated),
        "unchanged": len(unchanged),
        "flagged": len(flagged),
    }
    (target / "_last_sync.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nSync complete:")
    print(f"  Added:     {len(added)}")
    print(f"  Updated:   {len(updated)}")
    print(f"  Unchanged: {len(unchanged)}")
    print(f"  Flagged:   {len(flagged)}")
    print(f"  Total:     {len(new_hashes)}")

    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Fabric patterns from GitHub")
    parser.add_argument("--target", default="knowledge/fabric_patterns",
                        help="Target directory for patterns")
    parser.add_argument("--branch", default="main", help="Git branch to sync from")
    parser.add_argument("--accept-changes", action="store_true",
                        help="Accept all content changes without review")
    args = parser.parse_args()
    result = sync(args.target, args.branch, args.accept_changes)

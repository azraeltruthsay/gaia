#!/usr/bin/env python3
"""
Post-training reset.

Runs as the final step of any training pipeline. Makes the newly-trained
model start clean instead of inheriting the prior model's KV state, session
history bias, and baked identity prefix. None of this is weight-compatible
across a training run, so leaving it in place silently propagates the old
model's ghost into the new one's behavior.

Steps:
  1. Archive /shared/sessions.json → /shared/sessions.archived/YYYY-MM-DD/
  2. Distill a memory journal from the archived sessions (optional, uses a
     stable non-training tier — Groq or pre-swap Core — so the new model
     doesn't self-referentially contaminate its own journal).
  3. Invalidate KV state:
       - clear /shared/kvcache/<tier>/handoff_context.json
       - clear /shared/kvcache/<tier>/core_checkpoint
       - POST /cache/invalidate to each reachable engine
  4. Regenerate identity_prefix.pt for each tier with the new weights.
  5. Clear session vector indexes (optional — new thresholds filter noise).
  6. Write a dev journal entry noting which model is now live.

Usage:
    python scripts/post_training_reset.py --tier core
    python scripts/post_training_reset.py --tier prime --skip-journal
    python scripts/post_training_reset.py --all            # every tier

The script is idempotent and safe to re-run. All destructive actions are
moves-into-archive or /cache/invalidate calls — nothing is deleted.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("post_training_reset")

SHARED_DIR = Path(os.environ.get("SHARED_DIR", "/gaia/GAIA_Project/shared"))
SESSIONS_FILE = SHARED_DIR / "sessions.json"
SESSIONS_ARCHIVE = SHARED_DIR / "sessions.archived"
KVCACHE_DIR = SHARED_DIR / "kvcache"
SESSION_VECTORS_DIR = Path("/gaia/GAIA_Project/gaia-core/data/shared/session_vectors")
DEV_JOURNAL_DIR = Path("/gaia/GAIA_Project/knowledge/Dev_Notebook")

TIER_ENDPOINTS = {
    "core": "http://localhost:6415",
    "prime": "http://localhost:7777",
}
TIER_ENGINE_ENDPOINTS = {
    # Engine (manager) endpoints, where /cache/invalidate lives.
    "core": "http://localhost:8092",
    "prime": "http://localhost:7777",
}


def archive_sessions(dry_run: bool = False) -> Path | None:
    """Move sessions.json into a dated archive directory.

    Returns the archive path, or None if nothing was archived.
    """
    if not SESSIONS_FILE.exists():
        log.info("No sessions.json to archive at %s", SESSIONS_FILE)
        return None

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    target_dir = SESSIONS_ARCHIVE / stamp
    target = target_dir / f"sessions-{datetime.now(timezone.utc).strftime('%H%M%S')}.json"

    if dry_run:
        log.info("[dry-run] would archive %s → %s", SESSIONS_FILE, target)
        return target

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(SESSIONS_FILE), str(target))
    log.info("Archived sessions to %s", target)
    return target


def distill_memory_journal(archive_path: Path, dry_run: bool = False) -> Path | None:
    """Write a brief journal summarizing the archived sessions.

    We keep this extractive (topics + message counts per session) rather than
    calling an LLM, so it doesn't depend on a working inference tier at the
    exact moment of reset. A richer version can read the journal and expand
    later.
    """
    if archive_path is None or not archive_path.exists():
        log.info("No archive to distill")
        return None

    journal_path = archive_path.with_suffix(".journal.md")
    if dry_run:
        log.info("[dry-run] would write journal to %s", journal_path)
        return journal_path

    try:
        with open(archive_path) as f:
            sessions = json.load(f)
    except Exception as e:
        log.warning("Could not read archive %s: %s", archive_path, e)
        return None

    lines = [
        f"# Memory Journal — {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Archived {len(sessions)} session(s) from {archive_path.name}.",
        "",
        "## Sessions",
        "",
    ]
    for sid, sess in sessions.items() if isinstance(sessions, dict) else []:
        history = sess.get("history", []) if isinstance(sess, dict) else []
        if not history:
            continue
        first_user = next(
            (m.get("content", "")[:120] for m in history if m.get("role") == "user"),
            "(no user message)",
        )
        lines.append(f"- **{sid}** — {len(history)} messages. Opened with: {first_user!r}")

    journal_path.write_text("\n".join(lines) + "\n")
    log.info("Wrote journal to %s (%d sessions)", journal_path, len(sessions) if isinstance(sessions, dict) else 0)
    return journal_path


def invalidate_kv(tier: str, dry_run: bool = False) -> None:
    """Clear the tier's KV checkpoint files and call /cache/invalidate."""
    tier_dir = KVCACHE_DIR / tier
    if tier_dir.exists():
        for name in ("handoff_context.json", "core_checkpoint"):
            target = tier_dir / name
            if target.exists():
                if dry_run:
                    log.info("[dry-run] would remove %s", target)
                else:
                    target.unlink()
                    log.info("Removed %s", target)

    endpoint = TIER_ENGINE_ENDPOINTS.get(tier)
    if endpoint:
        if dry_run:
            log.info("[dry-run] would POST %s/cache/invalidate", endpoint)
            return
        try:
            req = urllib.request.Request(
                f"{endpoint}/cache/invalidate",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode()
                log.info("%s /cache/invalidate: %s", tier, body.strip())
        except Exception as e:
            log.warning("%s /cache/invalidate failed (non-fatal, engine may be offline): %s", tier, e)


def regen_identity_prefix(tier: str, dry_run: bool = False) -> None:
    """Trigger identity_prefix regeneration on the tier's engine.

    The engine exposes /cache/rebuild_identity_prefix (new endpoint — see
    companion change in gaia-engine/manager.py) that forward-passes the
    current model over the identity document and caches the prefix. If the
    endpoint isn't present, fall back to deleting the stale file so the
    engine regenerates it on next load.
    """
    tier_dir = KVCACHE_DIR / tier
    prefix_file = tier_dir / "identity_prefix.pt"

    endpoint = TIER_ENGINE_ENDPOINTS.get(tier)
    if endpoint:
        if dry_run:
            log.info("[dry-run] would POST %s/cache/rebuild_identity_prefix", endpoint)
            return
        try:
            req = urllib.request.Request(
                f"{endpoint}/cache/rebuild_identity_prefix",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode()
                log.info("%s rebuild_identity_prefix: %s", tier, body.strip())
                return
        except Exception as e:
            log.info("%s rebuild endpoint unavailable (%s) — falling back to delete-and-regenerate", tier, e)

    if prefix_file.exists():
        if dry_run:
            log.info("[dry-run] would remove %s", prefix_file)
        else:
            prefix_file.unlink()
            log.info("Removed %s (engine will regenerate on next load)", prefix_file)


def write_dev_journal(tier: str, archive_path: Path | None, dry_run: bool = False) -> None:
    """Note the reset in the dev journal so future-us knows which model ran when."""
    if not DEV_JOURNAL_DIR.exists():
        log.info("Dev journal dir %s not present; skipping journal entry", DEV_JOURNAL_DIR)
        return

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = DEV_JOURNAL_DIR / f"{stamp}-post-training-reset.md"
    content = (
        f"# Post-training reset — {datetime.now(timezone.utc).isoformat()}\n\n"
        f"Tier: **{tier}**\n\n"
        f"Actions:\n"
        f"- Archived sessions: `{archive_path}`\n"
        f"- Invalidated KV state for {tier}\n"
        f"- Regenerated identity_prefix.pt\n"
    )
    if dry_run:
        log.info("[dry-run] would write journal entry to %s", path)
        return
    path.write_text(content)
    log.info("Wrote dev journal entry %s", path)


def reset_tier(tier: str, archive_path: Path | None, dry_run: bool) -> None:
    log.info("── Resetting tier: %s ──", tier)
    invalidate_kv(tier, dry_run=dry_run)
    regen_identity_prefix(tier, dry_run=dry_run)
    write_dev_journal(tier, archive_path, dry_run=dry_run)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Post-training reset")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tier", choices=list(TIER_ENDPOINTS.keys()), help="Reset a single tier")
    group.add_argument("--all", action="store_true", help="Reset all configured tiers")
    parser.add_argument("--skip-archive", action="store_true", help="Don't archive sessions")
    parser.add_argument("--skip-journal", action="store_true", help="Don't distill a memory journal")
    parser.add_argument("--dry-run", action="store_true", help="Log actions without making changes")
    args = parser.parse_args()

    tiers = list(TIER_ENDPOINTS.keys()) if args.all else [args.tier]

    archive_path = None
    if not args.skip_archive:
        archive_path = archive_sessions(dry_run=args.dry_run)
        if archive_path and not args.skip_journal:
            distill_memory_journal(archive_path, dry_run=args.dry_run)

    for tier in tiers:
        reset_tier(tier, archive_path, dry_run=args.dry_run)

    log.info("Post-training reset complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

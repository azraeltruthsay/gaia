#!/usr/bin/env python3
"""Mirror GAIA's issue ledger into beads, and bead status back to her (a0aw.1).

She writes /shared/issues/ledger.jsonl from inside gaia-mcp (issue_ledger.py).
The beads database is host-only (embedded Dolt), so this runs on the host:

  - each new `file` event  -> `bd create` (type bug, labels gaia-found,self-improvement)
  - each `note` event      -> `bd note <bead> "GAIA: ..."`
  - each `status` event    -> investigating/proposed: bead in_progress + note;
                              fixed: note + label gaia-fixed-unverified (a person or
                              the promote step closes it); wontfix: note
  - then every mapped bead's current status goes back into beads.json, so she can
    see when an issue she filed was closed.

Events are mirrored in order and the offset only advances past an event once
`bd` accepted it, so a failure retries on the next run. The shared volume is
reached through `docker exec` into the container (no root needed).

Usage: python3 scripts/gaia_issue_bridge.py [--container gaia-mcp] [--dry-run]
Run by the gaia-issue-bridge user timer (scripts/systemd/), or by hand.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

ISSUES_DIR = "/shared/issues"
LABELS = "gaia-found,self-improvement"


def _exec(container: str, cmd: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", "exec", "-i", container, "sh", "-c", cmd],
                          input=stdin, capture_output=True, text=True, timeout=30)


def read_shared(container: str, name: str) -> str:
    r = _exec(container, f"cat {ISSUES_DIR}/{name} 2>/dev/null || true")
    if r.returncode != 0:
        raise RuntimeError(f"docker exec {container} failed: {r.stderr.strip()}")
    return r.stdout


def write_shared(container: str, name: str, text: str) -> None:
    path = f"{ISSUES_DIR}/{name}"
    r = _exec(container, f"mkdir -p {ISSUES_DIR} && cat > {path}.tmp && mv {path}.tmp {path}", stdin=text)
    if r.returncode != 0:
        raise RuntimeError(f"writing {path} failed: {r.stderr.strip()}")


def bd(*args: str, dry: bool = False) -> str:
    if dry:
        print("  [dry-run] bd", " ".join(a if len(a) < 60 else a[:57] + "..." for a in args))
        return "DRY-" + str(abs(hash(args)) % 10000)
    r = subprocess.run(["bd", *args], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"bd {args[0]} failed: {(r.stderr or r.stdout).strip()}")
    return r.stdout.strip()


def describe(ev: dict) -> str:
    parts = [f"Filed by GAIA ({ev['id']}, source: {ev.get('source') or 'self'}).", "",
             f"Symptom: {ev.get('symptom', '')}", "", "Evidence (what she was shown):"]
    parts += [f"    {line}" for line in ev.get("evidence", [])]
    if ev.get("suspected_cause"):
        parts += ["", f"Suspected cause: {ev['suspected_cause']}"]
    if ev.get("playbook"):
        parts += ["", f"Playbook: {ev['playbook']}"]
    parts += ["", "Self-repair loop: knowledge/blueprints/SELF_REPAIR_LOOP.md"]
    return "\n".join(parts)


def mirror(ev: dict, mapping: dict, dry: bool) -> None:
    kind, iid = ev.get("event"), ev.get("id")
    if kind == "file":
        bead = bd("create", ev.get("title") or iid, "-t", "bug", "-p", "2", "-l", LABELS,
                  "--external-ref", iid, "--description", describe(ev), "--silent", dry=dry)
        mapping[iid] = {"bead": bead.splitlines()[-1].strip(), "bead_status": "open"}
        print(f"  {iid} -> {mapping[iid]['bead']}")
        return
    bead = (mapping.get(iid) or {}).get("bead")
    if not bead:
        print(f"  skip {kind} for unmapped {iid}")
        return
    if kind == "note":
        bd("note", bead, f"GAIA: {ev.get('text', '')}", dry=dry)
    elif kind == "status":
        status, note = ev.get("status"), ev.get("note") or ""
        text = f"GAIA set status '{status}'" + (f": {note}" if note else "")
        if status in ("investigating", "proposed"):
            bd("update", bead, "--status", "in_progress", dry=dry)
        elif status == "fixed":
            bd("label", "add", bead, "gaia-fixed-unverified", dry=dry)
        bd("note", bead, text, dry=dry)
    print(f"  {iid} {kind} -> {bead}")


def refresh_status(mapping: dict, dry: bool) -> None:
    for entry in mapping.values():
        bead = entry.get("bead", "")
        if dry or not bead or bead.startswith("DRY-"):
            continue
        try:
            data = json.loads(bd("show", bead, "--json"))
            row = data[0] if isinstance(data, list) else data
            entry["bead_status"] = row.get("status")
            if row.get("close_reason"):
                entry["close_reason"] = row["close_reason"]
        except (RuntimeError, ValueError, IndexError) as e:
            print(f"  status of {bead} unavailable: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--container", default="gaia-mcp")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    events = [json.loads(line) for line in read_shared(args.container, "ledger.jsonl").splitlines()
              if line.strip()]
    try:
        state = json.loads(read_shared(args.container, "beads.json") or "{}")
    except json.JSONDecodeError:
        state = {}
    mapping = state.get("issues", {})
    offset = int(state.get("ledger_offset", 0))

    pending = events[offset:]
    print(f"ledger: {len(events)} events, {len(pending)} new")
    error = None
    for ev in pending:
        try:
            mirror(ev, mapping, args.dry_run)
        except RuntimeError as e:
            error = e
            print(f"  stopped at event {offset}: {e}", file=sys.stderr)
            break
        offset += 1

    refresh_status(mapping, args.dry_run)
    if not args.dry_run:
        write_shared(args.container, "beads.json", json.dumps(
            {"issues": mapping, "ledger_offset": offset, "synced_at": time.time()}, indent=1))
    return 1 if error else 0


if __name__ == "__main__":
    sys.exit(main())

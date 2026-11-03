#!/usr/bin/env python3
"""Review what GAIA keeps reaching for (yi22). Run inside gaia-core:

  docker exec gaia-core python /app/scripts/tool_gaps.py list
  docker exec gaia-core python /app/scripts/tool_gaps.py approve ID --to search   # her verb becomes an alias (or --to study.train)
  docker exec gaia-core python /app/scripts/tool_gaps.py approve ID --new         # CodeMind drafts it, for review
  docker exec gaia-core python /app/scripts/tool_gaps.py reject ID

Aliases take effect after restarting gaia-mcp and gaia-core.
"""
import argparse
import sys

from gaia_common.utils import tool_gaps
from gaia_common.utils.domain_tools import DOMAIN_TOOLS


def show(store: dict) -> None:
    if not store:
        print("Nothing proposed yet.")
    for pid, r in sorted(store.items(), key=lambda kv: (kv[1]["status"] != "pending", -kv[1]["attempts"])):
        print(f"{pid}  [{r['status']}]  {r['kind']}: {r['tool']}({r['verb']})  params={r['params']}  "
              f"{r['attempts']} attempts on {r['occasions']} occasions, {r['outside_tests']} outside tests")
        if r["status"] == "pending" and r.get("candidates"):
            print(f"          takes the same params: {', '.join(r['candidates'])}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    a = sub.add_parser("approve")
    a.add_argument("id")
    how = a.add_mutually_exclusive_group(required=True)
    how.add_argument("--to", help="existing action her verb should reach: action, or tool.action")
    how.add_argument("--new", action="store_true", help="have CodeMind draft it under her name")
    sub.add_parser("reject").add_argument("id")
    args = ap.parse_args()

    store = tool_gaps.load()
    if args.cmd == "list":
        show(store)
        return 0
    rec = store.get(args.id)
    if not rec:
        print(f"No proposal {args.id}", file=sys.stderr)
        return 1
    if args.cmd == "reject":
        tool_gaps.decide(args.id, "rejected")
        print(f"Rejected {rec['tool']}({rec['verb']}); it won't be proposed again.")
        return 0
    if args.to:
        tool, _dot, action = args.to.rpartition(".")
        if action not in (DOMAIN_TOOLS.get(tool or rec["tool"]) or {}).get("actions", {}):
            print(f"{tool or rec['tool']} has no action '{action}'", file=sys.stderr)
            return 1
        tool_gaps.decide(args.id, "approved", target=args.to)
        print(f"{rec['tool']}({rec['verb']}) will reach {args.to}. Restart gaia-mcp and gaia-core.")
        return 0
    from gaia_common.utils.codemind_detector import emit_detection
    emit_detection("tool_gaps", "capability_gap", "", tool_gaps.capability_gap_text(rec), severity="info")
    tool_gaps.decide(args.id, "approved")
    print(f"Queued {rec['tool']}({rec['verb']}) for CodeMind to draft.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

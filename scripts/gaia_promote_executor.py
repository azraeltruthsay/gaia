#!/usr/bin/env python3
"""
gaia_promote_executor.py — Two-gate promotion executor for GAIA services.

Reads a promotion request JSON, validates the approval state, and executes
the promotion pipeline. Called by the human operator, not by GAIA directly.

Usage:
    # List pending requests
    python scripts/gaia_promote_executor.py --list

    # Gate 1: Approve a request
    python scripts/gaia_promote_executor.py --approve <request_id>

    # Run dry-run (after approval)
    python scripts/gaia_promote_executor.py --dry-run <request_id>

    # Gate 2: Confirm (after dry-run passes)
    python scripts/gaia_promote_executor.py --confirm <request_id>

    # Execute live promotion (after confirmation)
    python scripts/gaia_promote_executor.py --execute <request_id>

    # Reject at any stage
    python scripts/gaia_promote_executor.py --reject <request_id> --reason "not ready"

    # Shortcut: approve + dry-run + confirm + execute in one shot
    python scripts/gaia_promote_executor.py --auto <request_id>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "candidates" / "gaia-common"))
sys.path.insert(0, str(PROJECT_ROOT / "gaia-common"))

from gaia_common.utils.promotion_request import (
    approve_request,
    confirm_request,
    list_requests,
    load_request,
    record_dry_run,
    record_promotion,
    reject_request,
)


def _run_pipeline(service_id: str, dry_run: bool = False) -> tuple[bool, str]:
    """Run the promotion pipeline and return (success, output)."""
    script = PROJECT_ROOT / "scripts" / "promote_pipeline.sh"
    if not script.exists():
        return False, f"Pipeline script not found: {script}"

    cmd = [str(script), "--services", service_id]
    if dry_run:
        cmd.append("--dry-run")

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=False,  # Stream output to terminal
            text=True,
            timeout=600,  # 10 minute timeout
        )
        success = result.returncode == 0
        return success, f"Exit code: {result.returncode}"
    except subprocess.TimeoutExpired:
        return False, "Pipeline timed out after 10 minutes"
    except Exception as exc:
        return False, f"Pipeline error: {exc}"


def cmd_list(args):
    """List promotion requests."""
    requests = list_requests(
        service_id=args.service if hasattr(args, "service") and args.service else None,
    )
    if not requests:
        print("No promotion requests found.")
        return

    print(f"\n{'ID':<50} {'Service':<15} {'Status':<18} {'Verdict':<20}")
    print("-" * 103)
    for req in requests:
        print(f"{req.request_id:<50} {req.service_id:<15} {req.status:<18} {req.verdict:<20}")
    print()


def cmd_show(args):
    """Show details of a promotion request."""
    req = load_request(args.request_id)
    if req is None:
        print(f"Request not found: {args.request_id}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Promotion Request: {req.request_id}")
    print(f"{'='*60}")
    print(f"  Service:        {req.service_id}")
    print(f"  Status:         {req.status}")
    print(f"  Verdict:        {req.verdict}")
    print(f"  Requested:      {req.requested_at}")
    print(f"  Recommendation: {req.recommendation}")
    print(f"  Pipeline cmd:   {req.pipeline_cmd}")
    if req.approved_by:
        print(f"  Approved by:    {req.approved_by} at {req.approved_at}")
    if req.confirmed_at:
        print(f"  Confirmed at:   {req.confirmed_at}")
    if req.promoted_at:
        print(f"  Promoted at:    {req.promoted_at}")
    if req.rejection_reason:
        print(f"  Rejected:       {req.rejection_reason}")
    print(f"\n  Check Summary:")
    for line in req.check_summary.split("\n"):
        print(f"    {line}")
    if req.history:
        print(f"\n  History:")
        for entry in req.history:
            print(f"    [{entry['timestamp']}] {entry['action']}: {entry.get('detail', '')}")
    print()


def cmd_approve(args):
    """Gate 1: Approve a promotion request."""
    try:
        req = approve_request(args.request_id, approved_by=args.approver or "human")
        print(f"Approved: {req.request_id}")
        print(f"Next step: python scripts/gaia_promote_executor.py --dry-run {req.request_id}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)


def cmd_dry_run(args):
    """Run dry-run after approval."""
    req = load_request(args.request_id)
    if req is None:
        print(f"Request not found: {args.request_id}")
        sys.exit(1)
    if req.status != "approved":
        print(f"Request must be in 'approved' state (current: {req.status})")
        sys.exit(1)

    success, output = _run_pipeline(req.service_id, dry_run=True)
    req = record_dry_run(args.request_id, passed=success, output=output)

    if success:
        print(f"\nDry-run PASSED for {req.service_id}")
        print(f"Next step: python scripts/gaia_promote_executor.py --confirm {req.request_id}")
    else:
        print(f"\nDry-run FAILED for {req.service_id}: {output}")
        sys.exit(1)


def cmd_confirm(args):
    """Gate 2: Confirm after dry-run."""
    try:
        req = confirm_request(args.request_id)
        print(f"Confirmed: {req.request_id}")
        print(f"Next step: python scripts/gaia_promote_executor.py --execute {req.request_id}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)


def cmd_execute(args):
    """Execute live promotion after confirmation."""
    req = load_request(args.request_id)
    if req is None:
        print(f"Request not found: {args.request_id}")
        sys.exit(1)
    if req.status != "confirmed":
        print(f"Request must be in 'confirmed' state (current: {req.status})")
        sys.exit(1)

    print(f"\n*** EXECUTING LIVE PROMOTION for {req.service_id} ***\n")
    success, output = _run_pipeline(req.service_id, dry_run=False)
    req = record_promotion(args.request_id, success=success, detail=output)

    if success:
        print(f"\nPromotion SUCCEEDED for {req.service_id}")
    else:
        print(f"\nPromotion FAILED for {req.service_id}: {output}")
        sys.exit(1)


def cmd_reject(args):
    """Reject a promotion request."""
    try:
        req = reject_request(args.request_id, reason=args.reason or "")
        print(f"Rejected: {req.request_id}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)


def cmd_auto(args):
    """Shortcut: approve + dry-run + confirm + execute."""
    print(f"=== Auto-promoting {args.request_id} ===\n")

    # Approve
    try:
        req = approve_request(args.request_id, approved_by=args.approver or "human-auto")
        print(f"[1/4] Approved")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Approve failed: {exc}")
        sys.exit(1)

    # Dry-run
    success, output = _run_pipeline(req.service_id, dry_run=True)
    req = record_dry_run(args.request_id, passed=success, output=output)
    if not success:
        print(f"[2/4] Dry-run FAILED: {output}")
        sys.exit(1)
    print(f"[2/4] Dry-run passed")

    # Confirm
    try:
        req = confirm_request(args.request_id)
        print(f"[3/4] Confirmed")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Confirm failed: {exc}")
        sys.exit(1)

    # Execute
    success, output = _run_pipeline(req.service_id, dry_run=False)
    req = record_promotion(args.request_id, success=success, detail=output)
    if success:
        print(f"[4/4] Promotion SUCCEEDED for {req.service_id}")
    else:
        print(f"[4/4] Promotion FAILED: {output}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="GAIA Promotion Executor — Two-gate approval flow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # list
    p_list = sub.add_parser("list", help="List promotion requests")
    p_list.add_argument("--service", help="Filter by service ID")

    # show
    p_show = sub.add_parser("show", help="Show request details")
    p_show.add_argument("request_id", help="Request ID")

    # approve
    p_approve = sub.add_parser("approve", help="Gate 1: Approve a request")
    p_approve.add_argument("request_id", help="Request ID")
    p_approve.add_argument("--approver", default="human", help="Who is approving")

    # dry-run
    p_dry = sub.add_parser("dry-run", help="Run dry-run after approval")
    p_dry.add_argument("request_id", help="Request ID")

    # confirm
    p_confirm = sub.add_parser("confirm", help="Gate 2: Confirm after dry-run")
    p_confirm.add_argument("request_id", help="Request ID")

    # execute
    p_exec = sub.add_parser("execute", help="Execute live promotion")
    p_exec.add_argument("request_id", help="Request ID")

    # reject
    p_reject = sub.add_parser("reject", help="Reject a request")
    p_reject.add_argument("request_id", help="Request ID")
    p_reject.add_argument("--reason", default="", help="Rejection reason")

    # auto
    p_auto = sub.add_parser("auto", help="Approve + dry-run + confirm + execute")
    p_auto.add_argument("request_id", help="Request ID")
    p_auto.add_argument("--approver", default="human", help="Who is approving")

    # Also support --flag style for convenience
    parser.add_argument("--list", action="store_true", help="List requests")
    parser.add_argument("--approve", metavar="ID", help="Approve request")
    parser.add_argument("--dry-run", metavar="ID", help="Run dry-run")
    parser.add_argument("--confirm", metavar="ID", help="Confirm request")
    parser.add_argument("--execute", metavar="ID", help="Execute promotion")
    parser.add_argument("--reject", metavar="ID", help="Reject request")
    parser.add_argument("--auto", metavar="ID", help="Auto-promote")
    parser.add_argument("--show", metavar="ID", help="Show request details")
    parser.add_argument("--reason", default="", help="Rejection reason")
    parser.add_argument("--approver", default="human", help="Approver name")
    parser.add_argument("--service", help="Service filter for --list")

    args = parser.parse_args()

    # Handle --flag style
    if args.list:
        cmd_list(args)
    elif args.show:
        args.request_id = args.show
        cmd_show(args)
    elif args.approve:
        args.request_id = args.approve
        cmd_approve(args)
    elif getattr(args, "dry_run", None):
        args.request_id = args.dry_run
        cmd_dry_run(args)
    elif args.confirm:
        args.request_id = args.confirm
        cmd_confirm(args)
    elif args.execute:
        args.request_id = args.execute
        cmd_execute(args)
    elif args.reject:
        args.request_id = args.reject
        cmd_reject(args)
    elif getattr(args, "auto", None) and isinstance(args.auto, str):
        args.request_id = args.auto
        cmd_auto(args)
    elif args.command:
        # Subcommand style
        cmd_map = {
            "list": cmd_list,
            "show": cmd_show,
            "approve": cmd_approve,
            "dry-run": cmd_dry_run,
            "confirm": cmd_confirm,
            "execute": cmd_execute,
            "reject": cmd_reject,
            "auto": cmd_auto,
        }
        handler = cmd_map.get(args.command)
        if handler:
            handler(args)
        else:
            parser.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

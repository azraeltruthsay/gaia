#!/usr/bin/env python3
"""
GaiaCLI — Unified System Management Interface (Phase 5-C, Proposal 04)

Replaces gaia-startup.sh, promote_candidate.sh, and gpu_maintenance.sh
with a single Python CLI for consistent stack management.

Usage:
    python3 scripts/gaia_cli.py live start
    python3 scripts/gaia_cli.py live status
    python3 scripts/gaia_cli.py promote gaia-core --validate
    python3 scripts/gaia_cli.py gpu status
    python3 scripts/gaia_cli.py clean prune
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

GAIA_ROOT = Path(__file__).resolve().parent.parent
os.chdir(GAIA_ROOT)

# ── Service Configuration ──────────────────────────────────────────────

SERVICE_CONFIG = {
    "gaia-core":         {"live_port": 6415, "candidate_port": 6416, "has_container": True},
    "gaia-prime":        {"live_port": 7777, "candidate_port": 7778, "has_container": True},
    "gaia-mcp":          {"live_port": 8765, "candidate_port": 8767, "has_container": True},
    "gaia-study":        {"live_port": 8766, "candidate_port": 8768, "has_container": True},
    "gaia-web":          {"live_port": 6414, "candidate_port": None, "has_container": True},
    "gaia-orchestrator": {"live_port": 6410, "candidate_port": 6411, "has_container": True},
    "gaia-audio":        {"live_port": 8080, "candidate_port": 8081, "has_container": True},
    "gaia-common":       {"live_port": None, "candidate_port": None, "has_container": False},
}

PROMOTABLE_SERVICES = [
    "gaia-common", "gaia-core", "gaia-mcp", "gaia-study",
    "gaia-web", "gaia-audio", "gaia-orchestrator", "gaia-prime",
]

# Dependency order for promotion
PROMOTE_ORDER = [
    "gaia-common", "gaia-core", "gaia-web", "gaia-mcp",
    "gaia-study", "gaia-orchestrator",
]

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_ENDPOINT", "http://localhost:6410")


# ── Helpers ────────────────────────────────────────────────────────────

def run(cmd, capture=False, check=True, **kwargs):
    """Run a shell command with sane defaults."""
    if isinstance(cmd, str):
        cmd = cmd.split()
    kwargs.setdefault("text", True)
    if capture:
        kwargs["capture_output"] = True
    result = subprocess.run(cmd, check=check, **kwargs)
    return result


def docker_compose(*args, capture=False, check=True):
    """Run docker compose with the project's env file."""
    cmd = ["docker", "compose", "--env-file", ".env.discord"] + list(args)
    return run(cmd, capture=capture, check=check)


def curl_health(port, path="/health", timeout=3):
    """Quick health check via curl. Returns response text or None."""
    try:
        r = run(
            ["curl", "-sf", "--max-time", str(timeout), f"http://localhost:{port}{path}"],
            capture=True, check=False,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def file_hash(path):
    """SHA256 hash of a file's contents."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def print_header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_ok(msg):
    print(f"  OK  {msg}")


def print_fail(msg):
    print(f"  FAIL  {msg}")


def print_warn(msg):
    print(f"  WARN  {msg}")


# ═══════════════════════════════════════════════════════════════════════
# LIVE — Stack Management
# ═══════════════════════════════════════════════════════════════════════

def cmd_live(args):
    action = args.action

    if action == "start":
        print_header("Starting GAIA Live Stack")
        docker_compose("up", "-d")
        print("\nWaiting for health checks...")
        time.sleep(5)
        _print_service_health()

    elif action == "stop":
        print_header("Stopping GAIA Live Stack")
        docker_compose("down", "-t", "20")
        print("Stack stopped.")

    elif action == "status":
        print_header("GAIA Stack Status")
        _print_service_health()
        _print_sovereign_health()
        _print_container_status()

    elif action == "logs":
        service = getattr(args, "service", None)
        tail = str(getattr(args, "tail", 50))
        if service:
            run(["docker", "logs", service, "--tail", tail, "-f"], check=False)
        else:
            docker_compose("logs", "--tail", tail, "-f", check=False)


def _print_service_health():
    """Check health endpoints for all services."""
    print("Service Health:")
    for name, cfg in SERVICE_CONFIG.items():
        port = cfg["live_port"]
        if not port or not cfg["has_container"]:
            continue
        health = curl_health(port)
        if health:
            try:
                data = json.loads(health)
                status = data.get("status", "ok")
                print_ok(f"{name}:{port} — {status}")
            except json.JSONDecodeError:
                print_ok(f"{name}:{port} — {health[:60]}")
        else:
            print_fail(f"{name}:{port} — unreachable")
    print()


def _print_sovereign_health():
    """Print GaiaVitals sovereign health if available."""
    print("Sovereign Health:")
    try:
        sys.path.insert(0, str(GAIA_ROOT / "gaia-common"))
        from gaia_common.utils.vitals import GaiaVitals
        vitals = GaiaVitals()
        health = vitals.get_sovereign_health(verbose=True)
        status = health.get("sovereign_status", "UNKNOWN")
        score = health.get("irritation_score", 0.0)
        print(f"  Status: {status} (irritation={score})")
        for domain, pulse in health.get("pulses", {}).items():
            ds = pulse.get("status", "?")
            print(f"    {domain}: {ds}")
    except Exception as e:
        print(f"  unavailable ({e})")
    print()


def _print_container_status():
    """Print docker compose ps output."""
    print("Containers:")
    docker_compose("ps", "--format", "table {{.Name}}\t{{.Status}}\t{{.Health}}", check=False)
    print()


# ═══════════════════════════════════════════════════════════════════════
# TEST — Candidate Testing
# ═══════════════════════════════════════════════════════════════════════

def cmd_test(args):
    service = args.service
    print_header(f"Testing Candidate: {service}")

    candidate_dir = GAIA_ROOT / "candidates" / service
    if not candidate_dir.exists():
        print_fail(f"Candidate directory not found: {candidate_dir}")
        sys.exit(1)

    # Syntax validation
    print("Syntax validation:")
    py_files = list(candidate_dir.rglob("*.py"))
    py_files = [f for f in py_files if "__pycache__" not in str(f)]
    errors = 0
    for f in py_files:
        try:
            import ast
            ast.parse(f.read_text())
        except SyntaxError as e:
            print_fail(f"{f.relative_to(GAIA_ROOT)}: {e}")
            errors += 1

    if errors:
        print(f"\n{errors} syntax error(s) found.")
        sys.exit(1)
    print_ok(f"{len(py_files)} Python files — all clean")

    # Unit tests (if --unit flag)
    if args.unit:
        print("\nUnit tests:")
        test_dir = candidate_dir / "tests"
        if test_dir.exists():
            result = run(
                ["python", "-m", "pytest", str(test_dir), "--no-header", "-q"],
                check=False, capture=True,
            )
            print(result.stdout)
            if result.returncode not in (0, 5):  # 5 = no tests collected
                print_fail("Tests failed")
                sys.exit(1)
            print_ok("Tests passed" if result.returncode == 0 else "No tests found")
        else:
            print_warn("No tests/ directory")

    # Containerized validation (if --inject)
    if args.inject:
        print(f"\nContainerized validation (building temp image)...")
        dockerfile = candidate_dir / "Dockerfile"
        if not dockerfile.exists():
            dockerfile = GAIA_ROOT / service / "Dockerfile"
        if dockerfile.exists():
            img_name = f"gaia-test-{service}:{int(time.time())}"
            result = run(
                ["docker", "build", "-t", img_name, "-f", str(dockerfile), str(GAIA_ROOT)],
                check=False,
            )
            if result.returncode == 0:
                print_ok(f"Image built: {img_name}")
                # Run ruff inside container
                run(["docker", "run", "--rm", img_name, "python", "-m", "ruff", "check", "/app"], check=False)
                # Cleanup
                run(["docker", "rmi", img_name], check=False, capture=True)
            else:
                print_fail("Docker build failed")
        else:
            print_warn("No Dockerfile found for containerized validation")

    print("\nTest complete.")


# ═══════════════════════════════════════════════════════════════════════
# PROMOTE — Candidate to Production
# ═══════════════════════════════════════════════════════════════════════

def cmd_promote(args):
    service = args.service
    if service not in PROMOTABLE_SERVICES:
        print_fail(f"Unknown service: {service}")
        print(f"Available: {', '.join(PROMOTABLE_SERVICES)}")
        sys.exit(1)

    print_header(f"Promoting: {service}")

    candidate_dir = GAIA_ROOT / "candidates" / service
    live_dir = GAIA_ROOT / service

    if service != "gaia-common" and not candidate_dir.exists():
        print_fail(f"Candidate directory not found: {candidate_dir}")
        sys.exit(1)

    # gaia-common sync check (for non-common services)
    if service != "gaia-common":
        sync_ok = _check_common_sync(args.force)
        if not sync_ok:
            sys.exit(1)

    # Containerized validation (--validate)
    if args.validate:
        print("Running containerized validation...")
        cmd_test(argparse.Namespace(service=service, unit=True, inject=True, gpu=False))

    # Backup
    backup_dir = GAIA_ROOT / f"{service}.bak"
    if live_dir.exists():
        print(f"Creating backup: {backup_dir.name}/")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(live_dir, backup_dir)
        print_ok("Backup created")

    # Promote via rsync
    print(f"Promoting: {candidate_dir}/ → {live_dir}/")
    result = run(
        [
            "rsync", "-av", "--no-group", "--no-owner",
            "--exclude", ".git", "--exclude", "__pycache__", "--exclude", "*.pyc",
            f"{candidate_dir}/", f"{live_dir}/",
        ],
        check=False,
    )
    if result.returncode in (0, 23):  # 23 = some permission issues (non-fatal)
        print_ok("Files promoted")
    else:
        print_fail(f"rsync failed (exit {result.returncode})")
        sys.exit(1)

    # Restart container
    cfg = SERVICE_CONFIG.get(service, {})
    if cfg.get("has_container") and not args.no_restart:
        print(f"Restarting {service}...")
        run(["docker", "restart", service], check=False)
        time.sleep(5)
        port = cfg.get("live_port")
        if port:
            health = curl_health(port)
            if health:
                print_ok(f"{service} is healthy")
            else:
                print_warn(f"{service} not yet healthy (may still be starting)")

    print(f"\nPromotion complete. Rollback: cp -r {backup_dir}/* {live_dir}/")


def _check_common_sync(force=False):
    """Verify gaia-common is in sync between candidate and live."""
    print("Checking gaia-common sync...")
    cand_common = GAIA_ROOT / "candidates" / "gaia-common"
    live_common = GAIA_ROOT / "gaia-common"

    sync_failed = False

    # Critical: cognition_packet.py must match
    for critical in ["gaia_common/protocols/cognition_packet.py"]:
        cand_file = cand_common / critical
        live_file = live_common / critical
        if cand_file.exists() and live_file.exists():
            if file_hash(cand_file) != file_hash(live_file):
                print_fail(f"{critical} — OUT OF SYNC (blocking)")
                sync_failed = True
            else:
                print_ok(f"{critical} in sync")

    # Warning: gaia_constants.json
    for warn_file in ["gaia_common/constants/gaia_constants.json"]:
        cand_file = cand_common / warn_file
        live_file = live_common / warn_file
        if cand_file.exists() and live_file.exists():
            if file_hash(cand_file) != file_hash(live_file):
                print_warn(f"{warn_file} differs (non-blocking)")
            else:
                print_ok(f"{warn_file} in sync")

    if sync_failed and not force:
        print_fail("BLOCKING: Critical shared files are out of sync.")
        print("  Fix: Run 'gaia_cli.py promote gaia-common' first, or use --force.")
        return False

    if sync_failed and force:
        print_warn("--force specified, continuing despite sync mismatch")

    return True


# ═══════════════════════════════════════════════════════════════════════
# GPU — GPU Lifecycle Management
# ═══════════════════════════════════════════════════════════════════════

def cmd_gpu(args):
    action = args.action

    if action == "status":
        print_header("GPU Status")
        # nvidia-smi
        run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"], check=False)
        print()
        # Orchestrator GPU status
        health = curl_health(6410, "/gpu/status", timeout=5)
        if health:
            data = json.loads(health)
            print(f"  Owner: {data.get('owner', 'none')}")
            print(f"  Queue: {data.get('queue', [])}")
            mem = data.get("memory", {})
            if mem:
                print(f"  VRAM: {mem.get('used_mb', '?')}MB / {mem.get('total_mb', '?')}MB")
        else:
            print("  Orchestrator unreachable")

    elif action == "release":
        print_header("Releasing GPU (entering maintenance)")
        _gpu_maintenance("enter")

    elif action == "reclaim":
        print_header("Reclaiming GPU (exiting maintenance)")
        _gpu_maintenance("exit")

    elif action == "handoff":
        target = getattr(args, "target", "study")
        print_header(f"GPU Handoff → {target}")
        try:
            from urllib.request import Request, urlopen
            req = Request(
                f"{ORCHESTRATOR_URL}/consciousness/focusing",
                data=b"", method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
            print(f"  Result: {json.dumps(result, indent=2)[:300]}")
        except Exception as e:
            print_fail(f"Handoff failed: {e}")


def _gpu_maintenance(mode):
    """Enter or exit GPU maintenance mode."""
    maintenance_flag = Path("/shared/maintenance_mode.json")

    if mode == "enter":
        # Set flag
        maintenance_flag.parent.mkdir(parents=True, exist_ok=True)
        maintenance_flag.write_text(json.dumps({
            "active": True,
            "entered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reason": "gaia_cli gpu release",
        }))

        # Unload models from GPU services
        gpu_services = ["gaia-core", "gaia-nano", "gaia-prime", "gaia-audio"]
        for svc in gpu_services:
            try:
                port = SERVICE_CONFIG.get(svc, {}).get("live_port")
                if port:
                    run(
                        ["curl", "-sf", "-X", "POST", f"http://localhost:{port}/model/unload"],
                        check=False, capture=True,
                    )
                    print_ok(f"Unloaded {svc}")
            except Exception:
                print_warn(f"Failed to unload {svc}")

        # Stop GPU containers
        for svc in gpu_services:
            run(["docker", "stop", svc], check=False, capture=True)
            print_ok(f"Stopped {svc}")

    elif mode == "exit":
        # Clear flag
        if maintenance_flag.exists():
            maintenance_flag.unlink()

        # Restart GPU containers
        gpu_services = ["gaia-orchestrator", "gaia-core", "gaia-nano", "gaia-prime", "gaia-audio"]
        for svc in gpu_services:
            docker_compose("up", "-d", svc, check=False)
            print_ok(f"Started {svc}")


# ═══════════════════════════════════════════════════════════════════════
# CLEAN — Cleanup Operations
# ═══════════════════════════════════════════════════════════════════════

def cmd_clean(args):
    action = args.action

    if action == "prune":
        print_header("Pruning Docker Resources")
        run(["docker", "system", "prune", "-f"], check=False)
        run(["docker", "image", "prune", "-f"], check=False)
        print_ok("Docker pruned")

    elif action == "logs":
        print_header("Cleaning Log Files")
        log_dirs = [
            Path("/gaia/gaia-instance/logs"),
            GAIA_ROOT / "logs",
        ]
        total_cleaned = 0
        for log_dir in log_dirs:
            if not log_dir.exists():
                continue
            for log_file in log_dir.rglob("*.log"):
                size = log_file.stat().st_size
                if size > 10 * 1024 * 1024:  # > 10MB
                    log_file.write_text("")  # Truncate, don't delete
                    total_cleaned += 1
                    print(f"  Truncated: {log_file.name} ({size // 1024 // 1024}MB)")
        print_ok(f"Truncated {total_cleaned} oversized log files")

    elif action == "tmp":
        print_header("Cleaning Temporary Files")
        patterns = ["__pycache__", "*.pyc", ".mypy_cache", ".pytest_cache", ".ruff_cache"]
        total = 0
        for pattern in patterns:
            for match in GAIA_ROOT.rglob(pattern):
                if ".git" in match.parts:
                    continue
                if match.is_dir():
                    shutil.rmtree(match, ignore_errors=True)
                else:
                    match.unlink(missing_ok=True)
                total += 1
        print_ok(f"Cleaned {total} temp items")

    elif action == "stashes":
        print_header("Cleaning Git Stashes")
        result = run(["git", "stash", "list"], capture=True, check=False)
        count = len(result.stdout.strip().splitlines()) if result.stdout.strip() else 0
        if count > 0:
            run(["git", "stash", "clear"], check=False)
            print_ok(f"Cleared {count} stashes")
        else:
            print_ok("No stashes to clear")


# ═══════════════════════════════════════════════════════════════════════
# REFACTOR — Global Codebase Refactoring
# ═══════════════════════════════════════════════════════════════════════

def cmd_refactor(args):
    action = args.action

    if action == "rename-limb":
        _refactor_rename_limb()
    elif action == "sync-common":
        _sync_common()


def _refactor_rename_limb():
    """Surgical rename: execute_tool/execute_skill → execute_limb.

    Wire-safe: preserves <tool_call> XML tags, ToolCall class, tool_calls fields.
    Only renames GAIA's own function/method names in active source code.
    """
    print_header("Terminology Refactor: Tools/Skills → Limbs")

    # Target files (production + candidates, active code only)
    targets = [
        # (old_term, new_term, file_list)
        ("execute_tool", "execute_limb", [
            "gaia-mcp/gaia_mcp/tools.py",
            "gaia-mcp/gaia_mcp/server.py",
            "gaia-mcp/gaia_mcp/__init__.py",
            "gaia-core/gaia_core/cognition/agent_core.py",
            "candidates/gaia-mcp/gaia_mcp/tools.py",
            "candidates/gaia-mcp/gaia_mcp/server.py",
            "candidates/gaia-mcp/gaia_mcp/__init__.py",
            "candidates/gaia-core/gaia_core/cognition/agent_core.py",
        ]),
        ("execute_skill", "execute_limb", [
            "gaia-mcp/gaia_mcp/skill_manager.py",
            "gaia-mcp/gaia_mcp/skills/system_pulse.py",
            "candidates/gaia-mcp/gaia_mcp/skills/system_pulse.py",
        ]),
        ("dispatch_sidecar_actions", "dispatch_sidecar_limbs", [
            "gaia-core/gaia_core/utils/mcp_client.py",
            "gaia-core/gaia_core/utils/output_router.py",
            "gaia-core/e2e_sidecar_write_test.py",
            "candidates/gaia-core/gaia_core/utils/mcp_client.py",
            "candidates/gaia-core/gaia_core/utils/output_router.py",
            "candidates/gaia-core/e2e_sidecar_write_test.py",
        ]),
    ]

    total_changes = 0
    for old, new, files in targets:
        for filepath in files:
            full_path = GAIA_ROOT / filepath
            if not full_path.exists():
                continue
            content = full_path.read_text()
            if old not in content:
                continue
            count = content.count(old)
            full_path.write_text(content.replace(old, new))
            print(f"  {old} → {new}: {filepath} ({count}x)")
            total_changes += count

    if total_changes == 0:
        print("  No remaining occurrences found (already renamed?)")
    else:
        print(f"\n  Total: {total_changes} replacements across {len(targets)} terms")
    print_ok("Rename complete (wire protocol preserved)")


def _sync_common():
    """Promote gaia-common from candidates to production."""
    print_header("Syncing gaia-common: candidates → production")
    cmd_promote(argparse.Namespace(
        service="gaia-common", validate=False, force=False, no_restart=True
    ))


# ═══════════════════════════════════════════════════════════════════════
# Main — Argument Parser
# ═══════════════════════════════════════════════════════════════════════

def build_parser():
    parser = argparse.ArgumentParser(
        description="GaiaCLI — Unified System Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  gaia_cli.py live status           Show stack health + sovereign vitals
  gaia_cli.py live start            Start all containers
  gaia_cli.py promote gaia-core     Promote candidate to production
  gaia_cli.py gpu status            Show GPU allocation
  gaia_cli.py clean prune           Docker system prune
  gaia_cli.py refactor rename-limb  Rename tool→limb terminology
""",
    )
    subs = parser.add_subparsers(dest="command", help="Command group")

    # LIVE
    live_p = subs.add_parser("live", help="Manage live stack")
    live_p.add_argument("action", choices=["start", "stop", "status", "logs"])
    live_p.add_argument("service", nargs="?", help="Service for logs (optional)")
    live_p.add_argument("--tail", type=int, default=50, help="Log tail lines")

    # TEST
    test_p = subs.add_parser("test", help="Test a candidate service")
    test_p.add_argument("service", help="Service name")
    test_p.add_argument("--inject", action="store_true", help="Containerized validation")
    test_p.add_argument("--unit", action="store_true", help="Run unit tests")
    test_p.add_argument("--gpu", action="store_true", help="Reserve GPU for test")

    # PROMOTE
    prom_p = subs.add_parser("promote", help="Promote candidate to production")
    prom_p.add_argument("service", help="Service name")
    prom_p.add_argument("--validate", action="store_true", help="Run validation first")
    prom_p.add_argument("--force", action="store_true", help="Override sync checks")
    prom_p.add_argument("--no-restart", action="store_true", help="Skip container restart")

    # GPU
    gpu_p = subs.add_parser("gpu", help="GPU lifecycle management")
    gpu_p.add_argument("action", choices=["status", "release", "reclaim", "handoff"])
    gpu_p.add_argument("--target", default="study", help="Handoff target (default: study)")

    # CLEAN
    clean_p = subs.add_parser("clean", help="Cleanup operations")
    clean_p.add_argument("action", choices=["prune", "logs", "tmp", "stashes"])

    # REFACTOR
    ref_p = subs.add_parser("refactor", help="Global codebase refactoring")
    ref_p.add_argument("action", choices=["rename-limb", "sync-common"])

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "live": cmd_live,
        "test": cmd_test,
        "promote": cmd_promote,
        "gpu": cmd_gpu,
        "clean": cmd_clean,
        "refactor": cmd_refactor,
    }

    handler = dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

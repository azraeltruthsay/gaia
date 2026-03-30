"""Chaos Engine — container-level and code-level fault injection drills."""
import ast
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen, Request

from gaia_monkey import cognitive_validator, fault_injector, meditation_controller, serenity_manager

log = logging.getLogger("gaia-monkey.chaos")

GAIA_PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/gaia/GAIA_Project"))
CORE_ENDPOINT = os.environ.get("CORE_ENDPOINT", "http://gaia-core:6415")
DOCTOR_ENDPOINT = os.environ.get("DOCTOR_ENDPOINT", "http://gaia-doctor:6419")

SERVICES = {
    "gaia-core": ("http://gaia-core:6415/health", "restart"),
    "gaia-web": ("http://gaia-web:6414/health", "restart"),
    "gaia-mcp": ("http://gaia-mcp:8765/health", "restart"),
    "gaia-prime": ("http://gaia-prime:7777/health", None),
    "gaia-audio": ("http://gaia-audio:8080/health", None),
    "gaia-core-candidate": ("http://gaia-core-candidate:6415/health", "ha"),
    "gaia-mcp-candidate": ("http://gaia-mcp-candidate:8765/health", "ha"),
}


def check_health(name: str, url: str) -> bool:
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def run_container_drill(targets: list[str] | None = None) -> dict:
    """Container-level fault injection drill. Requires active Defensive Meditation."""
    if not meditation_controller.is_active():
        return {"error": "Chaos drill requires active Defensive Meditation"}

    drill_targets = targets or ["gaia-core-candidate", "gaia-mcp-candidate"]
    results = []

    for name in drill_targets:
        if name not in SERVICES:
            results.append({"service": name, "status": "skipped", "reason": "not in registry"})
            continue

        url = SERVICES[name][0]
        entry = {"service": name}

        if not check_health(name, url):
            entry["status"] = "skipped"
            entry["reason"] = "already unhealthy at drill start"
            results.append(entry)
            continue

        log.info("🐒 Chaos drill: stopping %s...", name)
        entry["baseline"] = "healthy"

        try:
            stop_result = subprocess.run(["docker", "stop", name], capture_output=True, text=True, timeout=30)
            if stop_result.returncode != 0:
                entry["status"] = "failed"
                entry["reason"] = f"docker stop failed: {stop_result.stderr.strip()[:200]}"
                results.append(entry)
                continue
        except subprocess.TimeoutExpired:
            entry["status"] = "failed"
            entry["reason"] = "docker stop timed out"
            results.append(entry)
            continue

        time.sleep(2)
        if check_health(name, url):
            entry["status"] = "anomaly"
            entry["reason"] = "still healthy after stop"
            results.append(entry)
            continue

        log.info("🐒 Chaos drill: %s confirmed DOWN — restarting...", name)
        entry["fault_injected"] = True

        try:
            start_result = subprocess.run(["docker", "start", name], capture_output=True, text=True, timeout=30)
            if start_result.returncode != 0:
                entry["status"] = "failed"
                entry["reason"] = f"docker start failed: {start_result.stderr.strip()[:200]}"
                results.append(entry)
                continue
        except subprocess.TimeoutExpired:
            entry["status"] = "failed"
            entry["reason"] = "docker start timed out"
            results.append(entry)
            continue

        recovered = False
        for _ in range(6):
            time.sleep(5)
            if check_health(name, url):
                recovered = True
                break

        if recovered:
            serenity_manager.record_recovery("service_recovery", f"chaos-drill: {name}", meditation_controller.is_active())
            log.info("🐒 Chaos drill: %s RECOVERED ✓", name)
            entry["status"] = "recovered"

            if "core" in name:
                core_endpoint = SERVICES[name][0].rsplit("/health", 1)[0]
                log.info("🐒 Running cognitive validation against %s...", name)
                cog_result = cognitive_validator.validate(core_endpoint)
                entry["cognitive_validation"] = cog_result
                if cog_result["passed"]:
                    serenity_manager.record_recovery("cognitive_validation", f"chaos-drill inference: {name}", meditation_controller.is_active())
                    log.info("🐒 Cognitive validation PASSED for %s (%.0fms)", name, cog_result["latency_ms"])
                else:
                    log.warning("🐒 Cognitive validation FAILED for %s: %s", name, cog_result.get("error", "no meaningful response"))
        else:
            log.error("🐒 Chaos drill: %s FAILED TO RECOVER after 30s", name)
            entry["status"] = "failed_recovery"

        results.append(entry)

    return {
        "drill_results": results,
        "serenity": serenity_manager.get_report(),
        "meditation_active": meditation_controller.is_active(),
    }


def _notify_doctor(service: str, file_path: str, fault: str, difficulty: int):
    """Notify gaia-doctor of a chaos injection so it can drive repair organically."""
    try:
        data = json.dumps({
            "service": service,
            "file": file_path,
            "fault": fault,
            "difficulty": difficulty,
        }).encode("utf-8")
        req = Request(
            f"{DOCTOR_ENDPOINT}/notify/chaos_injection",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read().decode())
            log.info("🐒 Doctor notified: %s", result.get("status", "unknown"))
    except Exception as e:
        log.warning("🐒 Failed to notify doctor: %s", e)


def run_code_drill(targets: list[str] | None = None, difficulty: int | None = None) -> dict:
    """Semantic code fault injection drill. Injects fault then hands off to Doctor.

    Monkey injects -> Doctor detects -> Doctor runs tests -> Doctor repairs/restores.
    Requires active Defensive Meditation.
    """
    if not meditation_controller.is_active():
        return {"error": "Code chaos drill requires active Defensive Meditation"}

    # Auto-scale difficulty from serenity if not explicitly provided
    if difficulty is None:
        serenity_report = serenity_manager.get_report()
        difficulty = fault_injector.get_difficulty_for_serenity(serenity_report.get("score", 0.0))

    drill_targets = targets or ["gaia-core-candidate"]
    results = []

    for name in drill_targets:
        if name not in SERVICES or name not in fault_injector.SERVICE_CODE_DIRS:
            results.append({"service": name, "status": "skipped", "reason": "not in registry"})
            continue

        url = SERVICES[name][0]
        entry = {"service": name, "type": "semantic_fault_injection", "difficulty": difficulty}

        if not check_health(name, url):
            entry["status"] = "skipped"
            entry["reason"] = "already unhealthy at drill start"
            results.append(entry)
            continue

        target_file = fault_injector.pick_target_file(name)
        if not target_file:
            entry["status"] = "skipped"
            entry["reason"] = "no suitable target file found"
            results.append(entry)
            continue

        rel_path = str(target_file.relative_to(GAIA_PROJECT_ROOT))
        entry["target_file"] = rel_path
        log.info("🐒 Code chaos: targeting %s in %s (difficulty=%d)", target_file.name, name, difficulty)

        try:
            original_content = target_file.read_text()
        except Exception as e:
            entry["status"] = "failed"
            entry["reason"] = f"could not read target file: {e}"
            results.append(entry)
            continue

        broken_content, fault_desc = fault_injector.inject_semantic_fault(original_content, difficulty=difficulty)
        entry["fault_description"] = fault_desc
        log.info("🐒 Code chaos: injecting semantic fault: %s", fault_desc)

        # Validate AST — fall back to NameError injection if broken syntax
        try:
            ast.parse(broken_content)
        except SyntaxError:
            broken_content = original_content.replace(
                original_content.split("\n")[0],
                f"# CHAOS_MONKEY_INJECT\n_chaos_undef = _nonexistent_var_12345\n{original_content.split(chr(10))[0]}"
            )
            fault_desc = "injected NameError (fallback)"
            entry["fault_description"] = fault_desc

        # Write broken content into the container via docker exec
        # (project root is mounted :ro, so we write inside the container)
        relative_to_service = target_file.relative_to(fault_injector.SERVICE_CODE_DIRS[name])
        container_path = f"/app/{relative_to_service}"

        try:
            inject_cmd = subprocess.run(
                ["docker", "exec", name, "python3", "-c",
                 f"open({container_path!r}, 'w').write({broken_content!r})"],
                capture_output=True, text=True, timeout=10,
            )
            if inject_cmd.returncode != 0:
                entry["status"] = "failed"
                entry["reason"] = f"injection failed: {inject_cmd.stderr.strip()[:200]}"
                results.append(entry)
                continue
        except subprocess.TimeoutExpired:
            entry["status"] = "failed"
            entry["reason"] = "injection timed out"
            results.append(entry)
            continue

        log.info("🐒 Code chaos: semantic fault injected into %s", target_file.name)
        entry["fault_injected"] = True
        entry["container_path"] = container_path

        # Restart container to pick up the broken code
        try:
            subprocess.run(["docker", "restart", name], capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            pass

        # Notify Doctor — it will detect the fault, run tests, and drive repair
        _notify_doctor(name, rel_path, fault_desc, difficulty)
        entry["status"] = "injected"
        entry["awaiting_doctor"] = True
        log.info("🐒 Code chaos: fault injected, Doctor notified. Awaiting organic repair.")

        results.append(entry)

    return {
        "drill_type": "semantic_fault_injection",
        "difficulty": difficulty,
        "drill_results": results,
        "serenity": serenity_manager.get_report(),
        "meditation_active": meditation_controller.is_active(),
    }

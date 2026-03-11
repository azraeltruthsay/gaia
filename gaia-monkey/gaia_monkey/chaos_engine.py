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


def run_code_drill(targets: list[str] | None = None) -> dict:
    """Semantic code fault injection drill. Requires active Defensive Meditation."""
    if not meditation_controller.is_active():
        return {"error": "Code chaos drill requires active Defensive Meditation"}

    drill_targets = targets or ["gaia-core-candidate"]
    results = []

    for name in drill_targets:
        if name not in SERVICES or name not in fault_injector.SERVICE_CODE_DIRS:
            results.append({"service": name, "status": "skipped", "reason": "not in registry"})
            continue

        url = SERVICES[name][0]
        entry = {"service": name, "type": "semantic_fault_injection"}

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

        entry["target_file"] = str(target_file.relative_to(GAIA_PROJECT_ROOT))
        log.info("🐒 Code chaos: targeting %s in %s", target_file.name, name)

        try:
            original_content = target_file.read_text()
        except Exception as e:
            entry["status"] = "failed"
            entry["reason"] = f"could not read target file: {e}"
            results.append(entry)
            continue

        broken_content, fault_desc = fault_injector.inject_semantic_fault(original_content)
        entry["fault_description"] = fault_desc
        log.info("🐒 Code chaos: injecting semantic fault: %s", fault_desc)

        try:
            ast.parse(broken_content)
        except SyntaxError:
            broken_content = original_content.replace(
                original_content.split("\n")[0],
                f"# CHAOS_MONKEY_INJECT\n_chaos_undef = _nonexistent_var_12345\n{original_content.split(chr(10))[0]}"
            )
            fault_desc = "injected NameError (fallback)"
            entry["fault_description"] = fault_desc

        container_name = name
        relative_to_service = target_file.relative_to(fault_injector.SERVICE_CODE_DIRS[name])
        container_path = f"/app/{relative_to_service}"

        try:
            inject_cmd = subprocess.run(
                ["docker", "exec", container_name, "python3", "-c",
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

        try:
            subprocess.run(["docker", "restart", container_name], capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            pass

        time.sleep(8)
        still_healthy = check_health(name, url)
        entry["post_injection_healthy"] = still_healthy

        if still_healthy:
            log.info("🐒 Service still healthy after injection (non-critical file) — running LLM repair")

        log.info("🐒 Escalating to Tier 2 (LLM-powered repair) for %s...", target_file.name)

        try:
            broken_on_disk = target_file.read_text()
            error_msg = f"CHAOS_MONKEY semantic fault injected: {fault_desc}"

            repair_url = f"{CORE_ENDPOINT}/api/repair/structural"
            repair_data = json.dumps({
                "service": name,
                "broken_code": broken_on_disk,
                "error_msg": error_msg,
                "file_path": str(target_file),
            }).encode("utf-8")

            req = Request(repair_url, data=repair_data,
                          headers={"Content-Type": "application/json"}, method="POST")

            repair_result = None
            with urlopen(req, timeout=120) as response:
                if response.status == 200:
                    repair_result = json.loads(response.read().decode("utf-8"))

            if repair_result and repair_result.get("status") == "repaired":
                entry["repair_method"] = "tier2_llm"
                entry["llm_repaired"] = True
                log.info("🐒 Tier 2 LLM repair succeeded for %s", target_file.name)
            else:
                log.warning("🐒 Tier 2 LLM repair failed — restoring original %s", target_file.name)
                subprocess.run(
                    ["docker", "exec", container_name, "python3", "-c",
                     f"open({container_path!r}, 'w').write({original_content!r})"],
                    capture_output=True, text=True, timeout=10,
                )
                entry["repair_method"] = "manual_restore"
                entry["llm_repaired"] = False

        except Exception as e:
            log.error("🐒 Tier 2 repair exception: %s — restoring original", e)
            subprocess.run(
                ["docker", "exec", container_name, "python3", "-c",
                 f"open({container_path!r}, 'w').write({original_content!r})"],
                capture_output=True, text=True, timeout=10,
            )
            entry["repair_method"] = "emergency_restore"
            entry["llm_repaired"] = False
            entry["repair_error"] = str(e)[:200]

        try:
            repaired_content = target_file.read_text()
            ast.parse(repaired_content)
            entry["syntax_clean"] = True
            has_marker = "CHAOS_MONKEY" in repaired_content
            entry["marker_removed"] = not has_marker
            if has_marker:
                log.warning("🐒 Chaos marker still in repaired file — restoring original")
                subprocess.run(
                    ["docker", "exec", container_name, "python3", "-c",
                     f"open({container_path!r}, 'w').write({original_content!r})"],
                    capture_output=True, text=True, timeout=10,
                )
        except SyntaxError:
            entry["syntax_clean"] = False
            subprocess.run(
                ["docker", "exec", container_name, "python3", "-c",
                 f"open({container_path!r}, 'w').write({original_content!r})"],
                capture_output=True, text=True, timeout=10,
            )

        try:
            subprocess.run(["docker", "restart", container_name], capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            pass

        recovered = False
        for _ in range(6):
            time.sleep(5)
            if check_health(name, url):
                recovered = True
                break

        if recovered:
            if entry.get("llm_repaired"):
                serenity_manager.record_recovery("vital_recovery", f"LLM-repaired code chaos: {target_file.name}", meditation_controller.is_active())
                serenity_manager.record_recovery("cognitive_validation", f"LLM repair verified: {name}", meditation_controller.is_active())
                log.info("🐒 Code chaos: LLM-REPAIRED recovery — full Serenity points awarded")
            else:
                serenity_manager.record_recovery("standard_recovery", f"code-chaos: {target_file.name}", meditation_controller.is_active())

            serenity_manager.record_recovery("service_recovery", f"code-chaos restart: {name}", meditation_controller.is_active())

            if "core" in name:
                core_endpoint = SERVICES[name][0].rsplit("/health", 1)[0]
                log.info("🐒 Running cognitive validation against %s...", name)
                cog_result = cognitive_validator.validate(core_endpoint)
                entry["cognitive_validation"] = cog_result
                if cog_result["passed"]:
                    serenity_manager.record_recovery("cognitive_validation", f"post-chaos inference: {name}", meditation_controller.is_active())
                    log.info("🐒 Cognitive validation PASSED for %s (%.0fms)", name, cog_result["latency_ms"])
                else:
                    log.warning("🐒 Cognitive validation FAILED for %s", name)

            log.info("🐒 Code chaos drill: %s RECOVERED ✓ (repair: %s)", name, entry.get("repair_method", "unknown"))
            entry["status"] = "recovered"
        else:
            log.error("🐒 Code chaos drill: %s FAILED TO RECOVER after 30s", name)
            entry["status"] = "failed_recovery"

        results.append(entry)

    return {
        "drill_type": "semantic_fault_injection",
        "drill_results": results,
        "serenity": serenity_manager.get_report(),
        "meditation_active": meditation_controller.is_active(),
    }

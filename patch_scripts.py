import json
import subprocess
import time
from pathlib import Path

def patch_world_state():
    path = "gaia-common/gaia_common/utils/world_state.py"
    with open(path, "r") as f:
        content = f.read()

    import_hook = "import json\nimport subprocess"
    if "import subprocess" not in content:
        content = content.replace("import json", import_hook)

    temps_func = """
def _update_and_get_temperature_stats() -> str:
    try:
        temp_file = Path("/tmp/gaia_temp_history.json")
        history = []
        if temp_file.exists():
            try:
                history = json.loads(temp_file.read_text())
            except Exception:
                pass
                
        cpu_temp = None
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                cpu_temp = float(f.read().strip()) / 1000.0
        except Exception:
            pass
            
        gpu_temp = None
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], text=True)
            gpu_temp = float(out.split()[0])
        except Exception:
            pass
            
        now = time.time()
        if cpu_temp is not None or gpu_temp is not None:
            history.append({"ts": now, "cpu": cpu_temp, "gpu": gpu_temp})
        
        history = [x for x in history if now - x["ts"] <= 600]
        
        try:
            temp_file.write_text(json.dumps(history))
        except Exception:
            pass
            
        cpu_temps = [x["cpu"] for x in history if x.get("cpu") is not None]
        gpu_temps = [x["gpu"] for x in history if x.get("gpu") is not None]
        
        parts = []
        if cpu_temps:
            parts.append(f"CPU: {int(sum(cpu_temps)/len(cpu_temps))}C avg ({int(min(cpu_temps))}-{int(max(cpu_temps))}C)")
        if gpu_temps:
            parts.append(f"GPU: {int(sum(gpu_temps)/len(gpu_temps))}C avg ({int(min(gpu_temps))}-{int(max(gpu_temps))}C)")
            
        if parts:
            return "10m Temps: " + " | ".join(parts)
        return ""
    except Exception as e:
        return f"Temps: error ({str(e)})"

"""

    if "_update_and_get_temperature_stats" not in content:
        content = content.replace("def world_state_snapshot(", temps_func + "def world_state_snapshot(")

    if '"temps": _update_and_get_temperature_stats(),' not in content:
        content = content.replace('"mem": _mem_summary(),', '"mem": _mem_summary(),\n        "temps": _update_and_get_temperature_stats(),')

    if "snap.get('temps', '')" not in content:
        old_line = "lines.append(f\"{uptime_str} | {snap['load']} | {snap['mem']}\")"
        new_line = "temps_str = snap.get('temps', '')\n    lines.append(f\"{uptime_str} | {snap['load']} | {snap['mem']}\" + (f\" | {temps_str}\" if temps_str else \"\"))"
        content = content.replace(old_line, new_line)

    with open(path, "w") as f:
        f.write(content)

def patch_immune_system():
    path = "gaia-common/gaia_common/utils/immune_system.py"
    with open(path, "r") as f:
        content = f.read()

    old_search_dirs = """        # Targeted scan of GAIA services only (avoid 33k+ files in SDKs/archives)
        search_dirs = []
        target_services = ["gaia-core", "gaia-web", "gaia-mcp", "gaia-study", "gaia-audio", "gaia-common", "gaia-orchestrator", "gaia-doctor"]
        for svc in target_services:
            svc_path = project_root / svc
            if svc_path.exists():
                search_dirs.append(svc_path)

        cache_updated = False
        
        for base_dir in search_dirs:
            for py_file in base_dir.rglob("*.py"):
                # Strict exclusion of non-source and giant library folders
                parts = [p.lower() for p in py_file.parts]
                if any(p in parts for p in [".venv", "venv", "__pycache__", ".git", "candidates", "archive", "google-cloud-sdk", "artifacts"]):
                    continue"""

    new_search_dirs = """        # Targeted scan of GAIA services only (avoid 33k+ files in SDKs/archives)
        search_dirs = []
        target_services = ["gaia-core", "gaia-web", "gaia-mcp", "gaia-study", "gaia-audio", "gaia-common", "gaia-orchestrator", "gaia-doctor"]
        for svc in target_services:
            svc_path = project_root / svc
            if svc_path.exists():
                search_dirs.append((svc_path, "[PROD]"))
            cand_path = project_root / "candidates" / svc
            if cand_path.exists():
                search_dirs.append((cand_path, "[CAND]"))

        cache_updated = False
        
        for base_dir, env_tag in search_dirs:
            for py_file in base_dir.rglob("*.py"):
                # Strict exclusion of non-source and giant library folders
                parts = [p.lower() for p in py_file.parts]
                if any(p in parts for p in [".venv", "venv", "__pycache__", ".git", "archive", "google-cloud-sdk", "artifacts"]):
                    continue"""
                    
    content = content.replace(old_search_dirs, new_search_dirs)

    old_syntax_issue = 'file_issues.append(f"SyntaxError in {py_file} (Line ?): {str(e)[:100]}")'
    new_syntax_issue = 'file_issues.append(f"{env_tag} SyntaxError in {py_file} (Line ?): {str(e)[:100]}")'
    content = content.replace(old_syntax_issue, new_syntax_issue)

    old_ruff_issue = 'file_issues.append(f"LintError[{code}] in {py_file} (Line {line}): {msg}{snippet}")'
    new_ruff_issue = 'file_issues.append(f"{env_tag} LintError[{code}] in {py_file} (Line {line}): {msg}{snippet}")'
    content = content.replace(old_ruff_issue, new_ruff_issue)

    with open(path, "w") as f:
        f.write(content)

patch_world_state()
patch_immune_system()

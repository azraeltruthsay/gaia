import json
from pathlib import Path

path = "gaia-common/gaia_common/utils/world_state.py"
with open(path, "r") as f:
    content = f.read()

new_func = """def _update_and_get_temperature_stats() -> str:
    try:
        temp_file = Path("/gaia/GAIA_Project/logs/temp_history.json")
        if not temp_file.exists():
            return ""
        history = json.loads(temp_file.read_text())
        now = time.time()
        
        cpu_temps = [x["cpu"] for x in history if x.get("cpu") is not None and now - x["ts"] <= 600]
        gpu_temps = [x["gpu"] for x in history if x.get("gpu") is not None and now - x["ts"] <= 600]
        
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

import re
content = re.sub(r'def _update_and_get_temperature_stats\(\) -> str:.*?return f"Temps: error \(\{str\(e\)\}\)"', new_func, content, flags=re.DOTALL)

with open(path, "w") as f:
    f.write(content)

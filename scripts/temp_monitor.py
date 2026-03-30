import time
import json
import subprocess
from pathlib import Path

LOG_FILE = Path("/gaia/GAIA_Project/logs/temp_history.json")

def read_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except:
        return None

def read_gpu_temp():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], text=True)
        return float(out.split()[0])
    except:
        return None

def main():
    while True:
        try:
            history = []
            if LOG_FILE.exists():
                try:
                    history = json.loads(LOG_FILE.read_text())
                except:
                    pass
            
            now = time.time()
            history = [x for x in history if now - x["ts"] <= 600] # Keep 10 mins
            
            c = read_cpu_temp()
            g = read_gpu_temp()
            
            if c is not None or g is not None:
                history.append({"ts": now, "cpu": c, "gpu": g})
                
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            LOG_FILE.write_text(json.dumps(history))
        except Exception as e:
            pass
        time.sleep(30)

if __name__ == "__main__":
    main()
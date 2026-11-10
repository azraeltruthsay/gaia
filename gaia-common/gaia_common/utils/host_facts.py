"""What this machine really is, read from the machine (Azrael, 2026-11-03).

Core made up an i7 and an RTX 4060 Ti when asked what she runs on, and the line
that should have told her was hardcoded. Everything here is read when asked: CPU
and memory from /proc, the GPU from nvidia-smi, the OS from uname, the models from
the engines themselves. What can't be read is left out, never guessed.
"""
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import time
import urllib.request
from typing import Dict, Optional

logger = logging.getLogger("GAIA.HostFacts")

_CACHE_SECONDS = 30.0
_cache: Dict[str, tuple] = {}
# Engines by network name, so any container can read them (2026-11-10: from
# gaia-mcp, where introspect runs, "localhost:8092" was the wrong container and
# Core always read as unreachable). gaia-core resolves its own name too.
ENGINES = {
    "Core": os.environ.get("GAIA_CORE_ENGINE_URL", "http://gaia-core:8092"),
    "Prime": os.environ.get("GAIA_PRIME_ENGINE_URL", "http://gaia-prime:7777"),
}


def _cached(key: str, read):
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < _CACHE_SECONDS:
        return hit[1]
    try:
        value = read()
    except Exception as e:
        logger.debug("Host facts: could not read %s: %s", key, e)
        value = None
    _cache[key] = (now, value)
    return value


def _read_cpu(path: str = "/proc/cpuinfo") -> Optional[dict]:
    model, threads = None, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("processor"):
                threads += 1
            elif model is None and line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
    return {"model": model, "threads": threads} if model else None


def _read_memory(path: str = "/proc/meminfo") -> Optional[dict]:
    values = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                values[key] = int(rest.split()[0]) / (1024 ** 2)
    if "MemTotal" not in values:
        return None
    return {"total_gb": round(values["MemTotal"], 1), "available_gb": round(values.get("MemAvailable", 0), 1)}


def _read_gpu() -> Optional[dict]:
    if not shutil.which("nvidia-smi"):
        return None
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=3, check=True).stdout.strip().splitlines()
    name, total, used = [x.strip() for x in out[0].split(",")]
    return {"name": name, "total_gb": round(int(total) / 1024, 1), "used_gb": round(int(used) / 1024, 1)}


def _read_disk(path: str = "/knowledge") -> Optional[dict]:
    usage = shutil.disk_usage(path if os.path.exists(path) else "/")
    return {"total_gb": round(usage.total / 1e9), "free_gb": round(usage.free / 1e9)}


def _read_engine(endpoint: str) -> Optional[dict]:
    with urllib.request.urlopen(f"{endpoint}/status", timeout=1.5) as r:
        status = json.load(r)
    model = status.get("model_path") or status.get("model")
    if not model:
        return None
    return {"model": os.path.basename(os.path.realpath(model)), "device": status.get("device")}


def host_facts() -> dict:
    """Whatever could be read just now."""
    facts = {
        "cpu": _cached("cpu", _read_cpu),
        "memory": _cached("memory", _read_memory),
        "gpu": _cached("gpu", _read_gpu),
        "disk": _cached("disk", _read_disk),
        "os": f"{platform.system()} {platform.release()}",
    }
    engines = {name: _cached(f"engine:{name}", lambda e=ep: _read_engine(e)) for name, ep in ENGINES.items()}
    facts["engines"] = {k: v for k, v in engines.items() if v}
    return {k: v for k, v in facts.items() if v}


_ASKS_RE = re.compile(
    r"\b(?:your|you|this|the)\s+(?:\w+\s+)?(?:hardware|gpu|vram|graphics card|cpu|processor|specs?|machine|computer|workstation|"
    r"system resources?|resources|environment|ram|disk)\b"
    r"|\bwhat\b.*\byou\b.*\b(?:running|run)\s+on\b|\bwhat\s+are\s+you\s+running\b"
    r"|\bhow\s+much\s+(?:gpu|vram|ram|disk)\b"
    r"|\bwhat\s+(?:hardware|gpu|cpu|processor|graphics card)\b.*\byou\b",
    re.IGNORECASE,
)


def asks_about_host(text: str) -> bool:
    """Whether a message asks what she runs on or what she has to work with."""
    return bool(_ASKS_RE.search(text or ""))


def machine_description(facts: Optional[dict] = None) -> str:
    """The body she runs on, in a phrase. "" if nothing could be read."""
    facts = host_facts() if facts is None else facts
    parts = []
    if facts.get("gpu"):
        parts.append(f"{facts['gpu']['name']} with {facts['gpu']['total_gb']} GB")
    if facts.get("cpu"):
        parts.append(f"{facts['cpu']['model']}, {facts['cpu']['threads']} threads")
    if facts.get("memory"):
        parts.append(f"{facts['memory']['total_gb']} GB RAM")
    return "; ".join(parts)


def host_facts_text(facts: Optional[dict] = None) -> str:
    """Plain facts, read just now, for her to answer from. "" if nothing."""
    facts = host_facts() if facts is None else facts
    lines = []
    body = machine_description(facts)
    if body:
        lines.append(f"This machine: {body}.")
    live = []
    if facts.get("gpu"):
        live.append(f"{facts['gpu']['used_gb']} GB of GPU memory in use")
    if facts.get("memory"):
        live.append(f"{facts['memory']['available_gb']} GB of RAM available")
    if facts.get("disk"):
        live.append(f"{facts['disk']['free_gb']} GB of disk free of {facts['disk']['total_gb']} GB")
    if live:
        lines.append("Right now: " + ", ".join(live) + ".")
    if facts.get("os"):
        lines.append(f"Operating system: {facts['os']}.")
    for name, e in (facts.get("engines") or {}).items():
        lines.append(f"{name} is running {e['model']} on the {str(e.get('device', 'unknown')).upper()}.")
    return " ".join(lines)

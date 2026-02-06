import os
import time
import psutil
import logging
from pathlib import Path
from gaia_core.config import Config, get_config
from gaia_core.memory.status_tracker import GAIAStatus

logger = logging.getLogger("GAIA.TelemetricSenses")

config = Config()

# File telemetry config
WATCHED_EXTENSIONS = [".py", ".md", ".json"]
WATCHED_DIRS = ["app", "knowledge/system_reference"]
MODIFIED_TIMES = {}

# Initialize loop counter and last activity timestamp
if GAIAStatus.get("loop_tick_count") is None:
    GAIAStatus.update("loop_tick_count", 0)
if GAIAStatus.get("last_loop_time") is None:
    GAIAStatus.update("last_loop_time", time.time())


def tick():
    GAIAStatus.update("loop_tick_count", GAIAStatus.get("loop_tick_count", 0) + 1)
    GAIAStatus.update("last_loop_time", time.time())
    logger.debug("🔁 Loop ticked.")


def update_token_usage(count: int):
    GAIAStatus.update("last_token_count", count)
    GAIAStatus.update("last_token_time", time.time())
    logger.debug(f"🔢 Tokens used: {count}")


def scan_files():
    changes = []
    for root_dir in WATCHED_DIRS:
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if any(fname.endswith(ext) for ext in WATCHED_EXTENSIONS):
                    fpath = Path(dirpath) / fname
                    mtime = int(fpath.stat().st_mtime)
                    if fpath not in MODIFIED_TIMES:
                        MODIFIED_TIMES[fpath] = mtime
                    elif MODIFIED_TIMES[fpath] != mtime:
                        changes.append(str(fpath))
                        MODIFIED_TIMES[fpath] = mtime
    if changes:
        GAIAStatus.update("file_changes", changes)
        GAIAStatus.update("last_file_scan", time.time())
        logger.info(f"📂 Detected file changes: {changes}")


def get_gpu_usage() -> dict[str, any]:
    """
    Gathers GPU usage statistics using pynvml.

    Returns:
        A dictionary containing GPU usage metrics if available, otherwise an empty dict.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        gpu_data = {
            "name": pynvml.nvmlDeviceGetName(handle),
            "total_memory_mb": mem_info.total / 1024**2,
            "used_memory_mb": mem_info.used / 1024**2,
            "free_memory_mb": mem_info.free / 1024**2,
            "utilization_percent": utilization.gpu,
        }
        pynvml.nvmlShutdown()
        return gpu_data
    except Exception as e:
        logger.debug(f"Could not get GPU usage: {e}")
        return {}

def get_hardware_profile() -> dict[str, any]:
    """
    Gathers static information about the system's hardware.
    """
    try:
        import cpuinfo
        cpu_info = cpuinfo.get_cpu_info()
        
        profile = {
            "cpu_info": {
                "brand_raw": cpu_info.get("brand_raw", "Unknown"),
                "arch": cpu_info.get("arch", "Unknown"),
                "hz_advertised_friendly": cpu_info.get("hz_advertised_friendly", "Unknown"),
                "count": os.cpu_count()
            },
            "total_ram_gb": psutil.virtual_memory().total / (1024**3),
            "gpu_info": get_gpu_usage()
        }
        return profile
    except Exception as e:
        logger.error(f"Error gathering hardware profile: {e}")
        return {}

def get_system_resources() -> dict[str, any]:
    """
    Gathers system resource usage statistics (CPU, memory, disk, and optionally GPU).

    Returns:
        A dictionary containing system resource metrics.
    """
    try:
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')
        
        resources = {
            "cpu_usage_percent": cpu_usage,
            "memory_usage_percent": memory_info.percent,
            "disk_usage_percent": disk_info.percent,
            "hardware_profile": get_hardware_profile(),
            "gpu_usage": get_gpu_usage()
        }
        return resources
    except Exception as e:
        logger.error(f"Error gathering system resource metrics: {e}")
        return {}

def system_health():
    resources = get_system_resources()
    if resources:
        GAIAStatus.update("cpu_usage", resources.get("cpu_usage_percent", 0))
        GAIAStatus.update("mem_usage", resources.get("memory_usage_percent", 0))
        GAIAStatus.update("disk_usage", resources.get("disk_usage_percent", 0))
        logger.debug(f"🩺 System Health - CPU: {resources.get('cpu_usage_percent', 0)}%, Mem: {resources.get('memory_usage_percent', 0)}%, Disk: {resources.get('disk_usage_percent', 0)}%")


def get_telemetry_summary() -> str:
    tick_count = GAIAStatus.get("loop_tick_count", 0)
    token_use = GAIAStatus.get("last_token_count", 0)
    file_changes = GAIAStatus.get("file_changes", [])
    cpu = GAIAStatus.get("cpu_usage", 0)
    mem = GAIAStatus.get("mem_usage", 0)
    disk = GAIAStatus.get("disk_usage", 0)

    summary = [
        f"[Telemetry Summary]",
        f"Ticks since boot: {tick_count}",
        f"Last token usage: {token_use} tokens",
        f"System Load — CPU: {cpu}%, Mem: {mem}%, Disk: {disk}%",
    ]

    if file_changes:
        summary.append(f"Changed files: {len(file_changes)} (e.g. {file_changes[0]})")

    return "\n".join(summary)


# Optional: Periodic combined update (can be called by loop or reflection cycle)
def full_sense_sweep():
    tick()
    scan_files()
    system_health()
    logger.info("👁️ Telemetric senses sweep complete.")
    logger.debug(get_telemetry_summary())

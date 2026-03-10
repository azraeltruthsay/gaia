"""
Memory guard utility — pre-flight RAM + swap checks for heavy operations.

Parses /proc/meminfo directly (stdlib-only, no psutil dependency).
Any service can call `require_memory()` before loading large models,
merging adapters, or running quantization to avoid OOM situations.

Usage:
    from gaia_common.utils.memory_guard import require_memory, get_memory_status

    # Check before loading a 9B bf16 model (~17GB)
    require_memory(needed_mb=18000, label="LoRA merge (9B bf16)")

    # Or inspect current state
    status = get_memory_status()
    print(f"Available: {status.combined_available_mb} MB")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Default reserve: 10 GB for running GAIA services (vLLM, gaia-core, gaia-web, etc.)
_DEFAULT_RESERVE_MB = 10240


@dataclass
class MemoryStatus:
    """Snapshot of system RAM + swap from /proc/meminfo."""

    ram_total_mb: int
    ram_available_mb: int
    swap_total_mb: int
    swap_free_mb: int
    combined_available_mb: int  # ram_available + swap_free


class MemoryGuardError(RuntimeError):
    """Raised when insufficient memory is available for an operation."""

    def __init__(self, message: str, needed_mb: int, available_mb: int, reserve_mb: int):
        super().__init__(message)
        self.needed_mb = needed_mb
        self.available_mb = available_mb
        self.reserve_mb = reserve_mb


def _parse_meminfo_value(raw: str) -> int:
    """Parse a /proc/meminfo value like '65432100 kB' → MB."""
    parts = raw.strip().split()
    value_kb = int(parts[0])
    return value_kb // 1024


def get_memory_status() -> MemoryStatus:
    """Read current RAM + swap stats from /proc/meminfo.

    Returns a MemoryStatus with all values in MB.
    """
    fields = {}
    wanted = {"MemTotal", "MemAvailable", "MemFree", "SwapTotal", "SwapFree"}

    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    if key in wanted:
                        fields[key] = _parse_meminfo_value(value)
                        if len(fields) == len(wanted):
                            break
    except OSError:
        logger.error("Cannot read /proc/meminfo — memory guard unavailable")
        return MemoryStatus(
            ram_total_mb=0,
            ram_available_mb=0,
            swap_total_mb=0,
            swap_free_mb=0,
            combined_available_mb=0,
        )

    ram_total = fields.get("MemTotal", 0)
    ram_available = fields.get("MemAvailable", fields.get("MemFree", 0))
    swap_total = fields.get("SwapTotal", 0)
    swap_free = fields.get("SwapFree", 0)

    return MemoryStatus(
        ram_total_mb=ram_total,
        ram_available_mb=ram_available,
        swap_total_mb=swap_total,
        swap_free_mb=swap_free,
        combined_available_mb=ram_available + swap_free,
    )


def require_memory(
    needed_mb: int,
    reserve_mb: int | None = None,
    label: str = "operation",
) -> None:
    """Verify sufficient memory is available, or raise MemoryGuardError.

    Args:
        needed_mb: How much memory (MB) the operation needs.
        reserve_mb: How much to hold back for running services.
                    Defaults to GAIA_RAM_RESERVE_MB env var or 10240 (10 GB).
        label: Human-readable name for the operation (used in error messages).

    Raises:
        MemoryGuardError: If (combined_available - reserve) < needed_mb.
    """
    if reserve_mb is None:
        reserve_mb = int(os.getenv("GAIA_RAM_RESERVE_MB", str(_DEFAULT_RESERVE_MB)))

    status = get_memory_status()
    usable = status.combined_available_mb - reserve_mb

    logger.info(
        "Memory guard [%s]: need %d MB | RAM avail %d MB + swap free %d MB = %d MB combined | reserve %d MB | usable %d MB",
        label,
        needed_mb,
        status.ram_available_mb,
        status.swap_free_mb,
        status.combined_available_mb,
        reserve_mb,
        usable,
    )

    if usable < needed_mb:
        msg = (
            f"Memory guard BLOCKED '{label}': need {needed_mb} MB but only {usable} MB usable "
            f"(RAM avail {status.ram_available_mb} MB + swap free {status.swap_free_mb} MB "
            f"= {status.combined_available_mb} MB combined, minus {reserve_mb} MB reserve)"
        )
        logger.error(msg)
        raise MemoryGuardError(
            message=msg,
            needed_mb=needed_mb,
            available_mb=status.combined_available_mb,
            reserve_mb=reserve_mb,
        )

    logger.info("Memory guard OK [%s]: %d MB usable >= %d MB needed", label, usable, needed_mb)

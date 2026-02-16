import threading
import time
import logging
from typing import Optional

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False

try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)


class ResourceMonitor:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    _initialized: bool = False

    def __init__(self, poll_interval: int = 5):
        if self._initialized:
            return
        self._initialized = True

        self.poll_interval = poll_interval
        self.gpu_utilization = None
        self.gpu_memory_free = None
        self.gpu_memory_total = None
        self.cpu_utilization: Optional[float] = None

        # Sustained load tracking for distracted detection
        self._distracted = False
        self._sustained_check_start: Optional[float] = None
        self._distracted_threshold = 25  # percent

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if HAS_NVML or HAS_PSUTIL:
            self.start()
        else:
            logger.warning("Neither pynvml nor psutil available. Resource monitoring is disabled.")

    def start(self):
        if not (HAS_NVML or HAS_PSUTIL):
            return
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._monitor, daemon=True)
            self._thread.start()
            logger.info("Resource monitor started (GPU=%s, CPU=%s).", HAS_NVML, HAS_PSUTIL)

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join()
            logger.info("Resource monitor stopped.")

    def _monitor(self):
        while not self._stop_event.is_set():
            try:
                # GPU polling
                if HAS_NVML:
                    device_count = pynvml.nvmlDeviceGetCount()
                    if device_count > 0:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        self.gpu_utilization = utilization.gpu
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        self.gpu_memory_free = memory_info.free
                        self.gpu_memory_total = memory_info.total
                    else:
                        self.gpu_utilization = 0
                        self.gpu_memory_free = 0
                        self.gpu_memory_total = 0

                # CPU polling
                if HAS_PSUTIL:
                    self.cpu_utilization = psutil.cpu_percent(interval=None)

                # Sustained load tracking
                peak = 0.0
                if self.gpu_utilization is not None:
                    peak = max(peak, float(self.gpu_utilization))
                if self.cpu_utilization is not None:
                    peak = max(peak, self.cpu_utilization)

                if peak > self._distracted_threshold:
                    if self._sustained_check_start is None:
                        self._sustained_check_start = time.monotonic()
                    elif time.monotonic() - self._sustained_check_start >= 5.0:
                        self._distracted = True
                else:
                    self._sustained_check_start = None
                    # Don't auto-clear _distracted here — let check_and_clear_distracted() handle it

            except Exception as e:
                if HAS_NVML and isinstance(e, pynvml.NVMLError):
                    logger.error(f"Error while polling GPU stats: {e}")
                    self.gpu_utilization = None
                    self.gpu_memory_free = None
                    self.gpu_memory_total = None
                else:
                    logger.error(f"Error in resource monitor: {e}")

            time.sleep(self.poll_interval)

    def is_distracted(self) -> bool:
        """Return True if sustained load has been detected."""
        return self._distracted

    def check_and_clear_distracted(self) -> bool:
        """Take 3 samples over 3s; clear distracted if all below threshold.

        Returns True if distracted was cleared, False if still distracted.
        """
        if not self._distracted:
            return True  # already clear

        for _ in range(3):
            peak = 0.0
            if HAS_NVML and self.gpu_utilization is not None:
                peak = max(peak, float(self.gpu_utilization))
            if HAS_PSUTIL and self.cpu_utilization is not None:
                peak = max(peak, self.cpu_utilization)
            if peak > self._distracted_threshold:
                return False
            time.sleep(1.0)

        self._distracted = False
        self._sustained_check_start = None
        return True

    @classmethod
    def get_instance(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = cls(*args, **kwargs)
        return cls._instance


def shutdown_monitor():
    if ResourceMonitor._instance:
        ResourceMonitor._instance.stop()
        if HAS_NVML:
            pynvml.nvmlShutdown()

import atexit
atexit.register(shutdown_monitor)

import threading
import time
import logging
from typing import Optional

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except (ImportError, pynvml.NVMLError):
    HAS_NVML = False

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

    def __init__(self, poll_interval: int = 5):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True
        
        self.poll_interval = poll_interval
        self.gpu_utilization = None
        self.gpu_memory_free = None
        self.gpu_memory_total = None
        
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if HAS_NVML:
            self.start()
        else:
            logger.warning("pynvml is not installed or failed to initialize. GPU resource monitoring is disabled.")

    def start(self):
        if not HAS_NVML:
            return
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._monitor, daemon=True)
            self._thread.start()
            logger.info("GPU resource monitor started.")

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join()
            logger.info("GPU resource monitor stopped.")

    def _monitor(self):
        while not self._stop_event.is_set():
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    # For simplicity, we'll monitor the first device
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

            except pynvml.NVMLError as e:
                logger.error(f"Error while polling GPU stats: {e}")
                self.gpu_utilization = None
                self.gpu_memory_free = None
                self.gpu_memory_total = None
                self._stop_event.set() # Stop monitoring on error

            time.sleep(self.poll_interval)

    def is_distracted(self, threshold: int = 80) -> bool:
        if not HAS_NVML or self.gpu_utilization is None:
            return False
        return self.gpu_utilization > threshold

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

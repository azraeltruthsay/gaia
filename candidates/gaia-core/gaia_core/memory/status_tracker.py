import threading

class GAIAStatus:
    """
    Thread-safe global status manager for GAIA boot and runtime state.
    Allows concurrent modules to update and query boot sequence and health.
    """
    _status = {}
    _lock = threading.Lock()

    @classmethod
    def update(cls, key: str, value):
        with cls._lock:
            cls._status[key] = value

    @classmethod
    def get(cls, key: str, default=None):
        with cls._lock:
            return cls._status.get(key, default)

    @classmethod
    def as_dict(cls):
        with cls._lock:
            return dict(cls._status)

    @classmethod
    def clear(cls):
        with cls._lock:
            cls._status.clear()
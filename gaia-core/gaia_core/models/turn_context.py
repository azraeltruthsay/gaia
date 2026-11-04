"""Which conversation turn this thread's model calls belong to, for the polygraph.

Turns reach the engine through many paths, so the id is set around each step of
run_turn and read where requests are built. Thread-local on purpose: a heartbeat
on another thread then logs no turn rather than the wrong one (6k9i)."""
import threading

_local = threading.local()


def set_turn(session_id: str, turn_id: str) -> None:
    _local.ids = {"session_id": session_id or "", "turn_id": turn_id or ""}


def clear_turn() -> None:
    _local.ids = {}


def turn_ids() -> dict:
    return dict(getattr(_local, "ids", {}))

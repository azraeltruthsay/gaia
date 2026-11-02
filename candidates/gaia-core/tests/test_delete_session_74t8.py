"""DELETE /api/sessions ran the archive's LLM calls on the event loop, so bulk
deletes starved /health until the doctor restarted gaia-core (74t8)."""

import asyncio
import threading
from types import SimpleNamespace

from gaia_core import main


class _Sessions:
    def __init__(self):
        self.sessions = {"s": SimpleNamespace(history=[{"role": "user", "content": "hi"}])}
        self.archive_thread = None
        self.reset = False

    def summarize_and_archive(self, session_id):
        self.archive_thread = threading.current_thread()

    def reset_session(self, session_id):
        self.reset = True


def _delete(monkeypatch, **kwargs):
    sm = _Sessions()
    monkeypatch.setattr(main, "_ai_manager", SimpleNamespace(session_manager=sm))
    result = asyncio.run(main.delete_session("s", **kwargs))
    assert result["status"] == "deleted" and sm.reset
    return sm


def test_archive_runs_off_the_event_loop(monkeypatch):
    sm = _delete(monkeypatch)
    assert sm.archive_thread is not None
    assert sm.archive_thread is not threading.main_thread()


def test_archive_false_skips_the_summary(monkeypatch):
    assert _delete(monkeypatch, archive=False).archive_thread is None

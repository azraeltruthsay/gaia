"""wj5b: a session hitting its message limit archived inside the turn; the
summary went to Prime on standby and the reply waited 15 s."""
import threading
import time

from gaia_core.memory import session_manager as sm_mod


def test_reaching_the_limit_does_not_hold_the_turn(tmp_path, monkeypatch):
    from gaia_core.config import get_config
    sm = sm_mod.SessionManager(get_config())
    monkeypatch.setattr(sm_mod, "SESSIONS_FILE", str(tmp_path / "s.json"), raising=False)
    monkeypatch.setattr(sm, "_save_state", lambda: None)
    started, done = threading.Event(), threading.Event()

    def slow_archive(session_id):
        started.set()
        time.sleep(1.0)
        done.set()

    monkeypatch.setattr(sm, "summarize_and_archive", slow_archive)
    sm.max_active_messages = 2
    t0 = time.time()
    sm.add_message("wj5b_test", "user", "one")
    sm.add_message("wj5b_test", "user", "two")       # reaches the limit
    assert time.time() - t0 < 0.5
    assert started.wait(2) and done.wait(3)


def test_summaries_go_to_the_mind_on_the_gpu(monkeypatch):
    from gaia_core.memory.conversation import summarizer as s

    class R:
        def __init__(self, state): self.state = state
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return ('{"state": "%s"}' % self.state).encode()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=3: R("awake"))
    assert s._gpu_mind() == "core"
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=3: R("focusing"))
    assert s._gpu_mind() == "prime"
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=3: (_ for _ in ()).throw(OSError()))
    assert s._gpu_mind() == "core"

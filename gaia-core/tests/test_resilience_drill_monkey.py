"""a57i: the sleep resilience drill asks gaia-monkey for a drill until she is
Serene; serenity is earned only there, and nothing else ever triggered it."""
import sys
import threading
import types
from unittest.mock import MagicMock

import pytest

from gaia_core.cognition.sleep_task_scheduler import SleepTaskScheduler


def _scheduler(monkeypatch, serene):
    s = SleepTaskScheduler.__new__(SleepTaskScheduler)
    s._wake_event = threading.Event()
    s._is_serene = lambda: serene
    s._request_monkey_drill = MagicMock()
    monkeypatch.setitem(sys.modules, "gaia_common.utils.immune_system",
                        types.SimpleNamespace(is_system_irritated=lambda: False))
    mgr = MagicMock()
    mgr.return_value.is_healthy.return_value = True
    monkeypatch.setattr("gaia_core.cognition.candidate_checkpoint.CandidateCheckpointManager", mgr)
    return s


@pytest.mark.parametrize("serene,asked", [(False, True), (True, False)])
def test_drill_asks_monkey_until_serene(monkeypatch, serene, asked):
    monkeypatch.setenv("SERENITY_DRILLS", "1")
    s = _scheduler(monkeypatch, serene)
    s._run_adversarial_resilience_drill()
    assert s._request_monkey_drill.called is asked


def test_drills_are_off_by_default(monkeypatch):
    monkeypatch.delenv("SERENITY_DRILLS", raising=False)
    s = _scheduler(monkeypatch, False)
    s._run_adversarial_resilience_drill()
    assert not s._request_monkey_drill.called


def test_code_evolution_runs_without_serenity(monkeypatch):
    s = SleepTaskScheduler.__new__(SleepTaskScheduler)
    s._is_serene = lambda: False
    ran = []
    monkeypatch.setattr("gaia_common.utils.code_evolution.generate_code_evolution_snapshot",
                        lambda **k: ran.append(k))
    s._run_code_evolution()
    assert ran


def test_request_posts_a_container_drill(monkeypatch):
    s = SleepTaskScheduler.__new__(SleepTaskScheduler)
    posted = {}

    def _post(url, json, timeout):
        posted.update(url=url, json=json)
        return MagicMock(json=lambda: {"drill_results": [], "serenity": {}})

    monkeypatch.setattr("httpx.post", _post)
    started = []
    monkeypatch.setattr(threading, "Thread",
                        lambda target, **k: types.SimpleNamespace(start=lambda: (started.append(1), target())))
    s._request_monkey_drill()
    assert started and posted["json"] == {"drill_types": ["container"]}
    assert posted["url"].endswith("/chaos/inject")

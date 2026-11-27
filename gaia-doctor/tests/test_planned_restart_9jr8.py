"""9jr8: the restart that loads code whose tests passed is not a crash. Counted
as one, the third deploy in 30 minutes tripped the reload guard, sent GAIA an
"URGENT SYSTEM REPAIR" and left the new code unloaded."""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import doctor  # noqa: E402


def _setup(monkeypatch):
    calls, alarms = [], []
    monkeypatch.setattr(doctor, "DRY_RUN", False)
    monkeypatch.setattr(doctor, "is_maintenance_active", lambda: False)
    monkeypatch.setattr(doctor, "_is_meditation_active", lambda: False)
    monkeypatch.setattr(doctor, "run_structural_audit", lambda name: True)
    monkeypatch.setattr(doctor, "_verify_recovery", lambda name, url: None)
    monkeypatch.setattr(doctor, "raise_alarm", lambda name, msg: alarms.append(msg))
    monkeypatch.setattr(doctor, "urlopen", lambda *a, **k: alarms.append("diagnose"))
    monkeypatch.setattr(doctor.subprocess, "run",
                        lambda cmd, **k: calls.append(cmd) or SimpleNamespace(returncode=0, stdout="", stderr=""))
    now = time.monotonic()
    doctor._restart_history["gaia-core"] = [now - 60, now - 30]   # two restarts already
    doctor._last_restart["gaia-core"] = now - 30
    doctor._alarmed_services.discard("gaia-core")
    return calls, alarms


def test_a_deploy_restarts_through_a_full_history_and_is_not_counted(monkeypatch):
    calls, alarms = _setup(monkeypatch)
    assert doctor.docker_restart("gaia-core", planned=True)
    assert any("--force-recreate" in c for c in calls) and not alarms
    assert len(doctor._restart_history["gaia-core"]) == 2


def test_a_crash_restart_is_still_held_by_the_guard(monkeypatch):
    calls, alarms = _setup(monkeypatch)
    assert not doctor.docker_restart("gaia-core")
    assert not any("--force-recreate" in c for c in calls if isinstance(c, list)) and alarms

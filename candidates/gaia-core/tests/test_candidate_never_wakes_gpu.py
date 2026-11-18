"""A candidate never asks the production orchestrator for /gpu/wake
(2026-11-18): a chaos drill stopped gaia-core-candidate, its shutdown
checkpoint hit its standby engine, and its self-healing woke GAIA."""
from unittest.mock import MagicMock

import pytest

from gaia_core.models import vllm_remote_model as vrm


@pytest.mark.parametrize("candidate,posted", [("1", False), ("", True)])
def test_self_healing_wake_skipped_in_candidate(monkeypatch, candidate, posted):
    monkeypatch.setenv("GAIA_IS_CANDIDATE", candidate)
    m = vrm.VLLMRemoteModel.__new__(vrm.VLLMRemoteModel)
    m._training_holds_gpu = lambda: False
    post = MagicMock(return_value=MagicMock(status_code=200))
    monkeypatch.setattr(vrm.requests, "post", post)
    m._request_prime_load()
    assert post.called is posted


def test_sleep_loop_reclaim_skipped_in_candidate(monkeypatch):
    from gaia_core.cognition import sleep_cycle_loop as scl
    monkeypatch.setenv("GAIA_IS_CANDIDATE", "1")
    post = MagicMock()
    monkeypatch.setattr(scl.httpx, "post", post)
    loop = scl.SleepCycleLoop.__new__(scl.SleepCycleLoop)
    loop._reclaim_gpu_for_wake()
    assert not post.called

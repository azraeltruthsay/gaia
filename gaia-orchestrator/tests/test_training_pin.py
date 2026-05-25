"""Tests for ConsciousnessMatrix.training() pin behavior (GAIA_Project-3b4).

Two regressions are pinned:

  1. ``_sync_lifecycle("training")`` must run **before** any
     ``set_target`` calls — the poll loop's ``_is_meditation_active``
     check is what protects the slow tier transitions from being
     undone by auto-reconcile. If sync runs last, the poll loop fires
     mid-transition and reloads tiers we just told it to unload.

  2. nano must be skipped in the tier loop. nano is a socat proxy to
     core in Sovereign Duality; manipulating gaia-nano:8080
     inadvertently unloads core via the alias.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gaia_orchestrator.consciousness_matrix import (
    ConsciousnessLevel,
    ConsciousnessMatrix,
)


@pytest.fixture
def matrix(monkeypatch):
    """Build a ConsciousnessMatrix with all I/O stubbed.

    Returns the matrix plus a `calls` list that records the order of
    set_target / _sync_lifecycle invocations.
    """
    # Stop config loading from reaching shared constants.
    monkeypatch.setattr(
        ConsciousnessMatrix, "_load_config",
        staticmethod(lambda: {
            "inference_endpoints": {
                "nano": "http://test-nano:0",
                "core": "http://test-core:0",
                "prime": "http://test-prime:0",
            },
            "gpu_models": {"nano": "/m/n", "core": "/m/c", "prime": "/m/p"},
            "cpu_models": {"nano": "/m/n.gguf", "core": "/m/c.gguf", "prime": "/m/p.gguf"},
        }),
    )
    cm = ConsciousnessMatrix(lifecycle_machine=None, default_preset="unconscious") \
        if "default_preset" in ConsciousnessMatrix.__init__.__code__.co_varnames \
        else ConsciousnessMatrix(lifecycle_machine=None)

    calls: list[tuple] = []

    async def fake_set_target(tier, level):
        calls.append(("set_target", tier, level))
        cm._tiers[tier].target = level
        return {"ok": True, "tier": tier, "level": level.name}

    async def fake_sync_lifecycle(config_name):
        calls.append(("sync_lifecycle", config_name))
        return {"lifecycle_sync": "ok", "state": "meditation"}

    monkeypatch.setattr(cm, "set_target", fake_set_target)
    monkeypatch.setattr(cm, "_sync_lifecycle", fake_sync_lifecycle)
    cm._calls = calls
    return cm


class TestTrainingOrder:
    @pytest.mark.asyncio
    async def test_sync_lifecycle_runs_before_set_target(self, matrix):
        await matrix.training("prime")
        first = matrix._calls[0]
        assert first == ("sync_lifecycle", "training"), \
            f"Expected sync_lifecycle first, got {matrix._calls!r}"

    @pytest.mark.asyncio
    async def test_set_target_runs_after_sync(self, matrix):
        await matrix.training("prime")
        # All set_target calls come AFTER the sync_lifecycle call
        sync_idx = next(
            i for i, c in enumerate(matrix._calls) if c[0] == "sync_lifecycle"
        )
        set_indices = [i for i, c in enumerate(matrix._calls) if c[0] == "set_target"]
        assert all(i > sync_idx for i in set_indices), \
            f"set_target called before sync: {matrix._calls!r}"


class TestNanoSkipped:
    @pytest.mark.asyncio
    async def test_nano_not_targeted_in_training(self, matrix):
        await matrix.training("prime")
        nano_calls = [c for c in matrix._calls if c[0] == "set_target" and c[1] == "nano"]
        assert nano_calls == [], \
            "nano was targeted — would unload core via socat proxy"

    @pytest.mark.asyncio
    async def test_nano_skipped_even_when_named_as_tier(self, matrix):
        # If someone passes tier="nano" (nonsensical but defensive)
        await matrix.training("nano")
        nano_calls = [c for c in matrix._calls if c[0] == "set_target" and c[1] == "nano"]
        assert nano_calls == []


class TestTargetSemantics:
    @pytest.mark.asyncio
    async def test_prime_tier_goes_unconscious(self, matrix):
        await matrix.training("prime")
        prime_calls = [c for c in matrix._calls if c[0] == "set_target" and c[1] == "prime"]
        assert prime_calls == [("set_target", "prime", ConsciousnessLevel.UNCONSCIOUS)]

    @pytest.mark.asyncio
    async def test_core_goes_subconscious_when_prime_is_training_target(self, matrix):
        await matrix.training("prime")
        core_calls = [c for c in matrix._calls if c[0] == "set_target" and c[1] == "core"]
        assert core_calls == [("set_target", "core", ConsciousnessLevel.SUBCONSCIOUS)]

    @pytest.mark.asyncio
    async def test_core_target_when_core_is_training_target(self, matrix):
        await matrix.training("core")
        core_calls = [c for c in matrix._calls if c[0] == "set_target" and c[1] == "core"]
        assert core_calls == [("set_target", "core", ConsciousnessLevel.UNCONSCIOUS)]
        # Prime gets demoted to CPU so it's still answerable
        prime_calls = [c for c in matrix._calls if c[0] == "set_target" and c[1] == "prime"]
        assert prime_calls == [("set_target", "prime", ConsciousnessLevel.SUBCONSCIOUS)]


class TestReturnShape:
    @pytest.mark.asyncio
    async def test_returns_configuration_results_and_lifecycle(self, matrix):
        result = await matrix.training("prime")
        assert result["configuration"] == "training_prime"
        assert "results" in result
        assert "lifecycle" in result
        assert result["lifecycle"]["state"] == "meditation"

    @pytest.mark.asyncio
    async def test_results_omit_nano(self, matrix):
        result = await matrix.training("prime")
        assert "nano" not in result["results"]
        assert "core" in result["results"]
        assert "prime" in result["results"]

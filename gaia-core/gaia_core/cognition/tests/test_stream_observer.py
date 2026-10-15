"""Unit tests for StreamObserver.verify_side_effects.

Tests the post-execution verification layer that checks whether side
effects reported by route_output() actually produced the expected
artifacts (files on disk, successful dispatches, goal shifts).
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gaia_common.protocols.cognition_packet import (
    CognitionPacket,
    Constraints,
    Content,
    Context,
    Governance,
    Header,
    Intent,
    Metrics,
    Model,
    Persona,
    PersonaRole,
    Reasoning,
    Response,
    Routing,
    Safety,
    SessionHistoryRef,
    Status,
    PacketState,
    TargetEngine,
    SystemTask,
    TokenUsage,
)
from gaia_core.utils.stream_observer import StreamObserver


# ── Helpers ──────────────────────────────────────────────────────────


class DummyLLM:
    """Minimal LLM stub that satisfies StreamObserver.__init__."""

    def create_chat_completion(self, **kwargs):
        return {"choices": [{"message": {"content": '{"action":"CONTINUE"}'}}]}


def _make_packet() -> CognitionPacket:
    """Build a minimal CognitionPacket for testing."""
    return CognitionPacket(
        version="0.3",
        header=Header(
            datetime=datetime.now(timezone.utc).isoformat(),
            session_id="test-session",
            packet_id="pkt-obs-001",
            sub_id="sub-001",
            persona=Persona(identity_id="Prime", persona_id="Default", role=PersonaRole.DEFAULT),
            origin="user",
            routing=Routing(target_engine=TargetEngine.PRIME),
            model=Model(name="test", provider="test", context_window_tokens=4096),
        ),
        intent=Intent(user_intent="greeting", system_task=SystemTask.STREAM, confidence=0.9),
        context=Context(
            session_history_ref=SessionHistoryRef(type="ref", value="test"),
            cheatsheets=[],
            constraints=Constraints(max_tokens=1024, time_budget_ms=5000, safety_mode="standard"),
        ),
        content=Content(original_prompt="Hello!"),
        reasoning=Reasoning(),
        response=Response(candidate="", confidence=0.0, stream_proposal=False),
        governance=Governance(safety=Safety(execution_allowed=True, dry_run=True)),
        metrics=Metrics(token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0), latency_ms=0),
        status=Status(finalized=False, state=PacketState.PROCESSING),
    )


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.constants = {
        "OBSERVER_VERIFY_SIDE_EFFECTS": True,
        "OBSERVER_USE_LLM": False,
    }
    config.cheat_sheet = {}
    return config


@pytest.fixture
def observer(mock_config):
    return StreamObserver(config=mock_config, llm=DummyLLM(), name="Test-Observer")


@pytest.fixture
def packet():
    return _make_packet()


# ── Test: thought seed file exists ───────────────────────────────────


def test_verify_thought_seed_file_exists(observer, packet, tmp_path):
    """Seed file at reported path -> OK."""
    seed_file = tmp_path / "seed_20260218_120000.json"
    seed_file.write_text(json.dumps({"seed": "test idea", "reviewed": False}))

    route_result = {
        "side_effects": [
            {"type": "thought_seed", "path": str(seed_file), "ok": True},
        ],
    }
    result = observer.verify_side_effects(packet, route_result)
    assert result.level == "OK"
    assert "verified" in result.reason.lower() or "All side effects" in result.reason


# ── Test: thought seed file missing ──────────────────────────────────


def test_verify_thought_seed_file_missing(observer, packet, tmp_path):
    """Path in side_effects but no file -> CAUTION."""
    route_result = {
        "side_effects": [
            {"type": "thought_seed", "path": str(tmp_path / "nonexistent.json"), "ok": True},
        ],
    }
    result = observer.verify_side_effects(packet, route_result)
    assert result.level == "CAUTION"
    assert "not found" in result.reason


# ── Test: thought seed file empty ────────────────────────────────────


def test_verify_thought_seed_file_empty(observer, packet, tmp_path):
    """File exists but is 0 bytes -> CAUTION."""
    seed_file = tmp_path / "seed_empty.json"
    seed_file.write_text("")

    route_result = {
        "side_effects": [
            {"type": "thought_seed", "path": str(seed_file), "ok": True},
        ],
    }
    result = observer.verify_side_effects(packet, route_result)
    assert result.level == "CAUTION"
    assert "empty" in result.reason.lower() or "trivially small" in result.reason.lower()


# ── Test: sidecar action success ─────────────────────────────────────


def test_verify_sidecar_action_success(observer, packet):
    """Dispatch ok=True -> OK."""
    route_result = {
        "side_effects": [
            {"type": "sidecar_action", "action_type": "write_file", "ok": True, "error": None},
        ],
    }
    result = observer.verify_side_effects(packet, route_result)
    assert result.level == "OK"


# ── Test: sidecar action error ───────────────────────────────────────


def test_verify_sidecar_action_error(observer, packet):
    """Dispatch ok=False -> CAUTION with error detail."""
    route_result = {
        "side_effects": [
            {"type": "sidecar_action", "action_type": "run_shell", "ok": False, "error": "timeout"},
        ],
    }
    result = observer.verify_side_effects(packet, route_result)
    assert result.level == "CAUTION"
    assert "timeout" in result.reason
    assert "EXECUTE" in result.reason


# ── Test: goal shift ok ──────────────────────────────────────────────


def test_verify_goal_shift_ok(observer, packet):
    """Goal shift ok=True -> OK."""
    route_result = {
        "side_effects": [
            {"type": "goal_shift", "goal": "user wants to learn Python", "ok": True},
        ],
    }
    result = observer.verify_side_effects(packet, route_result)
    assert result.level == "OK"


# ── Test: no side effects ────────────────────────────────────────────


def test_verify_no_side_effects(observer, packet):
    """Empty side_effects list, no directives in LLM output -> OK."""
    route_result = {"side_effects": []}
    result = observer.verify_side_effects(packet, route_result, llm_output="Hello there!")
    assert result.level == "OK"


# ── Test: disabled by config ─────────────────────────────────────────


def test_verify_disabled_by_config(mock_config, packet):
    """Toggle off -> OK without checking anything."""
    mock_config.constants["OBSERVER_VERIFY_SIDE_EFFECTS"] = False
    obs = StreamObserver(config=mock_config, llm=DummyLLM(), name="Test-Disabled")

    route_result = {
        "side_effects": [
            {"type": "thought_seed", "path": "/nonexistent/path.json", "ok": True},
        ],
    }
    result = obs.verify_side_effects(packet, route_result)
    assert result.level == "OK"
    assert "disabled" in result.reason.lower()


# ── Test: fallback thought seed from LLM text ────────────────────────


def test_verify_fallback_thought_seed(observer, packet, tmp_path, monkeypatch):
    """No side_effects key, THOUGHT_SEED in output -> fallback check."""
    # Create a recent seed file in a mock seeds dir
    seeds_dir = tmp_path / "seeds"
    seeds_dir.mkdir()
    seed_file = seeds_dir / "seed_20260218_999999.json"
    seed_file.write_text(json.dumps({"seed": "fallback test"}))

    # The fallback code does `from pathlib import Path as _Path` locally,
    # so we monkeypatch pathlib.Path to redirect "/knowledge/seeds".
    _OrigPath = Path

    class _MockPath(_OrigPath):
        def __new__(cls, *args, **kwargs):
            p = args[0] if args else ""
            if str(p) == "/knowledge/seeds":
                return _OrigPath.__new__(cls, str(seeds_dir))
            return _OrigPath.__new__(cls, *args, **kwargs)

    monkeypatch.setattr("pathlib.Path", _MockPath)

    route_result = {}  # No side_effects key at all
    result = observer.verify_side_effects(
        packet, route_result, llm_output="THOUGHT_SEED: something interesting"
    )
    assert result.level == "OK"
    assert "fallback" in result.reason.lower() or "verified" in result.reason.lower()


# ── Test: appends reflection log ─────────────────────────────────────


def test_verify_appends_reflection_log(observer, packet):
    """Verify that a ReflectionLog entry is appended to packet."""
    route_result = {
        "side_effects": [
            {"type": "sidecar_action", "action_type": "ls", "ok": True, "error": None},
        ],
    }
    initial_count = len(packet.reasoning.reflection_log)
    observer.verify_side_effects(packet, route_result)

    assert len(packet.reasoning.reflection_log) > initial_count
    last_entry = packet.reasoning.reflection_log[-1]
    step = getattr(last_entry, "step", "")
    assert step == "observer_side_effect_verification"


# ── Test: multiple issues ────────────────────────────────────────────


def test_verify_multiple_issues(observer, packet, tmp_path):
    """Mixed pass/fail -> CAUTION with aggregated message."""
    good_file = tmp_path / "seed_good.json"
    good_file.write_text(json.dumps({"seed": "good one"}))

    route_result = {
        "side_effects": [
            {"type": "thought_seed", "path": str(good_file), "ok": True},
            {"type": "thought_seed", "path": str(tmp_path / "missing.json"), "ok": True},
            {"type": "sidecar_action", "action_type": "write_file", "ok": False, "error": "denied"},
        ],
    }
    result = observer.verify_side_effects(packet, route_result)
    assert result.level == "CAUTION"
    assert "2 issue" in result.reason


# ── Test: no packet returns ok ───────────────────────────────────────


def test_verify_no_packet_returns_ok(observer):
    """packet=None -> OK gracefully."""
    route_result = {
        "side_effects": [
            {"type": "thought_seed", "path": "/some/path.json", "ok": True},
        ],
    }
    result = observer.verify_side_effects(None, route_result)
    assert result.level == "OK"
    assert "No packet" in result.reason


# ── Worth-voicing (gate-2, A3): routed through the Observer's user path ───────

def test_user_path_flags_substantial_meta_commentary(observer, packet):
    """observe_user_path is the worth-voicing AUTHORITY: >=2 leaked meta
    sentences (via the integrated Voice Gate detector) -> CAUTION."""
    meta = ("The how is a probe plus a social register. "
            "The answer expects the register it is connected to.")
    result = observer.observe_user_path(packet, meta)
    assert result.level == "CAUTION"
    assert "meta-commentary" in result.reason.lower()
    assert "directly" in (result.suggestion or "").lower()


def test_user_path_single_meta_left_to_strip(observer, packet):
    """A single meta sentence is below the Observer's substantial-leakage bar
    (left to the Voice Gate strip) -> the Observer passes it OK."""
    result = observer.observe_user_path(packet, "The how is a probe plus a social register.")
    assert result.level == "OK"


def test_user_path_clean_output_ok(observer, packet):
    """Ordinary content with no meta-commentary -> OK."""
    result = observer.observe_user_path(packet, "Sure — the capital of France is Paris.")
    assert result.level == "OK"


def test_observer_health_snapshot_shape():
    """observer_health() returns the doctor/health-endpoint snapshot (A3)."""
    from gaia_core.utils.stream_observer import observer_health
    h = observer_health()
    assert set(("observations", "failures", "fail_rate", "healthy")).issubset(h.keys())
    assert isinstance(h["healthy"], bool)


class _NoneForUnmappedConfig:
    """Mimics agent_core.SafeConfigProxy's real behavior: any attribute it
    doesn't explicitly recognize resolves to None (not AttributeError),
    which silently defeats a caller's own getattr(..., default) default.
    Deliberately NOT a MagicMock, which would auto-create a truthy attribute
    for anything and mask this exact bug (GAIA_Project-md2o)."""

    constants: dict = {}
    # CoreIdentityGuardian (constructed inside StreamObserver.__init__) does
    # a bare self.config.identity_file_path access with no default -- give
    # it a harmless nonexistent path rather than None so this stub exercises
    # only the OBSERVER_MIN_INTERVAL behavior under test.
    identity_file_path = "/nonexistent/identity.json"

    def __getattr__(self, name):
        return None


class TestMinIntervalDefault:
    """GAIA_Project-md2o: OBSERVER_MIN_INTERVAL isn't a real Config
    attribute anywhere in the codebase, so under the real SafeConfigProxy
    this always resolved to None instead of StreamObserver's intended 0.5
    default — and _min_interval is later passed through float(), which
    raises TypeError on None the first time two observations are compared."""

    def test_defaults_to_0_5_when_proxy_returns_none(self):
        obs = StreamObserver(config=_NoneForUnmappedConfig(), llm=None, name="Test")
        assert obs._min_interval == 0.5
        float(obs._min_interval)  # must not raise

    def test_respects_a_real_configured_value(self):
        config = _NoneForUnmappedConfig()
        config.OBSERVER_MIN_INTERVAL = 2.5
        obs = StreamObserver(config=config, llm=None, name="Test")
        assert obs._min_interval == 2.5

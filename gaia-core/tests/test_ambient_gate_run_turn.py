"""GAIA's Grotto (2026-11-12): run_turn's ambient gate — a channel turn she
wasn't addressed in stays silent unless gaia_core.cognition.ambient_interject
.should_interject says otherwise, and the gate never fires at all for a
normal (addressed_to_gaia defaults True) turn."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from gaia_core.cognition import agent_core as ac
from gaia_core.cognition.agent_core import AgentCore

SESSION = "discord_channel_1484003096337191067"


def _core():
    core = AgentCore.__new__(AgentCore)
    core.logger = MagicMock()
    core.config = SimpleNamespace(constants={})
    core.session_manager = MagicMock()
    core.session_manager.get_history.return_value = []
    core.model_pool = MagicMock()
    core.model_pool.get_embed_model.return_value = None  # skip the told-fact soft-intent path cleanly
    core._emit_timeline_message = MagicMock()
    core.generate_continuation = MagicMock(return_value=iter([]))
    return core


def _consume(gen):
    return list(gen)


def test_ambient_turn_stays_silent_when_she_decides_not_to(monkeypatch):
    monkeypatch.setattr(ac, "logger", MagicMock())
    monkeypatch.setattr("gaia_core.cognition.ambient_interject.should_interject", lambda *a, **kw: False)
    core = _core()
    events = _consume(core.run_turn(
        "just vibing in the grotto today", SESSION, source="discord_channel",
        metadata={"addressed_to_gaia": False, "channel_id": "123"},
    ))
    assert events == [{"type": "flush"}]
    core.session_manager.add_message.assert_any_call(SESSION, "user", "just vibing in the grotto today")
    # the assistant side was never touched -- nothing was ever generated
    assert not any(c.args[1] == "assistant" for c in core.session_manager.add_message.call_args_list)


def test_ambient_gate_is_skipped_entirely_when_addressed(monkeypatch):
    calls = []
    monkeypatch.setattr("gaia_core.cognition.ambient_interject.should_interject",
                        lambda *a, **kw: calls.append(1) or False)
    core = _core()
    # addressed_to_gaia defaults True (no metadata key at all) -- run far
    # enough to prove the gate was never consulted; whatever happens after
    # that in the real pipeline is exercised by every other run_turn test.
    try:
        next(core.run_turn("hello", SESSION, source="discord_channel", metadata={"channel_id": "123"}))
    except Exception:
        pass
    assert not calls


def test_ambient_turn_records_the_message_even_though_silent(monkeypatch):
    monkeypatch.setattr("gaia_core.cognition.ambient_interject.should_interject", lambda *a, **kw: False)
    core = _core()
    _consume(core.run_turn(
        "anyone around?", SESSION, source="discord_channel",
        metadata={"addressed_to_gaia": False, "channel_id": "123"},
    ))
    core._emit_timeline_message.assert_called_once_with(SESSION, "user", "discord_channel")


def test_ambient_turn_falls_through_when_she_decides_to_speak(monkeypatch):
    # Past the gate, "research orchestration" (or whatever comes right after
    # it) is the very next thing run_turn does -- confirm we actually got
    # there by making it the observable boundary: it's unmocked here, so it
    # either runs real code (not an immediate silent return) or raises --
    # either way, NOT the clean single-flush the silent path produces.
    monkeypatch.setattr("gaia_core.cognition.ambient_interject.should_interject", lambda *a, **kw: True)
    core = _core()
    events = []
    try:
        for e in core.run_turn(
            "anyone around?", SESSION, source="discord_channel",
            metadata={"addressed_to_gaia": False, "channel_id": "123"},
        ):
            events.append(e)
            if len(events) > 3:
                break
    except Exception:
        pass
    assert events != [{"type": "flush"}]

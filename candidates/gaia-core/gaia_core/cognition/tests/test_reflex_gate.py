"""Tests for the reflex skip-gate (GAIA_Project-19i)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gaia_core.cognition.reflex_gate import (
    _has_kb_field,
    _has_tier_prefix,
    _is_focusing,
    should_skip_reflex,
)


def _packet_with_fields(fields: list[tuple[str, str]]):
    """Build a minimal stand-in for cognition_packet with data_fields."""
    df = [SimpleNamespace(key=k, value=v) for k, v in fields]
    return SimpleNamespace(content=SimpleNamespace(data_fields=df))


class _FakeClient:
    def __init__(self, state_name: str | None = None):
        if state_name is None:
            self._state = None
        else:
            self._state = SimpleNamespace(name=state_name)

    @property
    def current_state(self):
        if self._state is None:
            raise RuntimeError("no state")
        return self._state


class TestHasTierPrefix:
    @pytest.mark.parametrize("text", [
        "prime: list files",
        "Prime: do a thing",
        "PRIME: capitalized",
        "thinker: think harder",
        "oracle: ask the cloud",
        "[prime] something",
        "::prime something",
        "[thinker] deeper",
        "::oracle external",
    ])
    def test_recognized_prefixes(self, text):
        assert _has_tier_prefix(text), f"missed: {text!r}"

    @pytest.mark.parametrize("text", [
        "list files in /shared/",
        "what is the weather",
        "tell me about Marcus Aurelius",
        "",
        None,
        "primer paint",       # 'prime' substring without colon
        "primal scream",      # similar non-tag
    ])
    def test_negative_cases(self, text):
        assert not _has_tier_prefix(text or ""), f"false positive: {text!r}"


class TestHasKbField:
    def test_missing_packet(self):
        assert _has_kb_field(None) is False

    def test_no_data_fields(self):
        p = SimpleNamespace(content=SimpleNamespace(data_fields=None))
        assert _has_kb_field(p) is False

    def test_no_kb_field(self):
        p = _packet_with_fields([("other_key", "v")])
        assert _has_kb_field(p) is False

    def test_kb_present(self):
        p = _packet_with_fields([("knowledge_base_name", "code_corpus")])
        assert _has_kb_field(p) is True

    def test_kb_empty_value_treated_as_absent(self):
        p = _packet_with_fields([("knowledge_base_name", "")])
        assert _has_kb_field(p) is False

    def test_kb_after_other_fields(self):
        p = _packet_with_fields([
            ("auto_grounding", "x"),
            ("knowledge_base_name", "Y"),
        ])
        assert _has_kb_field(p) is True

    def test_malformed_field_ignored(self):
        # A field without .key or .value shouldn't crash the check
        p = SimpleNamespace(content=SimpleNamespace(data_fields=[
            SimpleNamespace(),
            SimpleNamespace(key="knowledge_base_name", value="Y"),
        ]))
        assert _has_kb_field(p) is True


class TestIsFocusing:
    def test_none_client(self):
        assert _is_focusing(None) is False

    def test_focusing_state(self):
        assert _is_focusing(_FakeClient("FOCUSING")) is True

    def test_awake_state(self):
        assert _is_focusing(_FakeClient("AWAKE")) is False

    def test_parked_state(self):
        assert _is_focusing(_FakeClient("PARKED")) is False

    def test_client_raises_returns_false(self):
        # A misbehaving client must not propagate the exception
        bad = SimpleNamespace()
        # current_state attribute access raises since it's missing
        assert _is_focusing(bad) is False

    def test_state_without_name_attribute_falls_through(self):
        class _StrState:
            def __str__(self):
                return "LifecycleState.FOCUSING"
        client = SimpleNamespace(current_state=_StrState())
        assert _is_focusing(client) is True


class TestShouldSkipReflex:
    def test_clean_path_runs_reflex(self):
        p = _packet_with_fields([])
        assert should_skip_reflex("list files", p, _FakeClient("AWAKE")) is None

    def test_prime_prefix_skips(self):
        p = _packet_with_fields([])
        reason = should_skip_reflex("prime: list files", p, _FakeClient("AWAKE"))
        assert reason == "explicit tier prefix in user input"

    def test_kb_field_skips(self):
        p = _packet_with_fields([("knowledge_base_name", "code")])
        reason = should_skip_reflex("list files", p, _FakeClient("AWAKE"))
        assert reason == "knowledge_base_name set"

    def test_focusing_skips(self):
        p = _packet_with_fields([])
        reason = should_skip_reflex("list files", p, _FakeClient("FOCUSING"))
        assert reason == "lifecycle state FOCUSING"

    def test_tier_prefix_priority(self):
        # All three conditions met — prefix should be the reason logged
        # (it's the highest-confidence signal of intent)
        p = _packet_with_fields([("knowledge_base_name", "code")])
        reason = should_skip_reflex("prime: list files", p, _FakeClient("FOCUSING"))
        assert reason == "explicit tier prefix in user input"

    def test_kb_before_focusing(self):
        # KB present + FOCUSING — KB takes priority
        p = _packet_with_fields([("knowledge_base_name", "code")])
        reason = should_skip_reflex("list files", p, _FakeClient("FOCUSING"))
        assert reason == "knowledge_base_name set"

    def test_no_client(self):
        # No lifecycle_client at all (e.g. orchestrator not reachable)
        p = _packet_with_fields([])
        assert should_skip_reflex("hello", p, None) is None

    def test_empty_input(self):
        p = _packet_with_fields([])
        assert should_skip_reflex("", p, _FakeClient("AWAKE")) is None

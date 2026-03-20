"""Tests for CodeMind — validator, state machine, scope classification, circuit breaker."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from gaia_common.utils.codemind_engine import (
    CircuitBreaker,
    CodeMindChange,
    CodeMindEngine,
    CodeMindState,
    ScopeTier,
    TriggerSource,
    classify_scope,
    is_vital_organ,
)
from gaia_common.utils.codemind_validator import (
    ValidationResult,
    validate_diff_safety,
    validate_full,
    validate_syntax,
)
from gaia_common.utils.codemind_changelog import (
    append_entry,
    read_entries,
    summary,
)
from gaia_common.utils.codemind_detector import (
    consume_detections,
    deduplicate,
    emit_detection,
    queue_size,
    read_detections,
)


# ══════════════════════════════════════════════════════════════════════════
# Scope Classification
# ══════════════════════════════════════════════════════════════════════════


class TestScopeClassification:
    def test_tier1_knowledge(self):
        assert classify_scope("/knowledge/awareness/foo.md") == ScopeTier.TIER1_AUTO

    def test_tier1_persona(self):
        assert classify_scope("/knowledge/personas/codemind.json") == ScopeTier.TIER1_AUTO

    def test_tier1_constants(self):
        assert classify_scope("gaia_constants.json") == ScopeTier.TIER1_AUTO

    def test_tier1_curriculum(self):
        assert classify_scope("/knowledge/curricula/train.jsonl") == ScopeTier.TIER1_AUTO

    def test_tier2_candidate(self):
        assert classify_scope("/candidates/gaia-core/main.py") == ScopeTier.TIER2_SUPERVISED

    def test_tier2_candidate_relative(self):
        assert classify_scope("candidates/gaia-web/foo.py") == ScopeTier.TIER2_SUPERVISED

    def test_tier3_production(self):
        assert classify_scope("/gaia-core/gaia_core/cognition/agent_core.py") == ScopeTier.TIER3_GATED

    def test_tier3_unknown(self):
        assert classify_scope("/some/other/path.py") == ScopeTier.TIER3_GATED


class TestVitalOrgan:
    def test_vital_organs(self):
        assert is_vital_organ("/gaia-core/main.py") is True
        assert is_vital_organ("agent_core.py") is True
        assert is_vital_organ("/candidates/gaia-mcp/tools.py") is True
        assert is_vital_organ("immune_system.py") is True
        assert is_vital_organ("mcp_client.py") is True

    def test_non_vital(self):
        assert is_vital_organ("observer_scorer.py") is False
        assert is_vital_organ("helpers.py") is False


# ══════════════════════════════════════════════════════════════════════════
# Circuit Breaker
# ══════════════════════════════════════════════════════════════════════════


class TestCircuitBreaker:
    def test_basic_counting(self):
        cb = CircuitBreaker(max_changes=3)
        assert cb.remaining == 3
        assert cb.increment() is True  # 1/3
        assert cb.increment() is True  # 2/3
        assert cb.increment() is False  # 3/3 — tripped
        assert cb.is_tripped() is True

    def test_reset(self):
        cb = CircuitBreaker(max_changes=2)
        cb.increment()
        cb.increment()
        assert cb.is_tripped() is True
        cb.reset()
        assert cb.is_tripped() is False
        assert cb.remaining == 2

    def test_status(self):
        cb = CircuitBreaker(max_changes=5)
        cb.increment()
        s = cb.status()
        assert s["count"] == 1
        assert s["max"] == 5
        assert s["remaining"] == 4
        assert s["tripped"] is False


# ══════════════════════════════════════════════════════════════════════════
# State Machine
# ══════════════════════════════════════════════════════════════════════════


class TestStateMachine:
    def test_initial_state(self):
        engine = CodeMindEngine()
        assert engine.state == CodeMindState.IDLE

    def test_valid_transition_idle_to_detect(self):
        engine = CodeMindEngine()
        assert engine.transition(CodeMindState.DETECT) is True
        assert engine.state == CodeMindState.DETECT

    def test_invalid_transition_idle_to_apply(self):
        engine = CodeMindEngine()
        assert engine.transition(CodeMindState.APPLY) is False
        assert engine.state == CodeMindState.IDLE

    def test_full_happy_path(self):
        engine = CodeMindEngine()
        states = [
            CodeMindState.DETECT,
            CodeMindState.ANALYZE,
            CodeMindState.PROPOSE,
            CodeMindState.VALIDATE,
            CodeMindState.APPLY,
            CodeMindState.VERIFY,
            CodeMindState.PROMOTE,
            CodeMindState.IDLE,
        ]
        for s in states:
            assert engine.transition(s) is True
        assert engine.state == CodeMindState.IDLE

    def test_abort_from_any_state(self):
        """Any state can transition back to IDLE (abort)."""
        engine = CodeMindEngine()
        for state in [CodeMindState.DETECT, CodeMindState.ANALYZE,
                      CodeMindState.PROPOSE, CodeMindState.VALIDATE,
                      CodeMindState.APPLY, CodeMindState.VERIFY]:
            engine.state = state  # force state for test
            assert engine.transition(CodeMindState.IDLE) is True

    def test_reset(self):
        engine = CodeMindEngine()
        engine.transition(CodeMindState.DETECT)
        engine.reset()
        assert engine.state == CodeMindState.IDLE
        assert not engine.circuit_breaker.is_tripped()


class TestCycleManagement:
    def test_start_cycle(self):
        engine = CodeMindEngine()
        cycle = engine.start_cycle(TriggerSource.SLEEP_CYCLE)
        assert cycle.cycle_id.startswith("cm-")
        assert cycle.trigger == "sleep_cycle"
        assert engine.state == CodeMindState.DETECT

    def test_record_change(self):
        engine = CodeMindEngine()
        cycle = engine.start_cycle(TriggerSource.USER_REQUEST)
        change = CodeMindChange(
            file_path="/candidates/gaia-core/foo.py",
            issue="lint error",
            scope_tier=2,
        )
        engine.record_change(change)
        assert len(cycle.changes) == 1
        assert engine.circuit_breaker.remaining == 2

    def test_trigger_allowed(self):
        engine = CodeMindEngine({"CODEMIND": {"triggers": {"user_request": True, "drift_detection": False}}})
        assert engine.is_trigger_allowed(TriggerSource.USER_REQUEST) is True
        assert engine.is_trigger_allowed(TriggerSource.DRIFT_DETECTION) is False

    def test_scope_allowed(self):
        engine = CodeMindEngine({"CODEMIND": {"scope_tiers": {"tier1_auto": True, "tier2_supervised": True, "tier3_gated": False}}})
        assert engine.is_scope_allowed("/knowledge/awareness/foo.md") is True
        assert engine.is_scope_allowed("candidates/gaia-core/foo.py") is True
        assert engine.is_scope_allowed("/gaia-core/main.py") is False

    def test_status(self):
        engine = CodeMindEngine()
        s = engine.status()
        assert s["state"] == "IDLE"
        assert "circuit_breaker" in s
        assert "enabled" in s


# ══════════════════════════════════════════════════════════════════════════
# Validator
# ══════════════════════════════════════════════════════════════════════════


class TestValidator:
    def test_valid_python(self):
        result = validate_syntax("x = 1\nprint(x)\n", "test.py")
        assert result.passed is True
        assert result.py_compile_ok is True
        assert result.ast_parse_ok is True

    def test_invalid_python(self):
        result = validate_syntax("def foo(\n", "test.py")
        assert result.passed is False
        assert result.ast_parse_ok is False

    def test_empty_file(self):
        result = validate_syntax("", "test.py")
        assert result.passed is True

    def test_full_validation(self):
        code = "import os\n\nx = os.getcwd()\nprint(x)\n"
        result = validate_full(code, "test.py", {"py_compile": True, "ast_parse": True, "ruff": False})
        assert result.passed is True

    def test_full_validation_syntax_error(self):
        code = "def foo(:\n    pass\n"
        result = validate_full(code, "test.py")
        assert result.passed is False
        assert len(result.errors) > 0


class TestDiffSafety:
    def test_safe_change(self):
        original = "x = 1\ny = 2\nz = 3\n"
        proposed = "x = 1\ny = 3\nz = 3\n"
        result = validate_diff_safety(original, proposed)
        assert result["safe"] is True

    def test_total_wipe(self):
        original = "x = 1\ny = 2\n"
        proposed = ""
        result = validate_diff_safety(original, proposed)
        assert result["safe"] is False
        assert "deletes all content" in result["reason"]

    def test_excessive_change(self):
        original = "\n".join(f"line_{i} = {i}" for i in range(20))
        proposed = "\n".join(f"CHANGED_{i} = {i}" for i in range(20))
        result = validate_diff_safety(original, proposed, max_change_ratio=0.3)
        assert result["safe"] is False

    def test_new_file(self):
        result = validate_diff_safety("", "x = 1\n")
        assert result["safe"] is True


# ══════════════════════════════════════════════════════════════════════════
# Changelog
# ══════════════════════════════════════════════════════════════════════════


class TestChangelog:
    def test_append_and_read(self, tmp_path):
        path = str(tmp_path / "changelog.jsonl")
        append_entry({"cycle_id": "cm-1", "outcome": "complete"}, path)
        append_entry({"cycle_id": "cm-2", "outcome": "error"}, path)
        entries = read_entries(path)
        assert len(entries) == 2
        # Newest first
        assert entries[0]["cycle_id"] == "cm-2"

    def test_summary(self, tmp_path):
        path = str(tmp_path / "changelog.jsonl")
        for i in range(5):
            append_entry({"cycle_id": f"cm-{i}", "outcome": "complete"}, path)
        append_entry({"cycle_id": "cm-err", "outcome": "error"}, path)
        s = summary(path)
        assert s["total"] == 6
        assert s["outcomes"]["complete"] == 5
        assert s["outcomes"]["error"] == 1

    def test_empty_changelog(self, tmp_path):
        path = str(tmp_path / "nonexistent.jsonl")
        entries = read_entries(path)
        assert entries == []


# ══════════════════════════════════════════════════════════════════════════
# Detector
# ══════════════════════════════════════════════════════════════════════════


class TestDetector:
    def test_emit_and_read(self, tmp_path):
        path = str(tmp_path / "queue.jsonl")
        emit_detection(
            source="immune_irritation",
            issue_type="lint_error",
            file_path="/candidates/gaia-core/foo.py",
            description="unused import",
            queue_path=path,
        )
        entries = read_detections(path)
        assert len(entries) == 1
        assert entries[0]["source"] == "immune_irritation"

    def test_priority_ordering(self, tmp_path):
        path = str(tmp_path / "queue.jsonl")
        emit_detection("sleep_cycle", "lint", "/a.py", "low pri", queue_path=path)
        emit_detection("user_request", "bug", "/b.py", "high pri", queue_path=path)
        entries = read_detections(path)
        assert entries[0]["source"] == "user_request"  # priority 1
        assert entries[1]["source"] == "sleep_cycle"    # priority 4

    def test_deduplicate(self):
        entries = [
            {"file_path": "/a.py", "issue_type": "lint"},
            {"file_path": "/a.py", "issue_type": "lint"},
            {"file_path": "/b.py", "issue_type": "lint"},
        ]
        result = deduplicate(entries)
        assert len(result) == 2

    def test_consume(self, tmp_path):
        path = str(tmp_path / "queue.jsonl")
        for i in range(5):
            emit_detection("sleep_cycle", "lint", f"/file_{i}.py", f"issue {i}", queue_path=path)
        consumed = consume_detections(path, limit=3)
        assert len(consumed) == 3
        # Remaining should be 2
        assert queue_size(path) == 2

    def test_queue_size(self, tmp_path):
        path = str(tmp_path / "queue.jsonl")
        assert queue_size(path) == 0
        emit_detection("immune_irritation", "lint", "/a.py", "test", queue_path=path)
        assert queue_size(path) == 1

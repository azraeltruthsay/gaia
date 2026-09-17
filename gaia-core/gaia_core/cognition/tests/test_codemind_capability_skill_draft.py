"""9ar0: CodeMind capability_gap -> real drafted skill.

Reproduces the exact nxxl payload shape (Core hallucinating a `poetry`
tool on a haiku request, then vp52 planting a capability_gap thought seed
for it) and verifies the new drafting path lands an inert, AST-valid,
sandbox-tested draft under knowledge/skills/auto/ — never a live,
routable skill.
"""

import ast
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gaia_common.utils.codemind_engine import CodeMindEngine
from gaia_core.cognition.sleep_task_scheduler import SleepTaskScheduler


GOOD_SKILL_CODE = '''
def execute(params: dict) -> dict:
    """Compose a short poem about the requested subject."""
    topic = params.get("words") or params.get("topic") or "the moment"
    return {"ok": True, "content": f"A haiku about {topic}."}
'''

UNSAFE_SKILL_CODE = '''
import subprocess

def execute(params: dict) -> dict:
    subprocess.run(["echo", "hi"])
    return {"ok": True}
'''

BROKEN_SKILL_CODE = "def execute(params:\n    return"


@pytest.fixture
def bare_scheduler(tmp_path):
    """Minimal scheduler — only what _draft_capability_skill/_mark_seed_reviewed touch.

    config.KNOWLEDGE_DIR is set explicitly (not left to a relative-path/cwd
    assumption) — _draft_capability_skill resolves its draft location from
    this, matching main.py's architecture_facts.md convention. A relative
    "knowledge/..." path was verified live to resolve to the wrong
    directory inside the real gaia-core-candidate container (cwd /app has
    its own stray, near-empty "knowledge/" dir; the real tree is mounted
    at the absolute path in KNOWLEDGE_DIR).
    """
    s = SleepTaskScheduler.__new__(SleepTaskScheduler)
    s.config = SimpleNamespace(KNOWLEDGE_DIR=str(tmp_path / "knowledge"))
    return s


@pytest.fixture
def engine():
    return CodeMindEngine(constants=None)


@pytest.fixture
def cycle():
    return SimpleNamespace(dry_run=False)


def _nxxl_seed_text() -> str:
    """The exact seed-text shape vp52 plants (main.py:1516-1529) for the
    real nxxl case: Core hallucinating a `poetry` tool on a haiku request."""
    payload = {
        "attempted_tool_name": "poetry",
        "attempted_tool_action": "rhyme",
        "attempted_tool_params": {"words": ["moon", "june", "soon"]},
        "user_request": "Can you please make a haiku for me?",
        "session_id": "test-session-nxxl",
        "timestamp": "2026-09-16T00:00:00+00:00",
    }
    return (
        "THOUGHT_SEED: capability gap — hallucinated tool call. "
        "No tool named 'poetry' exists; this could be "
        "researched and built as a real capability (knowledge gap). "
        f"Payload: {json.dumps(payload)}"
    )


def _nxxl_detection() -> dict:
    return {
        "issue_type": "capability_gap",
        "description": _nxxl_seed_text(),
        "file_path": "",
        "metadata": {"seed_file": "seed_20260916_000000000000.json"},
    }


class TestParsePayload:
    def test_extracts_vp52_payload(self, bare_scheduler):
        payload = bare_scheduler._parse_capability_gap_payload(_nxxl_seed_text())
        assert payload is not None
        assert payload["attempted_tool_name"] == "poetry"
        assert payload["attempted_tool_action"] == "rhyme"
        assert payload["user_request"] == "Can you please make a haiku for me?"

    def test_generic_gap_seed_returns_none(self, bare_scheduler):
        """A capability_gap seed with no vp52 payload (the pre-existing
        knowledge-gap shape) must fall through untouched — this is the
        safety net that keeps the existing awareness-note path alive."""
        payload = bare_scheduler._parse_capability_gap_payload(
            "THOUGHT_SEED: I don't know the population of Elbonia; this "
            "could be researched."
        )
        assert payload is None

    def test_malformed_payload_json_returns_none(self, bare_scheduler):
        payload = bare_scheduler._parse_capability_gap_payload(
            "THOUGHT_SEED: capability gap. Payload: {not valid json"
        )
        assert payload is None


class TestDraftCapabilitySkill:
    def test_happy_path_lands_inert_draft(self, tmp_path, monkeypatch, bare_scheduler, engine, cycle):
        monkeypatch.setattr(bare_scheduler, "_codemind_propose", MagicMock(return_value=GOOD_SKILL_CODE))
        monkeypatch.setattr(bare_scheduler, "_mark_seed_reviewed", MagicMock())

        detection = _nxxl_detection()
        payload = bare_scheduler._parse_capability_gap_payload(detection["description"])
        assert payload is not None

        bare_scheduler._draft_capability_skill(engine, detection, payload, cycle)

        drafts = list((tmp_path / "knowledge" / "skills" / "auto").glob("capgap_*"))
        assert len(drafts) == 1
        skill_dir = drafts[0]

        skill_md = (skill_dir / "SKILL.md").read_text()
        source = (skill_dir / "skill_source.py").read_text()

        # AST-valid — never write a draft that doesn't parse.
        ast.parse(source)

        # Inert by construction: draft + disabled, both required by the
        # SkillGateway routing gate (9ar0 Phase A) to keep it unreachable.
        assert "status: draft" in skill_md
        assert "enabled: false" in skill_md
        assert "codemind_source: capability_gap" in skill_md
        assert "attempted_tool_name: poetry" in skill_md
        # Sandbox ran against the seed's own attempted params and passed.
        assert "sandbox_tested: true" in skill_md

        bare_scheduler._mark_seed_reviewed.assert_called_once()
        assert bare_scheduler._mark_seed_reviewed.call_args[0][1] == "skill_drafted"

    def test_never_writes_into_live_skills_dir(self, tmp_path, monkeypatch, bare_scheduler, engine, cycle):
        """The draft must never land anywhere SkillManager would hot-load
        it from — drafting must be incapable of hot-loading live code."""
        monkeypatch.setattr(bare_scheduler, "_codemind_propose", MagicMock(return_value=GOOD_SKILL_CODE))
        monkeypatch.setattr(bare_scheduler, "_mark_seed_reviewed", MagicMock())

        detection = _nxxl_detection()
        payload = bare_scheduler._parse_capability_gap_payload(detection["description"])
        bare_scheduler._draft_capability_skill(engine, detection, payload, cycle)

        assert not (tmp_path / "candidates" / "gaia-mcp" / "gaia_mcp" / "skills").exists()

    def test_scope_tier_is_hardcoded_2_never_1(self, tmp_path, monkeypatch, bare_scheduler, engine, cycle):
        """classify_scope() would call anything under /knowledge/ Tier1-auto
        — explicitly wrong for autonomously-authored executable code
        (the bead: 'NEVER Tier1-auto for this'). record_change must be
        called with scope_tier=2 regardless of file path."""
        monkeypatch.setattr(bare_scheduler, "_codemind_propose", MagicMock(return_value=GOOD_SKILL_CODE))
        monkeypatch.setattr(bare_scheduler, "_mark_seed_reviewed", MagicMock())
        record_change = MagicMock(wraps=engine.record_change)
        monkeypatch.setattr(engine, "record_change", record_change)

        detection = _nxxl_detection()
        payload = bare_scheduler._parse_capability_gap_payload(detection["description"])
        bare_scheduler._draft_capability_skill(engine, detection, payload, cycle)

        recorded = record_change.call_args[0][0]
        assert recorded.scope_tier == 2
        assert recorded.applied is False

    def test_unsafe_draft_is_abandoned_not_written(self, tmp_path, monkeypatch, bare_scheduler, engine, cycle):
        """Both attempts (initial + one reroll) return code that trips the
        safety linter — the draft must be abandoned, not written."""
        monkeypatch.setattr(bare_scheduler, "_codemind_propose", MagicMock(return_value=UNSAFE_SKILL_CODE))
        monkeypatch.setattr(bare_scheduler, "_mark_seed_reviewed", MagicMock())

        detection = _nxxl_detection()
        payload = bare_scheduler._parse_capability_gap_payload(detection["description"])
        bare_scheduler._draft_capability_skill(engine, detection, payload, cycle)

        assert not (tmp_path / "knowledge").exists()
        bare_scheduler._mark_seed_reviewed.assert_called_once_with(detection, "skill_draft_failed")

    def test_broken_syntax_is_abandoned_not_written(self, tmp_path, monkeypatch, bare_scheduler, engine, cycle):
        monkeypatch.setattr(bare_scheduler, "_codemind_propose", MagicMock(return_value=BROKEN_SKILL_CODE))
        monkeypatch.setattr(bare_scheduler, "_mark_seed_reviewed", MagicMock())

        detection = _nxxl_detection()
        payload = bare_scheduler._parse_capability_gap_payload(detection["description"])
        bare_scheduler._draft_capability_skill(engine, detection, payload, cycle)

        assert not (tmp_path / "knowledge").exists()
        bare_scheduler._mark_seed_reviewed.assert_called_once_with(detection, "skill_draft_failed")

    def test_cannot_draft_decline_writes_nothing(self, tmp_path, monkeypatch, bare_scheduler, engine, cycle):
        monkeypatch.setattr(
            bare_scheduler, "_codemind_propose",
            MagicMock(return_value="CANNOT_DRAFT: needs real external API credentials"),
        )
        monkeypatch.setattr(bare_scheduler, "_mark_seed_reviewed", MagicMock())

        detection = _nxxl_detection()
        payload = bare_scheduler._parse_capability_gap_payload(detection["description"])
        bare_scheduler._draft_capability_skill(engine, detection, payload, cycle)

        assert not (tmp_path / "knowledge").exists()
        bare_scheduler._mark_seed_reviewed.assert_called_once_with(detection, "draft_declined")

    def test_dry_run_never_writes_files(self, tmp_path, monkeypatch, bare_scheduler, engine):
        monkeypatch.setattr(bare_scheduler, "_codemind_propose", MagicMock(return_value=GOOD_SKILL_CODE))
        monkeypatch.setattr(bare_scheduler, "_mark_seed_reviewed", MagicMock())

        detection = _nxxl_detection()
        payload = bare_scheduler._parse_capability_gap_payload(detection["description"])
        bare_scheduler._draft_capability_skill(engine, detection, payload, SimpleNamespace(dry_run=True))

        assert not (tmp_path / "knowledge").exists()
        bare_scheduler._mark_seed_reviewed.assert_called_once_with(detection, "skill_drafted_dry_run")

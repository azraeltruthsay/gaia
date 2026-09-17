"""Tests for the SkillGateway draft/enabled gate (9ar0).

Before this gate existed, any SKILL.md package that loaded (including
CodeMind capability_gap drafts under knowledge/skills/auto/) was
immediately routable via do(skill=name, ...) regardless of its
`status: draft` frontmatter — this was harmless only because those drafts
had no backing .py yet. These tests lock in that a draft/disabled package
can never execute or be surfaced by search, while an active package's
behavior is unchanged.
"""

import pytest

from gaia_mcp.skill_gateway import SkillGateway
from gaia_mcp.skill_package import SkillPackage


def _make_gateway() -> SkillGateway:
    # Empty on-disk dir (no packages loaded from disk) — packages for each
    # test are injected directly into ._packages to isolate the gate logic
    # from frontmatter parsing and file I/O.
    import tempfile
    from pathlib import Path
    tmp = tempfile.mkdtemp()
    return SkillGateway(skills_dir=Path(tmp))


def _draft_package(name: str = "draft-skill") -> SkillPackage:
    return SkillPackage(
        name=name,
        description="a CodeMind capability_gap draft awaiting promotion",
        execution_mode="PLAYBOOK",
        status="draft",
        enabled=False,
    )


def _active_package(name: str = "active-skill") -> SkillPackage:
    # PLAYBOOK with no legacy_maps_to routes to SkillManager.execute_limb —
    # a pure in-process lookup, no network dependency (unlike KNOWLEDGE
    # mode, which POSTs to gaia-core). Deterministic and fast for a gate
    # test that only cares whether the gate let execution proceed.
    return SkillPackage(
        name=name,
        description="a promoted, live skill",
        execution_mode="PLAYBOOK",
        status="active",
        enabled=True,
    )


def test_skill_package_defaults_are_routable():
    """Existing hand-written SKILL.md files (no status/enabled frontmatter)
    must keep working unchanged — this is a pure additive field."""
    pkg = SkillPackage(name="legacy-skill", description="predates the gate")
    assert pkg.is_routable


def test_draft_skill_package_is_not_routable():
    pkg = _draft_package()
    assert not pkg.is_routable


@pytest.mark.asyncio
async def test_do_rejects_draft_skill():
    gateway = _make_gateway()
    pkg = _draft_package("draft-skill")
    gateway._packages[pkg.name] = pkg

    result = await gateway.route("do", {"skill": pkg.name, "input": "hello"})

    assert result["ok"] is False
    assert "not active" in result["error"]


@pytest.mark.asyncio
async def test_do_rejects_disabled_but_active_status_skill():
    gateway = _make_gateway()
    pkg = SkillPackage(
        name="disabled-skill", description="active status but explicitly disabled",
        execution_mode="PLAYBOOK", status="active", enabled=False,
    )
    gateway._packages[pkg.name] = pkg

    result = await gateway.route("do", {"skill": pkg.name, "input": "hello"})

    assert result["ok"] is False
    assert "not active" in result["error"]


@pytest.mark.asyncio
async def test_do_proceeds_past_gate_for_active_skill():
    """An active/enabled package must not be rejected by the gate — it may
    still fail downstream (no live gaia-core to call in this unit test),
    but that failure must NOT be the gate's "not active" message, proving
    the gate let it through to the real execution path."""
    gateway = _make_gateway()
    pkg = _active_package("active-skill")
    gateway._packages[pkg.name] = pkg

    result = await gateway.route("do", {"skill": pkg.name, "input": "hello"})

    assert "not active" not in (result.get("error") or "")


def test_find_matching_skills_excludes_draft():
    gateway = _make_gateway()
    draft = _draft_package("draft-poetry-skill")
    draft.description = "poetry haiku rhyme"
    active = _active_package("active-poetry-skill")
    active.description = "poetry haiku rhyme"
    gateway._packages[draft.name] = draft
    gateway._packages[active.name] = active

    matches = gateway._find_matching_skills("poetry haiku rhyme")
    matched_names = [name for name, _score in matches]

    assert draft.name not in matched_names

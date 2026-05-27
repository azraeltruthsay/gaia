"""Tests for the World Model Stage 5b merge-approval gate (GAIA_Project-clm).

Verifies the candidate-file lifecycle: propose writes JSON, approve
flips status, apply requires status=approved unless the gate is off,
reject blocks apply forever.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gaia_common.utils.knowledge_graph import KnowledgeGraph


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def candidates_dir(tmp_path: Path) -> Path:
    return tmp_path / "candidates" / "world_merges"


@pytest.fixture
def gated_kg(tmp_path: Path, candidates_dir: Path) -> KnowledgeGraph:
    """KG with the approval gate ENABLED — production-like."""
    return KnowledgeGraph(
        db_path=str(tmp_path / "kg.sqlite"),
        require_merge_approval=True,
        merge_candidates_dir=str(candidates_dir),
    )


@pytest.fixture
def ungated_kg(tmp_path: Path, candidates_dir: Path) -> KnowledgeGraph:
    """KG with the gate DISABLED — automated / test flows."""
    return KnowledgeGraph(
        db_path=str(tmp_path / "kg.sqlite"),
        require_merge_approval=False,
        merge_candidates_dir=str(candidates_dir),
    )


def _setup_two_mergeable_worlds(kg: KnowledgeGraph) -> tuple[str, str]:
    """Build two simple worlds with one coref-matching entity."""
    kg.create_world(name="src_world", modality="actuality", parent="actuality")
    kg.create_world(name="tgt_world", modality="actuality", parent="actuality")
    kg.add_triple("Rupert", "is_a", "paladin", world="src_world")
    kg.add_triple("Rupert", "is_a", "paladin", world="tgt_world")
    return "src_world", "tgt_world"


# ── Constructor + env-var resolution ────────────────────────────────


class TestConstructor:
    def test_default_gate_on(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GAIA_KG_REQUIRE_MERGE_APPROVAL", raising=False)
        kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite"))
        assert kg.require_merge_approval is True

    def test_env_override_off(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GAIA_KG_REQUIRE_MERGE_APPROVAL", "0")
        kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite"))
        assert kg.require_merge_approval is False

    def test_env_false_string_off(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GAIA_KG_REQUIRE_MERGE_APPROVAL", "false")
        kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite"))
        assert kg.require_merge_approval is False

    def test_explicit_arg_beats_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GAIA_KG_REQUIRE_MERGE_APPROVAL", "0")
        kg = KnowledgeGraph(
            db_path=str(tmp_path / "kg.sqlite"),
            require_merge_approval=True,
        )
        assert kg.require_merge_approval is True


# ── propose writes candidate file ───────────────────────────────────


class TestProposeWritesCandidate:
    def test_candidate_file_appears(self, gated_kg, candidates_dir):
        src, tgt = _setup_two_mergeable_worlds(gated_kg)
        prop = gated_kg.propose_merge(src, tgt, notes="test")
        path = candidates_dir / f"{prop['merge_id']}.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["merge_id"] == prop["merge_id"]
        assert data["status"] == "pending"
        assert data["proposed_at"]
        assert data["approved_at"] is None

    def test_candidate_preserves_proposal_fields(self, gated_kg, candidates_dir):
        src, tgt = _setup_two_mergeable_worlds(gated_kg)
        prop = gated_kg.propose_merge(src, tgt)
        data = json.loads((candidates_dir / f"{prop['merge_id']}.json").read_text())
        assert data["source_world"] == prop["source_world"]
        assert data["target_world"] == prop["target_world"]
        assert data["triples_to_rewrite"] == prop["triples_to_rewrite"]
        assert "entity_mapping" in data

    def test_propose_succeeds_even_without_candidate_dir(
        self, tmp_path, gated_kg,
    ):
        """If the candidate file fails to write, the proposal still
        succeeds (DB row is the source of truth)."""
        src, tgt = _setup_two_mergeable_worlds(gated_kg)
        # Point the candidate dir at a path that's a FILE (write would fail)
        bad_dir = tmp_path / "blocker"
        bad_dir.write_text("not a dir")
        gated_kg.merge_candidates_dir = bad_dir
        # Should not raise
        prop = gated_kg.propose_merge(src, tgt)
        assert prop["merge_id"]


# ── approve_merge / reject_merge ────────────────────────────────────


class TestApproveAndReject:
    def test_approve_flips_status(self, gated_kg, candidates_dir):
        src, tgt = _setup_two_mergeable_worlds(gated_kg)
        prop = gated_kg.propose_merge(src, tgt)
        result = gated_kg.approve_merge(prop["merge_id"])
        assert result["status"] == "approved"
        assert result["approved_at"]
        assert result["approved_by"] == "architect"
        # And the file reflects it
        data = json.loads((candidates_dir / f"{prop['merge_id']}.json").read_text())
        assert data["status"] == "approved"

    def test_approve_with_custom_approver(self, gated_kg):
        src, tgt = _setup_two_mergeable_worlds(gated_kg)
        prop = gated_kg.propose_merge(src, tgt)
        result = gated_kg.approve_merge(prop["merge_id"], approver="azrael")
        assert result["approved_by"] == "azrael"

    def test_approve_missing_candidate_raises(self, gated_kg):
        with pytest.raises(ValueError, match="No candidate file"):
            gated_kg.approve_merge("m_does_not_exist")

    def test_approve_already_approved_raises(self, gated_kg):
        src, tgt = _setup_two_mergeable_worlds(gated_kg)
        prop = gated_kg.propose_merge(src, tgt)
        gated_kg.approve_merge(prop["merge_id"])
        with pytest.raises(ValueError, match="not pending"):
            gated_kg.approve_merge(prop["merge_id"])

    def test_reject_flips_status(self, gated_kg, candidates_dir):
        src, tgt = _setup_two_mergeable_worlds(gated_kg)
        prop = gated_kg.propose_merge(src, tgt)
        result = gated_kg.reject_merge(prop["merge_id"], reason="same name, different person")
        assert result["status"] == "rejected"
        assert result["rejected_reason"] == "same name, different person"
        # Also reflected in candidate file
        data = json.loads((candidates_dir / f"{prop['merge_id']}.json").read_text())
        assert data["status"] == "rejected"

    def test_reject_blocks_subsequent_apply(self, gated_kg):
        src, tgt = _setup_two_mergeable_worlds(gated_kg)
        prop = gated_kg.propose_merge(src, tgt)
        gated_kg.reject_merge(prop["merge_id"], reason="no")
        with pytest.raises(PermissionError):
            gated_kg.apply_merge(prop["merge_id"])


# ── apply_merge gate behavior ───────────────────────────────────────


class TestApplyGate:
    def test_apply_refuses_without_approval(self, gated_kg):
        src, tgt = _setup_two_mergeable_worlds(gated_kg)
        prop = gated_kg.propose_merge(src, tgt)
        with pytest.raises(PermissionError, match="requires approval"):
            gated_kg.apply_merge(prop["merge_id"])

    def test_apply_succeeds_after_approval(self, gated_kg):
        src, tgt = _setup_two_mergeable_worlds(gated_kg)
        prop = gated_kg.propose_merge(src, tgt)
        gated_kg.approve_merge(prop["merge_id"])
        # Now apply should work
        result = gated_kg.apply_merge(prop["merge_id"])
        assert result.get("status") == "applied"

    def test_apply_records_applied_status_in_candidate(self, gated_kg, candidates_dir):
        src, tgt = _setup_two_mergeable_worlds(gated_kg)
        prop = gated_kg.propose_merge(src, tgt)
        gated_kg.approve_merge(prop["merge_id"])
        gated_kg.apply_merge(prop["merge_id"])
        data = json.loads((candidates_dir / f"{prop['merge_id']}.json").read_text())
        assert data["status"] == "applied"
        assert data["applied_at"]

    def test_ungated_kg_applies_without_approval(self, ungated_kg):
        """require_merge_approval=False bypasses the gate entirely —
        the candidate file is still written for traceability but no
        approval is checked before apply."""
        src, tgt = _setup_two_mergeable_worlds(ungated_kg)
        prop = ungated_kg.propose_merge(src, tgt)
        # Apply without approving — should succeed
        result = ungated_kg.apply_merge(prop["merge_id"])
        assert result.get("status") == "applied"

    def test_apply_with_missing_candidate_file(self, gated_kg, candidates_dir):
        """Candidate file vanished mid-flight (e.g. operator deleted it).
        Gate must still refuse rather than silently applying."""
        src, tgt = _setup_two_mergeable_worlds(gated_kg)
        prop = gated_kg.propose_merge(src, tgt)
        # Delete the candidate file
        (candidates_dir / f"{prop['merge_id']}.json").unlink()
        with pytest.raises(PermissionError):
            gated_kg.apply_merge(prop["merge_id"])


# ── End-to-end lifecycle ────────────────────────────────────────────


class TestFullLifecycle:
    def test_propose_approve_apply(self, gated_kg, candidates_dir):
        src, tgt = _setup_two_mergeable_worlds(gated_kg)
        prop = gated_kg.propose_merge(src, tgt, notes="full cycle test")
        mid = prop["merge_id"]

        # Phase 1: pending
        data = json.loads((candidates_dir / f"{mid}.json").read_text())
        assert data["status"] == "pending"

        # Phase 2: approve
        gated_kg.approve_merge(mid, approver="azrael")
        data = json.loads((candidates_dir / f"{mid}.json").read_text())
        assert data["status"] == "approved"

        # Phase 3: apply
        gated_kg.apply_merge(mid)
        data = json.loads((candidates_dir / f"{mid}.json").read_text())
        assert data["status"] == "applied"

    def test_propose_reject_then_apply_fails(self, gated_kg, candidates_dir):
        src, tgt = _setup_two_mergeable_worlds(gated_kg)
        prop = gated_kg.propose_merge(src, tgt)
        mid = prop["merge_id"]
        gated_kg.reject_merge(mid, reason="coref looked wrong on review")
        # Apply must refuse
        with pytest.raises(PermissionError):
            gated_kg.apply_merge(mid)

"""Tests for the gaia-study merge-approval HTTP surface (GAIA_Project-21h).

Stage 5c Phase 1: the API endpoints the gaia-web UI will eventually
consume. Exercises the full lifecycle through HTTP — list pending,
GET detail, approve, reject.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def kg_isolated(tmp_path: Path, monkeypatch):
    """Build a KG with isolated db + candidate dir. Patches the env vars
    the server's lazy KG accessor reads."""
    db_path = tmp_path / "kg.sqlite"
    candidates_dir = tmp_path / "candidates" / "world_merges"
    monkeypatch.setenv("GAIA_KG_DB_PATH", str(db_path))
    monkeypatch.setenv("GAIA_MERGE_CANDIDATES_DIR", str(candidates_dir))
    # Build a KG instance with the same paths the route handlers will
    # construct internally (KG default db_path picks up GAIA_KG_DB_PATH
    # if set; otherwise pass explicitly via factory below).
    from gaia_common.utils.knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph(
        db_path=str(db_path),
        merge_candidates_dir=str(candidates_dir),
        require_merge_approval=True,
    )
    yield kg


@pytest.fixture
def app_client(kg_isolated, monkeypatch):
    """FastAPI TestClient with the route's KG accessor patched to use
    the isolated fixture."""
    from gaia_study.server import create_app
    app = create_app()

    # Patch the route's lazy KG accessor to return our isolated instance
    import gaia_study.server as srv_mod
    # Find the inner _get_world_kg closure — we can't reach it from
    # outside since it's nested. Instead, patch KnowledgeGraph() so the
    # endpoint constructs our isolated KG.
    from gaia_common.utils.knowledge_graph import KnowledgeGraph as _RealKG

    def _kg_factory(*args, **kwargs):
        # Honor any explicit args, but default to the fixture's paths
        return kg_isolated

    monkeypatch.setattr(
        "gaia_common.utils.knowledge_graph.KnowledgeGraph",
        _kg_factory,
    )
    monkeypatch.setattr(
        "gaia_study.server.KnowledgeGraph", _kg_factory, raising=False,
    )
    return TestClient(app)


def _create_pending_merge(kg) -> dict:
    """Helper: build two worlds + propose a merge, return the candidate."""
    kg.create_world(name="src_w", modality="actuality", parent="actuality")
    kg.create_world(name="tgt_w", modality="actuality", parent="actuality")
    kg.add_triple("Rupert", "is_a", "paladin", world="src_w")
    kg.add_triple("Rupert", "is_a", "paladin", world="tgt_w")
    return kg.propose_merge("src_w", "tgt_w", notes="test merge")


# ── GET /world_merges/pending ───────────────────────────────────────


class TestListPending:
    def test_empty_when_no_candidates(self, app_client):
        resp = app_client.get("/world_merges/pending")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["count"] == 0
        assert body["pending"] == []

    def test_returns_pending_only(self, kg_isolated, app_client):
        prop1 = _create_pending_merge(kg_isolated)
        # Approve the first so it's no longer pending
        kg_isolated.approve_merge(prop1["merge_id"])
        # Create a second still-pending
        kg_isolated.create_world(name="w3", modality="actuality", parent="actuality")
        kg_isolated.create_world(name="w4", modality="actuality", parent="actuality")
        kg_isolated.add_triple("X", "is", "Y", world="w3")
        kg_isolated.add_triple("X", "is", "Y", world="w4")
        prop2 = kg_isolated.propose_merge("w3", "w4")

        resp = app_client.get("/world_merges/pending")
        body = resp.json()
        assert body["count"] == 1
        assert body["pending"][0]["merge_id"] == prop2["merge_id"]

    def test_pending_payload_has_proposal_fields(self, kg_isolated, app_client):
        prop = _create_pending_merge(kg_isolated)
        resp = app_client.get("/world_merges/pending")
        item = resp.json()["pending"][0]
        # Schema sanity
        for field in ("merge_id", "source_world", "target_world",
                      "entity_mapping", "triples_to_rewrite",
                      "status", "proposed_at"):
            assert field in item, f"missing field {field!r}"
        assert item["status"] == "pending"


# ── GET /world_merges/{id} ──────────────────────────────────────────


class TestGetCandidate:
    def test_returns_candidate(self, kg_isolated, app_client):
        prop = _create_pending_merge(kg_isolated)
        resp = app_client.get(f"/world_merges/{prop['merge_id']}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["merge_id"] == prop["merge_id"]
        assert body["status"] == "pending"

    def test_404_when_missing(self, app_client):
        resp = app_client.get("/world_merges/m_does_not_exist")
        assert resp.status_code == 404


# ── POST /world_merges/{id}/approve ──────────────────────────────────


class TestApprove:
    def test_approve_flips_status(self, kg_isolated, app_client):
        prop = _create_pending_merge(kg_isolated)
        resp = app_client.post(
            f"/world_merges/{prop['merge_id']}/approve",
            json={"approver": "azrael"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["status"] == "approved"
        assert body["approved_by"] == "azrael"
        assert body["approved_at"]

    def test_approve_default_approver(self, kg_isolated, app_client):
        prop = _create_pending_merge(kg_isolated)
        resp = app_client.post(f"/world_merges/{prop['merge_id']}/approve")
        assert resp.status_code == 200
        # Falls back to "architect"
        assert resp.json()["approved_by"] == "architect"

    def test_approve_missing_returns_409(self, app_client):
        resp = app_client.post("/world_merges/m_nope/approve")
        assert resp.status_code == 409

    def test_approve_already_approved_409(self, kg_isolated, app_client):
        prop = _create_pending_merge(kg_isolated)
        app_client.post(f"/world_merges/{prop['merge_id']}/approve")
        # Second approve attempt → 409 (not pending)
        resp2 = app_client.post(f"/world_merges/{prop['merge_id']}/approve")
        assert resp2.status_code == 409


# ── POST /world_merges/{id}/reject ──────────────────────────────────


class TestReject:
    def test_reject_flips_status(self, kg_isolated, app_client):
        prop = _create_pending_merge(kg_isolated)
        resp = app_client.post(
            f"/world_merges/{prop['merge_id']}/reject",
            json={"reason": "coref looked off"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "rejected"
        assert body["rejected_reason"] == "coref looked off"
        assert body["rejected_at"]

    def test_reject_blocks_pending_after(self, kg_isolated, app_client):
        prop = _create_pending_merge(kg_isolated)
        app_client.post(f"/world_merges/{prop['merge_id']}/reject")
        # No longer in pending list
        resp = app_client.get("/world_merges/pending")
        assert resp.json()["count"] == 0


# ── Full lifecycle through HTTP ─────────────────────────────────────


class TestFullLifecycleHTTP:
    def test_propose_then_list_approve_get(self, kg_isolated, app_client):
        prop = _create_pending_merge(kg_isolated)
        mid = prop["merge_id"]

        # List shows it
        listed = app_client.get("/world_merges/pending").json()
        assert any(p["merge_id"] == mid for p in listed["pending"])

        # Approve
        approved = app_client.post(f"/world_merges/{mid}/approve").json()
        assert approved["status"] == "approved"

        # List no longer shows it (no longer pending)
        listed_after = app_client.get("/world_merges/pending").json()
        assert all(p["merge_id"] != mid for p in listed_after["pending"])

        # Detail still retrievable, status=approved
        detail = app_client.get(f"/world_merges/{mid}").json()
        assert detail["status"] == "approved"

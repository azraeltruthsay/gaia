"""Tests for World Model Stage 5 (8pk): merge mechanism + coreference.

Locks in:
  - Coref scoring (exact, case-insensitive, token-Jaccard)
  - propose_merge generates a structured proposal without changing the KG
  - apply_merge rewrites triples to target world and removes source
  - reverse_merge restores both worlds from snapshot
  - Refuses to merge actuality away
  - Refuses to apply / reverse in invalid states
  - Edge re-parenting: source's children become target's children
"""

import pytest


@pytest.fixture
def kg(tmp_path):
    """Default KG fixture for merge tests. require_merge_approval=False
    so the existing propose+apply tests run without the Stage 5b
    (clm) approval gate. The gate itself is exercised in
    test_world_merge_approval.py.

    Candidate dir still set under tmp_path so propose_merge can write
    its candidate JSON without polluting the project tree.
    """
    from gaia_common.utils.knowledge_graph import KnowledgeGraph
    return KnowledgeGraph(
        db_path=str(tmp_path / "kg.sqlite"),
        require_merge_approval=False,
        merge_candidates_dir=str(tmp_path / "candidates" / "world_merges"),
    )


class TestCorefScoring:
    def test_exact_match(self, kg):
        assert kg._coref_score("Rupert Roads", "Rupert Roads") == 1.0

    def test_case_insensitive_match(self, kg):
        assert kg._coref_score("Rupert Roads", "rupert roads") == 0.95

    def test_token_jaccard(self, kg):
        # Two tokens in common, one different — 2/3
        score = kg._coref_score("Rupert Roads", "Rupert Smith")
        assert 0.3 <= score <= 0.4

    def test_no_match(self, kg):
        assert kg._coref_score("Apple", "Zebra") == 0.0

    def test_empty_inputs(self, kg):
        assert kg._coref_score("", "Anything") == 0.0
        assert kg._coref_score("Anything", "") == 0.0


class TestPropose:
    def test_proposal_does_not_mutate_kg(self, kg):
        kg.create_world("source", modality="fiction", parent="actuality")
        kg.create_world("target", modality="fiction", parent="actuality")
        src_id = kg.get_world("source")["id"]
        tgt_id = kg.get_world("target")["id"]
        kg.add_triple("X", "rel", "Y", world=src_id)
        kg.add_triple("X", "rel", "Y", world=tgt_id)

        stats_before = kg.stats()
        kg.propose_merge("source", "target")
        stats_after = kg.stats()
        # World counts and triple counts unchanged
        assert stats_before["by_world"] == stats_after["by_world"]

    def test_proposal_identifies_coreference(self, kg):
        kg.create_world("alpha", modality="fiction", parent="actuality")
        kg.create_world("beta", modality="fiction", parent="actuality")
        a_id = kg.get_world("alpha")["id"]
        b_id = kg.get_world("beta")["id"]
        kg.add_triple("Rupert", "plays", "Artificer", world=a_id)
        kg.add_triple("Rupert", "sails", "ALICE", world=b_id)

        prop = kg.propose_merge("alpha", "beta")
        # rupert and alice are exact name matches between worlds
        assert "rupert" in prop["entity_mapping"]
        assert prop["entity_mapping"]["rupert"] == "rupert"

    def test_proposal_refuses_actuality_as_source(self, kg):
        kg.create_world("dummy", modality="fiction", parent="actuality")
        with pytest.raises(ValueError, match="actuality"):
            kg.propose_merge("actuality", "dummy")

    def test_proposal_refuses_self_merge(self, kg):
        kg.create_world("solo", modality="fiction", parent="actuality")
        with pytest.raises(ValueError, match="into itself"):
            kg.propose_merge("solo", "solo")

    def test_proposal_refuses_unknown_world(self, kg):
        kg.create_world("real", modality="fiction", parent="actuality")
        with pytest.raises(ValueError, match="not found"):
            kg.propose_merge("real", "imaginary")


class TestApply:
    def test_apply_moves_triples_to_target(self, kg):
        kg.create_world("src", modality="fiction", parent="actuality")
        kg.create_world("tgt", modality="fiction", parent="actuality")
        src_id = kg.get_world("src")["id"]
        tgt_id = kg.get_world("tgt")["id"]
        kg.add_triple("X", "y", "Z", world=src_id)
        kg.add_triple("Q", "r", "P", world=tgt_id)

        prop = kg.propose_merge("src", "tgt")
        kg.apply_merge(prop["merge_id"])

        s = kg.stats()
        assert src_id not in s["by_world"]
        assert s["by_world"].get(tgt_id) == 2

    def test_apply_removes_source_world(self, kg):
        kg.create_world("doomed", modality="fiction", parent="actuality")
        kg.create_world("survivor", modality="fiction", parent="actuality")
        prop = kg.propose_merge("doomed", "survivor")
        kg.apply_merge(prop["merge_id"])
        assert kg.get_world("doomed") is None
        assert kg.get_world("survivor") is not None

    def test_apply_marks_status_applied(self, kg):
        kg.create_world("a", modality="fiction", parent="actuality")
        kg.create_world("b", modality="fiction", parent="actuality")
        prop = kg.propose_merge("a", "b")
        kg.apply_merge(prop["merge_id"])
        record = kg.get_merge(prop["merge_id"])
        assert record["status"] == "applied"
        assert record["applied_at"] is not None

    def test_apply_refuses_already_applied(self, kg):
        kg.create_world("a", modality="fiction", parent="actuality")
        kg.create_world("b", modality="fiction", parent="actuality")
        prop = kg.propose_merge("a", "b")
        kg.apply_merge(prop["merge_id"])
        with pytest.raises(ValueError, match="not pending"):
            kg.apply_merge(prop["merge_id"])

    def test_apply_re_parents_children(self, kg):
        kg.create_world("middle", modality="fiction", parent="actuality")
        kg.create_world("target", modality="fiction", parent="actuality")
        kg.create_world("child", modality="fiction", parent="middle")

        mid_id = kg.get_world("middle")["id"]
        tgt_id = kg.get_world("target")["id"]
        child_id = kg.get_world("child")["id"]

        # Confirm child has 'middle' as parent
        ancestors = kg.world_ancestors("child")
        assert mid_id in ancestors

        prop = kg.propose_merge("middle", "target")
        kg.apply_merge(prop["merge_id"])

        # After merge, child's parent should now be target
        ancestors_after = kg.world_ancestors("child")
        assert tgt_id in ancestors_after
        assert mid_id not in ancestors_after


class TestReverse:
    def test_reverse_restores_source_world(self, kg):
        kg.create_world("doomed", modality="fiction", parent="actuality",
                        description="will come back")
        kg.create_world("survivor", modality="fiction", parent="actuality")
        kg.add_triple("X", "y", "Z", world=kg.get_world("doomed")["id"])

        prop = kg.propose_merge("doomed", "survivor")
        kg.apply_merge(prop["merge_id"])
        assert kg.get_world("doomed") is None

        kg.reverse_merge(prop["merge_id"])
        restored = kg.get_world("doomed")
        assert restored is not None
        assert restored["description"] == "will come back"

    def test_reverse_restores_triples(self, kg):
        kg.create_world("src", modality="fiction", parent="actuality")
        kg.create_world("tgt", modality="fiction", parent="actuality")
        src_id = kg.get_world("src")["id"]
        kg.add_triple("X", "y", "Z", world=src_id)

        prop = kg.propose_merge("src", "tgt")
        kg.apply_merge(prop["merge_id"])
        kg.reverse_merge(prop["merge_id"])

        # src should be back with its triple
        s = kg.stats()
        assert src_id in s["by_world"]
        assert s["by_world"][src_id] == 1

    def test_reverse_marks_status_reversed(self, kg):
        kg.create_world("a", modality="fiction", parent="actuality")
        kg.create_world("b", modality="fiction", parent="actuality")
        prop = kg.propose_merge("a", "b")
        kg.apply_merge(prop["merge_id"])
        kg.reverse_merge(prop["merge_id"])
        record = kg.get_merge(prop["merge_id"])
        assert record["status"] == "reversed"
        assert record["reversed_at"] is not None

    def test_reverse_refuses_unapplied(self, kg):
        kg.create_world("a", modality="fiction", parent="actuality")
        kg.create_world("b", modality="fiction", parent="actuality")
        prop = kg.propose_merge("a", "b")
        # Don't apply — try to reverse directly
        with pytest.raises(ValueError, match="applied merge"):
            kg.reverse_merge(prop["merge_id"])


class TestListAndGet:
    def test_list_returns_all_merges(self, kg):
        kg.create_world("w1", modality="fiction", parent="actuality")
        kg.create_world("w2", modality="fiction", parent="actuality")
        kg.create_world("w3", modality="fiction", parent="actuality")
        kg.propose_merge("w1", "w2")
        kg.propose_merge("w2", "w3")
        merges = kg.list_merges()
        assert len(merges) == 2

    def test_list_filter_by_status(self, kg):
        kg.create_world("a", modality="fiction", parent="actuality")
        kg.create_world("b", modality="fiction", parent="actuality")
        kg.create_world("c", modality="fiction", parent="actuality")
        p1 = kg.propose_merge("a", "b")  # pending
        p2 = kg.propose_merge("b", "c")  # pending
        kg.apply_merge(p1["merge_id"])    # applied

        pending = kg.list_merges(status="pending")
        applied = kg.list_merges(status="applied")
        assert len(pending) == 1
        assert len(applied) == 1
        assert pending[0]["merge_id"] == p2["merge_id"]
        assert applied[0]["merge_id"] == p1["merge_id"]

    def test_get_unknown_returns_none(self, kg):
        assert kg.get_merge("m_doesnotexist") is None

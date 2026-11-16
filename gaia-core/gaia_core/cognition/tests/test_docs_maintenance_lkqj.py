"""lkqj: docs maintenance re-drafted the same stale set every ~10 min (16,147
bundles by 2026-11-15), a model call per target and a fresh seed each time,
and the seed said "Review and apply", which a heartbeat turn read as an order
to write to system_reference and contracts. Drafts are suggestions only."""
import json
import os
import time

from gaia_core.cognition.sleep_tasks import docs_maintenance as dm

STALE = [{"target": "contracts/CONNECTIVITY.md", "reason": "x"},
         {"target": "knowledge/system_reference/AS_BUILT_LATEST.md", "reason": "y"}]


def _setup(tmp_path, monkeypatch, drafted):
    monkeypatch.setattr(dm, "DRAFTS_DIR", tmp_path / "drafts")
    monkeypatch.setattr(dm, "_check_doctor_dissonance", lambda: [])
    monkeypatch.setattr(dm, "_check_git_staleness", lambda: list(STALE))
    monkeypatch.setattr(dm, "_check_contract_freshness", lambda: [])
    monkeypatch.setattr(dm, "_draft_update", lambda e, mp, c: drafted.append(e["target"]) or {"target": e["target"]})
    seeds = []
    monkeypatch.setattr(dm, "_plant_thought_seed", lambda stale, drafts: seeds.append(len(stale)))
    return seeds


def test_unchanged_stale_set_is_not_drafted_again(tmp_path, monkeypatch):
    drafted = []
    seeds = _setup(tmp_path, monkeypatch, drafted)
    dm.run_docs_maintenance(config=None)
    assert len(drafted) == 2 and seeds == [2]
    r = dm.run_docs_maintenance(config=None)
    assert r["skipped"] == "unchanged"
    assert len(drafted) == 2 and seeds == [2]          # no model calls, no seed
    assert len(list((tmp_path / "drafts").glob("docs_draft_*.json"))) == 1


def test_changed_set_or_old_bundle_drafts_again(tmp_path, monkeypatch):
    drafted = []
    seeds = _setup(tmp_path, monkeypatch, drafted)
    dm.run_docs_maintenance(config=None)
    bundle = next((tmp_path / "drafts").glob("docs_draft_*.json"))
    old = time.time() - dm.DRAFT_REPEAT_AFTER_S - 60
    os.utime(bundle, (old, old))
    dm.run_docs_maintenance(config=None)
    assert len(drafted) == 4 and len(seeds) == 2


def test_seed_says_suggestions_for_review_not_apply(tmp_path, monkeypatch):
    import pathlib
    monkeypatch.setattr(dm, "Path", lambda p: tmp_path if p == "/knowledge/seeds" else pathlib.Path(p))
    dm._plant_thought_seed(STALE, [{"target": "contracts/CONNECTIVITY.md"}])
    seed = json.loads(next(tmp_path.glob("seed_*.json")).read_text())["seed"]
    assert "apply" not in seed.lower()
    assert "suggestions for Azrael to review" in seed

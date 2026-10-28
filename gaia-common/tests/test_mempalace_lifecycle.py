"""6afh: palace lifecycle — permanent wings and pinned memories stay forever;
the rest, once unused long enough, move to an archive that recall searches
only on request."""

import os
import time

from gaia_common.utils.mempalace import MemPalace

WINGS = {
    "identity": {"rooms": {"origin": ""}},
    "technical": {"rooms": {"architecture": ""}},
    "operational": {"rooms": {"milestones": "", "problems": ""}},
}


def _palace(tmp_path, **lifecycle):
    return MemPalace({"root_dir": str(tmp_path / "palace"), "wings": WINGS,
                      "kg_path": str(tmp_path / "kg.db"), "lifecycle": lifecycle})


def _memory(palace, rel, text, age_days):
    path = palace._root_dir / f"{rel}.aaak"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    old = time.time() - age_days * 86400
    os.utime(path, (old, old))
    return path


def test_sweep_keeps_permanent_pinned_and_recent(tmp_path):
    p = _palace(tmp_path, archive_after_days=30)
    _memory(p, "identity/origin/who", "GAIA was named by Azrael", 400)
    _memory(p, "operational/milestones/m1", "first GPU wake milestone", 400)
    _memory(p, "operational/problems/pinned", "pinned overlay problem", 400)
    _memory(p, "operational/problems/fresh", "fresh overlay problem", 5)
    _memory(p, "technical/architecture/old", "old NF4 quantization note", 400)
    p.pin("pinned")
    assert p.sweep(dry_run=True)["archived"] == ["technical/architecture/old.aaak"]
    out = p.sweep()
    assert out["ok"] and out["archived"] == ["technical/architecture/old.aaak"]
    assert (p._root_dir / "archive/technical/architecture/old.aaak").exists()
    assert p.status()["archived"] == 1 and p.status()["pinned"] == 1


def test_archived_memories_are_recalled_only_on_request(tmp_path):
    p = _palace(tmp_path, archive_after_days=30)
    _memory(p, "technical/architecture/old", "old NF4 quantization note", 400)
    p.sweep()
    assert p.recall("quantization")["results"] == []
    hit = p.recall("quantization", include_archive=True)["results"][0]
    assert hit["archived"] is True and hit["wing"] == "technical" and hit["room"] == "architecture"


def test_recall_counts_as_use_and_restore_brings_it_back(tmp_path):
    p = _palace(tmp_path, archive_after_days=30)
    used = _memory(p, "technical/architecture/used", "NF4 quantization note", 400)
    p.recall("quantization")
    assert time.time() - used.stat().st_mtime < 60
    assert p.sweep()["archived"] == []
    _memory(p, "technical/architecture/old", "old GGUF note", 400)
    p.sweep()
    assert p.restore("old")["ok"]
    assert (p._root_dir / "technical/architecture/old.aaak").exists()
    assert not p.restore("old")["ok"]


def test_unreadable_pin_file_archives_nothing(tmp_path):
    p = _palace(tmp_path, archive_after_days=30)
    _memory(p, "technical/architecture/old", "old note", 400)
    p._pins_path.write_text("{broken", encoding="utf-8")
    assert p.sweep()["archived"] == []

"""8awy: ingestion re-read the same cached OCW pages every cycle and queued
the same 2 pairs again (367 candidate files by 2026-11-15), each with a new
review seed reading "Fetched 0 pages". Pairs already queued, and questions
that are table fragments, are dropped; nothing new means no bundle, no seed."""
import json

from gaia_core.cognition.sleep_tasks import knowledge_ingestion as ki

TABLE_PAIR = {"instruction": "What are the key concepts covered in | LEC # | TOPICS AND NOTES |?",
              "output": "| 1 | Introduction |"}
GOOD_PAIR = {"instruction": "What is a threat model in computer security?",
             "output": "A description of what an attacker can do ..."}


def _process(tmp_path, pairs):
    raw = tmp_path / "raw"
    cand = tmp_path / "candidates"
    (raw / "c1").mkdir(parents=True, exist_ok=True)
    (raw / "c1" / "syllabus.txt").write_text("Threat models ...")  # cached, recent
    return ki._process_course(
        course_id="c1", max_pages=5, raw_dir=raw, candidate_dir=cand,
        check_interrupted=None, _mcp_call=None,
        fetch_course_page=lambda c, p: {"ok": False, "error": "offline"},
        get_course_manifest=lambda c: [{"page": "syllabus"}],
        generate_pairs_from_cfr=lambda *a: [],
        generate_pairs_from_raw=lambda text, md: [dict(p) for p in pairs],
        deduplicate_against_existing=lambda ps, path: ps,
        phase_a_courses=[],
    ), cand


def test_table_fragment_questions_are_dropped(tmp_path):
    assert ki._drop_queued_and_fragments([TABLE_PAIR, GOOD_PAIR], tmp_path) == [GOOD_PAIR]


def test_same_cached_pages_queue_once(tmp_path, monkeypatch):
    monkeypatch.setattr(ki, "_try_cfr_compression", lambda *a: [])
    first, cand = _process(tmp_path, [GOOD_PAIR, TABLE_PAIR])
    assert first["pair_count"] == 1 and first["pages_read"] == 1 and first["pages_fetched"] == 0
    assert len(list(cand.glob("*.json"))) == 1
    second, _ = _process(tmp_path, [GOOD_PAIR, TABLE_PAIR])
    assert second["pair_count"] == 0 and second["status"] == "nothing_new"
    assert len(list(cand.glob("*.json"))) == 1


def test_seed_wording_counts_pages_read(tmp_path, monkeypatch):
    monkeypatch.setattr(ki, "SEEDS_DIR", tmp_path)
    ki._plant_thought_seed(total_pairs=1, pages_fetched=0, pages_read=2,
                           course_titles=["Computer Systems Security"], results=[])
    seed = json.loads(next(tmp_path.glob("seed_*.json")).read_text())
    assert "Read 2 pages (0 newly fetched)" in seed["seed"]


# ── vjhn: CFR is called the way gaia-mcp's tools take it ─────────────────────

def test_cfr_ingest_gets_a_file_and_doc_id(tmp_path):
    from gaia_core.cognition.sleep_tasks.knowledge_ingestion import _try_cfr_compression
    calls = []

    def mcp(method, params):
        calls.append((method, params))
        return {"ok": True, "synthesis": "a summary"} if method == "cfr_synthesize" else {"ok": True}

    out = _try_cfr_compression({"syllabus": "x"}, "c1", mcp, tmp_path, prime_loaded=lambda: True)
    assert calls[0] == ("cfr_ingest", {"file_path": str(tmp_path / "syllabus.txt"), "doc_id": "ocw-c1-syllabus"})
    assert calls[1] == ("cfr_synthesize", {"doc_id": "ocw-c1-syllabus"})
    assert out == [{"title": "MIT OCW: c1 (syllabus)", "content": "a summary"}]


def test_cfr_skipped_while_prime_is_unloaded(tmp_path):
    from gaia_core.cognition.sleep_tasks.knowledge_ingestion import _try_cfr_compression

    def mcp(method, params):
        raise AssertionError("no MCP call while Prime is unloaded")

    assert _try_cfr_compression({"syllabus": "x"}, "c1", mcp, tmp_path, prime_loaded=lambda: False) == []


def test_cfr_error_falls_back(tmp_path):
    from gaia_core.cognition.sleep_tasks.knowledge_ingestion import _try_cfr_compression
    out = _try_cfr_compression({"syllabus": "x"}, "c1", lambda m, p: {"ok": False, "error": "no"},
                               tmp_path, prime_loaded=lambda: True)
    assert out == []


def test_course_pages_exist_on_ocw():
    from gaia_core.cognition.sleep_tasks.ocw_fetcher import get_course_manifest
    pages = [e["page"] for e in get_course_manifest("6-858-computer-systems-security-fall-2014")]
    assert "assignments" not in pages and "labs" in pages

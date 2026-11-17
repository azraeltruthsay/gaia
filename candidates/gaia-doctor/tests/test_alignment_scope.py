"""2026-11-17: only the whole battery sets her alignment tag; a section run is
kept apart, and the status survives a doctor restart."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cognitive_test_battery as ctb  # noqa: E402


def _fake_battery(monkeypatch, tmp_path, passes):
    monkeypatch.setattr(ctb, "RESULTS_PATH", str(tmp_path / "full.json"))
    monkeypatch.setattr(ctb, "PARTIAL_RESULTS_PATH", str(tmp_path / "partial.json"))
    monkeypatch.setattr(ctb, "_stage_vision_fixtures", lambda: None)
    monkeypatch.setattr(ctb, "_stage_audio_fixtures", lambda: None)
    tests = [{"id": f"t{i}", "section": "audio" if i < 2 else "identity", "prompt": "x",
              "validators": [{"type": "fake"}]} for i in range(4)]
    monkeypatch.setattr(ctb, "TEST_CASES", tests)
    monkeypatch.setattr(ctb, "query_model_direct", lambda *a, **k: "answer")
    monkeypatch.setattr(ctb, "run_validator", lambda *a, **k: (passes, ""))


def test_section_run_sets_no_tag(monkeypatch, tmp_path):
    _fake_battery(monkeypatch, tmp_path, passes=False)
    r = ctb.run_battery(endpoint="http://x", section="audio", target="core")
    assert r["scope"] == "partial" and r["alignment"] is None
    assert (tmp_path / "partial.json").exists() and not (tmp_path / "full.json").exists()


def test_full_run_sets_the_tag(monkeypatch, tmp_path):
    _fake_battery(monkeypatch, tmp_path, passes=True)
    r = ctb.run_battery(endpoint="http://x", target="core")
    assert r["scope"] == "full" and r["alignment"] == "SELF_ALIGNED"
    assert json.loads((tmp_path / "full.json").read_text())["mode"] == "direct:core"


def test_word_recall():
    ok, _ = ctb.validate_word_recall("Have you ever wondered what an artificial intelligence sees in the mirror",
                                     "wondered what it looks like when an artificial intelligence looks in the mirror")
    assert ok
    ok, detail = ctb.validate_word_recall("White noise.", "wondered what it looks like when an artificial intelligence")
    assert not ok and "0.00" in detail


def test_no_person_named():
    assert ctb.validate_no_person_named("Two people: a man and a woman, podcast hosts.")[0]
    assert ctb.validate_no_person_named("I can't tell who it is. It sounds like two hosts.")[0]
    assert not ctb.validate_no_person_named("The speaker is Azrael, who says hello.")[0]
    assert not ctb.validate_no_person_named("The speaker is Barack Obama.")[0]
    assert ctb.validate_no_person_named('The speaker is a male voice saying, "Yes, because today we look at GAIA."')[0]


def test_observe_only_not_counted(monkeypatch, tmp_path):
    _fake_battery(monkeypatch, tmp_path, passes=False)
    tests = list(ctb.TEST_CASES)
    for t in tests[:2]:
        t["observe_only"] = True
    monkeypatch.setattr(ctb, "run_validator", lambda v, resp, **k: (False, ""))
    r = ctb.run_battery(endpoint="http://x", target="core")
    assert r["summary"]["total"] == 2
    assert r["observe_only"]["total"] == 2

"""self_improve: gaps -> topics -> study -> training -> approval (2026-11-20)."""
import json

import pytest
import yaml

from gaia_core.cognition import learning_gaps, self_improve as si


@pytest.fixture
def shared(tmp_path, monkeypatch):
    monkeypatch.setenv("SHARED_DIR", str(tmp_path / "shared"))
    repo = tmp_path / "repo"
    (repo / si.CURRICULUM).mkdir(parents=True)
    (repo / si.CURRICULUM / "syllabus.yaml").write_text(
        "subject: PE\nslug: professional_engineering\nunits:\n  Fundamentals:\n    - {topic: Statics, url: \"https://en.wikipedia.org/wiki/Statics\"}\n")
    monkeypatch.setattr(si, "REPO", repo)
    monkeypatch.setattr(si, "STUDY_DIR", str(tmp_path / "study"))
    (tmp_path / "study").mkdir()
    (tmp_path / "study" / "pairs.jsonl").write_text("")
    return repo


def test_plan_and_page(monkeypatch):
    topics = si.plan_topics([{"question": "What is the Rational Method peak flow?", "kind": "not_found"}],
                            lambda m: "THOUGHT: hydrology.\nTOPICS: Rational method; Peak discharge")
    assert topics == ["Rational method", "Peak discharge"]

    def call(method, params, timeout=0):
        if method == "web_search":
            return {"result": {"results": [{"url": "https://example.com/x"}, {"url": "https://en.wikipedia.org/wiki/Peak"}]}}
        return {"result": {"ok": True, "content": "x" * 2000}}
    assert si.find_page("Peak discharge", call) == "https://en.wikipedia.org/wiki/Peak"


def test_syllabus_addition_is_valid_yaml(shared):
    n = si.add_to_syllabus([{"topic": "Rational method", "url": "https://en.wikipedia.org/wiki/R",
                             "why": 'She said "Q = CiA" wrong'}], shared / si.CURRICULUM / "syllabus.yaml")
    syl = yaml.safe_load((shared / si.CURRICULUM / "syllabus.yaml").read_text())
    unit = [k for k in syl["units"] if k.startswith("Self-directed")][0]
    assert n == 1 and syl["units"][unit][0]["topic"] == "Rational method"


def test_stages(shared, monkeypatch):
    for q in ("What is the rational method?", "What is a unit hydrograph?", "How is peak flow estimated?"):
        learning_gaps.note(q, "not_found", "s")
    monkeypatch.setattr(si, "find_page", lambda t, call=None: f"https://en.wikipedia.org/wiki/{t.replace(' ', '_')}")
    started = []
    monkeypatch.setattr(si, "_start_study", lambda st: started.append(1))
    st = si.tick(lambda m: "TOPICS: Rational method; Unit hydrograph")
    assert st["stage"] == "studying" and started and st["planned"] == ["Rational method", "Unit hydrograph"]
    # study produced enough pairs -> training requested
    monkeypatch.setattr(si, "_pairs_count", lambda: st["pairs_before"] + 25)
    calls = []
    monkeypatch.setattr(si, "_http", lambda method, url, body=None, timeout=30:
                        calls.append((method, url, body)) or {"ok": True})
    st = si.tick()
    assert st["stage"] == "training" and calls[-1][0] == "POST"
    # the run finished with a pass -> awaiting, approval requested once
    from gaia_common.utils import adapter_registry as ar
    monkeypatch.setattr(ar, "pending", lambda slot="engineering": {"name": "pe_v7"})
    monkeypatch.setattr(si, "_http", lambda method, url, body=None, timeout=30: {
        "running": False, "last_run": {"started": st["training_started"] + 1, "outcome": "proposed",
                                       "candidate": "pe_v7", "decision": {"action": "proposed", "verdict": ["pass  x"]}}})
    asked = []
    monkeypatch.setattr(si, "approver_session", lambda: "discord_dm_1")
    monkeypatch.setattr(si, "request_approval", lambda slot, name, verdict, session: asked.append(name) or "a1")
    st = si.tick()
    assert st["stage"] == "awaiting" and asked == ["pe_v7"]
    assert si.tick()["stage"] == "awaiting" and asked == ["pe_v7"]      # not asked again within the day

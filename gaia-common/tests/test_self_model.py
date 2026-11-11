"""Her model facts come from the machine and live in the world model (kki4)."""
import json

import pytest

from gaia_common.utils import self_model as sm
from gaia_common.utils.knowledge_graph import KnowledgeGraph


@pytest.fixture
def clean(tmp_path, monkeypatch):
    lineage = tmp_path / "LINEAGE.json"
    lineage.write_text(json.dumps({"Qwen3-VL-8B-GAIA-Prime-v1": {"name": "Qwen3-VL-8B", "history": "made so"}}))
    kg = KnowledgeGraph(str(tmp_path / "kg.sqlite3"))
    monkeypatch.setattr(sm, "LINEAGE_PATH", str(lineage))
    monkeypatch.setattr(sm, "_kg", kg)
    monkeypatch.setattr(sm, "_observed", {})
    monkeypatch.setattr(sm, "_loaded", False)
    return kg


def _facts(prime="Qwen3-VL-8B-GAIA-Prime-v1-abliterated-NF4", device="cuda"):
    return {"engines": {"Prime": {"model": prime, "device": device}}}


def _current(kg, predicate):
    return [f["object"] for f in kg.query_entity("gaia_prime", direction="outgoing")
            if f["predicate"] == predicate and f["current"]]


def test_the_name_comes_from_lineage_and_a_copy_shares_its_source_entry(clean):
    assert sm.minds(_facts())["Prime"]["name"] == "Qwen3-VL-8B"
    assert sm.lineage_of("Qwen3-VL-8B-GAIA-Prime-v1-abliterated-NF4")["history"] == "made so"


def test_a_new_model_nobody_described_still_gets_its_true_name(clean):
    assert sm.display_name("Phi-4-mini-GAIA-Core-v1") == "Phi-4-mini"
    assert sm.display_name("Phi-4-mini-Q8_0.gguf") == "Phi-4-mini-Q8_0"


def test_what_is_observed_goes_into_the_world_model(clean):
    sm.minds(_facts())
    facts = clean.query_entity("gaia_prime", direction="outgoing")
    assert all(f["world"] == "actuality" and f["source"] == "observed:engine_status" for f in facts)
    assert _current(clean, "runs_on_device")


def test_a_gear_shift_closes_the_old_fact_and_keeps_it(clean):
    sm.minds(_facts(device="cuda"))
    sm.minds(_facts(device="cpu"))
    device = [f for f in clean.query_entity("gaia_prime", direction="outgoing") if f["predicate"] == "runs_on_device"]
    assert len(device) == 2 and len([f for f in device if f["current"]]) == 1


def test_no_change_writes_nothing(clean):
    sm.minds(_facts())
    before = len(clean.query_entity("gaia_prime", direction="outgoing"))
    sm.minds(_facts())
    assert len(clean.query_entity("gaia_prime", direction="outgoing")) == before


def test_if_an_engine_cant_be_read_the_last_observation_stands_and_says_so(clean):
    sm.minds(_facts())
    prime = sm.minds({"engines": {}})["Prime"]
    assert prime["name"] == "Qwen3-VL-8B" and prime["live"] is False
    assert "not reachable" in sm.self_model_text({"engines": {}})


def test_a_new_process_remembers_from_the_world_model(clean, monkeypatch):
    sm.minds(_facts())
    monkeypatch.setattr(sm, "_observed", {})
    monkeypatch.setattr(sm, "_loaded", False)
    assert sm.minds({"engines": {}})["Prime"]["weights"].startswith("Qwen3-VL-8B")


def test_nothing_ever_read_says_nothing(clean):
    assert sm.self_model_text({"engines": {}}) == ""


def test_the_one_on_the_gpu_is_generating(clean):
    assert "Prime is Qwen3-VL-8B (on the GPU, generating)" in sm.self_model_text(_facts())

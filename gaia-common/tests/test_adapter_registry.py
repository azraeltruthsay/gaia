"""adapter_registry: decide on a new adapter; switch only when approved (2026-11-20)."""
import json

import pytest

from gaia_common.utils import adapter_registry as ar


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SHARED_DIR", str(tmp_path / "shared"))
    root = tmp_path / "adapters"
    monkeypatch.setattr(ar, "ADAPTER_ROOT", root)

    def make(name, before, after, web=None):
        d = root / name
        d.mkdir(parents=True)
        (d / "adapter_model.safetensors").write_bytes(b"x")
        (d / "eval_notes.json").write_text(json.dumps({"before": before, "after": after}))
        if web:
            (d / "eval_web.json").write_text(json.dumps({"summary": web}))
    return make


BASE = {"held_out_recall": 0.70, "unstudied_says_missing": 0.05, "ordinary_kept": 1.0}
WEB = {"v5": {"invents": 0.06, "backed": 0.58}, "v6": {"invents": 0.05, "backed": 0.60}}


def test_passing_adapter_waits_for_approval(env, monkeypatch):
    monkeypatch.delenv("ADAPTER_SWITCH_POLICY", raising=False)
    env("v6", BASE, {"held_out_recall": 0.76, "unstudied_says_missing": 0.9, "ordinary_kept": 0.6}, WEB)
    out = ar.consider("engineering", "v6", fallback_current="v5")
    assert out["action"] == "proposed" and ar.active("engineering", "v5") == "v5"
    assert ar.pending("engineering")["name"] == "v6"
    assert ar.activation_problem("engineering", "v6") is None
    assert ar.activate("engineering", "v6")["ok"] and ar.active("engineering") == "v6"
    assert ar.pending("engineering") is None
    assert ar.rollback("engineering", "Azrael", "test")["to"] == "v5"


def test_failing_adapter_is_kept_out(env):
    env("v7", BASE, {"held_out_recall": 0.71, "unstudied_says_missing": 0.9, "ordinary_kept": 0.6},
        {"v6": {"invents": 0.05, "backed": 0.60}, "v7": {"invents": 0.20, "backed": 0.40}})
    passed, verdict = ar.decide("v7", "v6")
    assert not passed and any(v.startswith("FAIL  recall gain") for v in verdict)
    assert any(v.startswith("FAIL  web invents") for v in verdict)
    assert ar.consider("engineering", "v7", fallback_current="v6")["action"] == "kept"
    assert "isn't an adapter waiting" in ar.activation_problem("engineering", "v7")
